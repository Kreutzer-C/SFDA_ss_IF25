import argparse
import os
import sys
import logging
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from batchgenerators.utilities.file_and_folder_operations import *
from data import data_helper
from model.model_factory import get_backbone, Classifier
from data.data_info import get_data_info
import wandb


def get_args():
    parser = argparse.ArgumentParser(description="Source Domains ERM Pre-train")
    # Experiment Name
    # (Determines where the results are saved, highly recommended to keep it different for each experiment)
    parser.add_argument("--exp", type=str, default="Source_ERM")

    # Device Setting
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    # Backbone Network Setting
    parser.add_argument("--backbone", default="resnet50",
                        help="Which backbone network to use, only resnet18 | resnet50 | resnet 101 are supported")

    # Dataset Setting (Only --dataset needs to be determined, the others will be configured automatically)
    parser.add_argument("--dataset", "-d", default="Officehome")
    parser.add_argument("--Domain_ID", default=[])
    parser.add_argument("--classes", default=[])
    parser.add_argument("--n_classes", type=int, default=7, help="Number of classes")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--src_classes", type=int, default=None, help="Number of source classes (ODA)")

    # Training Setting
    parser.add_argument("--data_path", default='./dataset', help="your data_path")
    parser.add_argument("--source", default=None, help="Specify single source domain")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")

    # Data Augmentation Setting
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscale")

    return parser.parse_args()


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.clip_feature_dim = 512

        self.oda = False
        if args.src_classes is not None and args.src_classes < args.n_classes:
            self.oda = True

        self.featurizer = get_backbone(args.backbone, self.clip_feature_dim).to(device)
        if self.oda:
            self.classifier = Classifier(self.clip_feature_dim, args.src_classes).to(device)
        else:
            self.classifier = Classifier(self.clip_feature_dim, args.n_classes).to(device)
        self.model = nn.Sequential(
            self.featurizer,
            self.classifier
        )

        if self.oda:
            self.source_loader, self.val_loader = data_helper.get_ODA_train_dataloader(args)
            logging.info("Using ODA-Setting train dataloader")
        else:
            self.source_loader, self.val_loader = data_helper.get_train_dataloader(args)
        logging.info("Dataset size: train %d, val %d" % (
            len(self.source_loader.dataset), len(self.val_loader.dataset)))

        self.optimizer = optim.SGD(self.model.parameters(), weight_decay=0.0005, momentum=0.9, lr=args.learning_rate)
        # 线性预热+余弦退火衰减调度器
        warmup_epochs = int(args.epochs * args.warmup_ratio)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=args.epochs - warmup_epochs + 1,  # 余弦退火周期
            T_mult=1,  # 周期倍数
            eta_min=0.0  # 最小学习率
        )
        self.warmup_epochs = warmup_epochs
        self.base_lr = args.learning_rate
        self.criterion = nn.CrossEntropyLoss()

        self.current_epoch = None

        wandb.init(
            project="SFDA_ss_IF25",
            name=f"{args.exp}: {args.dataset}_{args.source}",
            tags=[args.dataset, args.source],
            config=vars(args)
        )

    def _do_epoch(self):
        self.model.train()
        for iter, ((data, class_l), d_idx) in enumerate(self.source_loader):
            data, class_l = data.to(self.device), class_l.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)

            # ERM
            loss = self.criterion(outputs, class_l)
            loss.backward()
            self.optimizer.step()

            iter_num = iter + self.current_epoch * len(self.source_loader)
            wandb.log({
                'train/loss': loss.item(),
                'train/step': iter_num
            })

            if iter % 10 == 0:
                logging.info("iter {}/{} loss: {:.6f}".format(iter, len(self.source_loader), loss.item()))

        self.model.eval()
        correct_predictions = 0
        total_samples = 0
        for iter, ((data, class_l), d_idx) in enumerate(self.val_loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            with torch.no_grad():
                outputs = self.model(data)
            pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            correct_predictions += (pred == class_l).sum().item()
            total_samples += class_l.size(0)

        accuracy = correct_predictions / total_samples
        return accuracy

    def _update_learning_rate(self, epoch):
        """更新学习率：线性预热 + 余弦退火"""
        if epoch < self.warmup_epochs:
            # 线性预热阶段
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # 余弦退火阶段
            self.scheduler.step()

    def do_training(self):
        best_acc = 0.0
        for self.current_epoch in tqdm(range(self.args.epochs)):
            accuracy = self._do_epoch()
            self._update_learning_rate(self.current_epoch)

            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f'Epoch: {self.current_epoch} Val Accuracy: {accuracy:.4f} LR: {current_lr:.6f}')
            wandb.log({
                'common/val_acc': accuracy,
                'common/lr': current_lr,
                'common/step': self.current_epoch
            })

            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(self.model.state_dict(),
                           os.path.join(self.args.output_folder, f"{self.args.source}_best.pth"))
                logging.info(f'NEW Val_best model checkpoint have been saved')
        wandb.finish()


def ERM_pretrain():
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_num
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:" + args.GPU_num if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        GPU_device = torch.cuda.get_device_properties(device)
        print(">>> Device:{}({},{}MB)".format(device, GPU_device.name, GPU_device.total_memory / 1024 ** 2))

    get_data_info(args)

    # 如果指定了source，只训练指定的域；否则训练所有域
    domains_to_train = [args.source] if args.source is not None else args.Domain_ID
    
    for domain in domains_to_train:
        args.source = domain

        args.output_folder = join(os.getcwd(), 'results', args.exp, args.backbone, args.dataset, args.source)
        if os.path.exists(args.output_folder):
            raise ValueError(f"Output path {args.output_folder} existed, please specify new one by setting <--exp>")
        else:
            maybe_mkdir_p(args.output_folder)
            print(">>> Output results will be saved at: {}".format(args.output_folder))

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=args.output_folder + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        logging.info("\n****************************")
        for key, value in vars(args).items():
            logging.info(f"{key}: {value}")
        logging.info("****************************\n")

        logging.info(f">>> Training on dataset {args.dataset} with source domain: {args.source}")

        trainer = Trainer(args, device)
        trainer.do_training()


if __name__ == "__main__":
    ERM_pretrain()
