import argparse
import os
import sys
import logging
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import wandb
from batchgenerators.utilities.file_and_folder_operations import *

from data import data_helper
from data.data_info import get_data_info
from model.model_factory import get_backbone, Classifier
from utils.loss import *


def get_args():
    parser = argparse.ArgumentParser(description="Source Model Self-Training")
    # Experiment Name
    # (Determines where the results are saved, highly recommended to keep it different for each experiment)
    parser.add_argument("--exp", type=str, default="Source_Self-Training")

    # Device Setting
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    # Backbone Network Setting
    parser.add_argument("--backbone", default="resnet50", help="target model arch, only resnet18 | resnet50 | resnet 101 are supported")

    # Dataset Setting (Only --dataset needs to be determined, the others will be configured automatically)
    parser.add_argument("--dataset", "-d", default="Officehome")
    parser.add_argument("--Domain_ID", default=[])
    parser.add_argument("--classes", default=[])
    parser.add_argument("--n_classes", type=int, default=7, help="Number of classes")

    # Training Setting (two learning rates may need to be adjusted depending on the specific task)
    parser.add_argument("--data_path", default='./dataset', help="your data_path")
    parser.add_argument("--source", "-src", default=None, help="Specify single source domain")
    parser.add_argument("--target", "-tgt", default=None, help="Specify single target domain")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate_tar", "-lr1", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")

    # Data Augmentation Setting
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--aug", action='store_true', help="Whether to use data augmentation")
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
        self.classes = args.classes
        self.n_classes = args.n_classes
        self.clip_feature_dim = 512  # For ViT-B/16 or ViT-B/32

        # Dataloader
        self.target_loader = data_helper.get_adapt_dataloader(args)
        logging.info("Dataset size: OOD(target) %d" % (len(self.target_loader.dataset)))
        self.max_iter = args.epochs * len(self.target_loader)

        # Define target model and initialize source domain pre-train parameters
        self.featurizer = get_backbone(args.backbone, self.clip_feature_dim).to(device)
        self.classifier = Classifier(self.clip_feature_dim, args.n_classes).to(device)
        for param in self.classifier.parameters():
            param.requires_grad = False
        self.model = nn.Sequential(
            self.featurizer,
            self.classifier
        )
        self.model.load_state_dict(torch.load(args.checkpoint_path))
        logging.info(f">>> Loading source model from: {args.checkpoint_path}")

        # Define optimizers and schedulers
        self.optimizer_tar = optim.AdamW(self.model.parameters(), lr=args.learning_rate_tar,
                                         betas=(0.9, 0.999))
        self.warmup_iters = int(self.max_iter * args.warmup_ratio)
        self.scheduler_tar = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_tar, 
            T_0=self.max_iter - self.warmup_iters + 1,  # 余弦退火周期
            T_mult=1,  # 周期倍数
            eta_min=0.0  # 最小学习率
        )

        self.criterion = nn.CrossEntropyLoss()

        self.best_epoch_acc = 0.0
        self.best_vlm_acc = 0.0
        self.current_epoch = None

        wandb.init(
            project="SFDA_ss_IF25",
            name=f"{args.exp}: {args.dataset}_{args.source[:2]}_to_{args.target[:2]}",
            tags=[args.dataset, args.source, args.target, "Source_Self-Training"],
            config=vars(args)
        )

    def _update_learning_rate(self, optimizer, scheduler, base_lr, iter_num):
        """更新学习率：线性预热 + 余弦退火"""
        if iter_num < self.warmup_iters:
            # 线性预热阶段
            lr = base_lr * (iter_num + 1) / self.warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # 余弦退火阶段
            scheduler.step()

    def _do_epoch(self):
        # Stage2: Self-training using y_tar as pseudo labels
        logging.info("\nDoing Target Model Self-Training...")
        self.featurizer.train()
        self.classifier.train()
        correct_predictions = 0
        total_samples = 0
        
        for iteration, ((img, class_l), d_idx) in enumerate(self.target_loader):
            img, class_l = img.to(self.device), class_l.to(self.device)

            self.optimizer_tar.zero_grad()
            
            # 前向传播
            z = self.featurizer(img)
            out_tar = self.classifier(z)
            p_tar = torch.softmax(out_tar, dim=1)
            y_tar = torch.argmax(p_tar, dim=1)  # 伪标签
            
            # 使用y_tar作为伪标签进行自训练，计算交叉熵损失
            loss_self_training = self.criterion(out_tar, y_tar)

            loss_self_training.backward()
            self.optimizer_tar.step()

            if iteration % (len(self.target_loader) // 3) == 0:
                logging.info("iter {}/{} loss_self_training: {:.6f}"
                             .format(iteration, len(self.target_loader), loss_self_training.item()))

            iter_num = iteration + self.current_epoch * len(self.target_loader)
            self._update_learning_rate(self.optimizer_tar, self.scheduler_tar, self.args.learning_rate_tar, iter_num)
            wandb.log({
                'stage2_self_training/loss_self_training': loss_self_training,
                'stage2_self_training/lr_tar': self.optimizer_tar.param_groups[0]['lr'],
                'stage2_self_training/step': iter_num
            })

            y_pred_tar = torch.argmax(p_tar, dim=1)
            correct_predictions += (y_pred_tar == class_l).sum().item()
            total_samples += class_l.size(0)

        tar_accuracy = correct_predictions / total_samples
        wandb.log({
            'common/tar_acc': tar_accuracy,
            'common/step': self.current_epoch
        })  
        return tar_accuracy, 0.0  # 返回0.0作为vlm_accuracy的占位符

    def do_training(self):
        for self.current_epoch in tqdm(range(self.args.epochs)):
            tar_accuracy, vlm_accuracy = self._do_epoch()

            if tar_accuracy >= self.best_epoch_acc:
                self.best_epoch_acc = tar_accuracy
                logging.info("\n*************NEW BEST!************")
                logging.info(f'Epoch: {self.current_epoch} Accuracy: {tar_accuracy:.4f}\n')
                target_save_path = join(self.args.output_folder, f'best_targetmodel.pth')
                torch.save(self.model.state_dict(), target_save_path)
            else:
                logging.info(f'Epoch: {self.current_epoch} Accuracy: {tar_accuracy:.4f}\n')

            if vlm_accuracy >= self.best_vlm_acc:
                self.best_vlm_acc = vlm_accuracy

    def do_test(self):
        logging.info("\n>>>=====================<<<")
        logging.info(f">>>Testing SFDA task: {self.args.source} --> {self.args.target}")

        self.model.load_state_dict(torch.load(join(self.args.output_folder, f'best_targetmodel.pth')))
        self.model.eval()
        correct_predictions = 0
        total_samples = 0
        self.test_loader = data_helper.get_test_dataloader(self.args)

        for _, ((data, class_l), d_idx) in enumerate(self.test_loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            with torch.no_grad():
                outputs = self.model(data)
            pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            correct_predictions += (pred == class_l).sum().item()
            total_samples += class_l.size(0)

        test_accuracy = correct_predictions / total_samples

        logging.info(f'Accuracy: {self.best_epoch_acc:.4f} (with best vlm_acc={self.best_vlm_acc:.4f})')
        logging.info(f"Test accuracy = {test_accuracy:.4f}")
        logging.info(">>>=====================<<<\n")
        wandb.log({"metrics/test_acc": test_accuracy,
                   "metrics/best_epoch_acc": self.best_epoch_acc,
                   "metrics/best_vlm_acc": self.best_vlm_acc})
        wandb.finish()


def train_adaptation():
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

    # 如果指定了source和target，只训练指定的域；否则训练所有域
    if args.source is not None and args.target is not None:
        if args.source not in args.Domain_ID:
            raise ValueError(f"Source domain {args.source} not included in domains of {args.dataset}: {args.Domain_ID}")
        if args.target not in args.Domain_ID:
            raise ValueError(f"Target domain {args.target} not included in domains of {args.dataset}: {args.Domain_ID}")

        args.output_folder = join(os.getcwd(), 'results', args.dataset, f"{args.source}_to_{args.target}", args.exp)
        if os.path.exists(args.output_folder):
            raise ValueError(f"Output path {args.output_folder} existed, please specify new one by setting <--exp>")
        else:
            maybe_mkdir_p(args.output_folder)
            print(">>> Output results will be saved at: {}".format(args.output_folder))
        
        args.checkpoint_path = join(os.getcwd(), 'pretrain', 'ERM', args.backbone, args.dataset, f"{args.source}_best.pth")

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=args.output_folder + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        logging.info("\n****************************")
        for key, value in vars(args).items():
            logging.info(f"{key}: {value}")
        logging.info("****************************\n")

        logging.info(f">>> Training on {args.dataset} with pre-trained source domain: {args.source}")
        logging.info(f">>> Adapting to target domain: {args.target}")

        trainer = Trainer(args, device)
        trainer.do_training()
        trainer.do_test()

    else:
        for src_domain in args.Domain_ID:
            for tgt_domain in args.Domain_ID:
                if src_domain == tgt_domain:
                    continue
                args.source = src_domain
                args.target = tgt_domain

                args.output_folder = join(os.getcwd(), 'results', args.dataset, f"{args.source}_to_{args.target}", args.exp)
                if os.path.exists(args.output_folder):
                    raise ValueError(f"Output path {args.output_folder} existed, please specify new one by setting <--exp>")
                else:
                    maybe_mkdir_p(args.output_folder)
                    print(">>> Output results will be saved at: {}".format(args.output_folder))
                
                args.checkpoint_path = join(os.getcwd(), 'pretrain', 'ERM', args.backbone, args.dataset, f"{args.source}_best.pth")

                for handler in logging.root.handlers[:]:
                    logging.root.removeHandler(handler)
                logging.basicConfig(filename=args.output_folder + "/log.txt", level=logging.INFO,
                                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
                logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

                logging.info("\n****************************")
                for key, value in vars(args).items():
                    logging.info(f"{key}: {value}")
                logging.info("****************************\n")

                logging.info(f">>> Training on {args.dataset} with pre-trained source domain: {args.source}")
                logging.info(f">>> Adapting to target domain: {args.target}")

                trainer = Trainer(args, device)
                trainer.do_training()
                trainer.do_test()


if __name__ == '__main__':
    train_adaptation()
