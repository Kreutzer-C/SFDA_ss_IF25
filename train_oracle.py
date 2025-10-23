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


def get_args():
    parser = argparse.ArgumentParser(description="Oracle Training with Target Domain Labels")
    # Experiment Name
    parser.add_argument("--exp", type=str, default="Oracle_Training")

    # Device Setting
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    # Backbone Network Setting
    parser.add_argument("--backbone", default="resnet50", help="target model arch, only resnet18 | resnet50 | resnet 101 are supported")

    # Dataset Setting
    parser.add_argument("--dataset", "-d", default="Officehome")
    parser.add_argument("--Domain_ID", default=[])
    parser.add_argument("--classes", default=[])
    parser.add_argument("--n_classes", type=int, default=7, help="Number of classes")

    # Training Setting
    parser.add_argument("--data_path", default='./dataset', help="your data_path")
    parser.add_argument("--source", "-src", default=None, help="Specify single source domain")
    parser.add_argument("--target", "-tgt", default=None, help="Specify single target domain")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay")

    # Data Augmentation Setting
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--aug", action='store_true', help="Whether to use data augmentation")
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscale")
    
    # Validation Setting
    parser.add_argument("--val_size", default=0.1, type=float, help="Validation set ratio")
    
    return parser.parse_args()


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.n_classes = args.n_classes
        self.clip_feature_dim = 512  # For ResNet backbone

        # 创建数据加载器
        self.train_loader, self.val_loader = data_helper.get_oracle_train_dataloader(args)
        logging.info(f"Dataset size: train {len(self.train_loader.dataset)}, val {len(self.val_loader.dataset)}")

        # 定义模型
        self.featurizer = get_backbone(args.backbone, self.clip_feature_dim).to(device)
        self.classifier = Classifier(self.clip_feature_dim, args.n_classes).to(device)
        self.model = nn.Sequential(self.featurizer, self.classifier)
        self.model.load_state_dict(torch.load(args.checkpoint_path))
        logging.info(f">>> Loading source model from: {args.checkpoint_path}")

        # 定义优化器和损失函数
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate, 
                                   weight_decay=args.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)

        self.best_val_acc = 0.0
        self.current_epoch = 0

        wandb.init(
            project="SFDA_ss_IF25",
            name=f"{args.exp}: {args.dataset}_{args.source}_to_{args.target}",
            tags=[args.dataset, args.source, args.target],
            config=vars(args)
        )

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, ((data, target), _) in enumerate(tqdm(self.train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct_predictions += (pred == target).sum().item()
            total_samples += target.size(0)

        avg_loss = total_loss / len(self.train_loader)
        train_acc = correct_predictions / total_samples
        
        return avg_loss, train_acc

    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, ((data, target), _) in enumerate(tqdm(self.val_loader, desc="Validating")):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, target)

                total_loss += loss.item()
                pred = torch.argmax(outputs, dim=1)
                correct_predictions += (pred == target).sum().item()
                total_samples += target.size(0)

        avg_loss = total_loss / len(self.val_loader)
        val_acc = correct_predictions / total_samples
        
        return avg_loss, val_acc

    def train(self):
        """主训练循环"""
        for epoch in range(self.args.epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录日志
            logging.info(f'Epoch {epoch+1}/{self.args.epochs}: '
                       f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # 记录到wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                logging.info(f"New best validation accuracy: {val_acc:.4f}")
                torch.save(self.model.state_dict(), 
                          join(self.args.output_folder, 'best_model.pth'))

    def test(self):
        """在测试集上评估模型"""
        # 加载最佳模型
        self.model.load_state_dict(torch.load(join(self.args.output_folder, 'best_model.pth')))
        
        # 创建测试数据加载器
        test_loader = data_helper.get_test_dataloader(self.args)
        
        self.model.eval()
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, ((data, target), _) in enumerate(tqdm(test_loader, desc="Testing")):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                pred = torch.argmax(outputs, dim=1)
                correct_predictions += (pred == target).sum().item()
                total_samples += target.size(0)

        test_acc = correct_predictions / total_samples
        logging.info(f"Test Accuracy: {test_acc:.4f}")
        
        wandb.log({
            'test_acc': test_acc,
            'best_val_acc': self.best_val_acc
        })
        wandb.finish()

def train_oracle():
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

    # 设置输出文件夹
    args.output_folder = join(os.getcwd(), 'results', args.dataset, f"{args.source}_to_{args.target}", args.exp)
    if os.path.exists(args.output_folder):
        raise ValueError(f"Output path {args.output_folder} existed, please specify new one by setting <--exp>")
    else:
        maybe_mkdir_p(args.output_folder)
        print(">>> Output results will be saved at: {}".format(args.output_folder))

    args.checkpoint_path = join(os.getcwd(), 'pretrain', 'ERM', args.backbone, args.dataset, f"{args.source}_best.pth")

    # 设置日志
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.output_folder + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("\n****************************")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    logging.info("****************************\n")

    logging.info(f">>> Oracle Training on {args.dataset} target domain: {args.target}")

    trainer = Trainer(args, device)
    trainer.train()
    trainer.test()


if __name__ == '__main__':
    train_oracle()
