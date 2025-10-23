"""
SFDA Test:
Test the target model performance after SFDA adaptation(manually)
"""

import argparse
import os
import sys
import shutil
import logging
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import pickle
from batchgenerators.utilities.file_and_folder_operations import *
from data import data_helper
from model.model_factory import get_backbone, Classifier
from data.data_info import get_data_info


def get_args():
    parser = argparse.ArgumentParser(description="SFDA Test")
    # Experiment Name
    # (The experiment name of SFDA adaptation stage)
    parser.add_argument("--exp", type=str, default="SFDA_Test")

    # Device Setting
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=0, help="seed")

    # Backbone Network Setting
    parser.add_argument("--backbone", default="resnet50",
                        help="Which backbone network to use, only resnet18 | resnet50 | resnet 101 are supported")

    # Dataset Setting (Only --dataset needs to be determined, the others will be configured automatically)
    parser.add_argument("--dataset", "-d", default="Officehome")
    parser.add_argument("--Domain_ID", default=[])
    parser.add_argument("--classes", default=[])
    parser.add_argument("--n_classes", type=int, default=7, help="Number of classes")

    # Training Setting
    parser.add_argument("--data_path", default='./dataset', help="your data_path")
    parser.add_argument("--model_path", default="/opt/data/private/SFDA_ss_IF25_results", help="path to the adapted model")
    parser.add_argument("--source", "-src", required=True, help="Specify single source domain")
    parser.add_argument("--target", "-tgt", required=True, help="Specify single target domain")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")

    # Data Augmentation Setting
    parser.add_argument("--image_size", type=int, default=224, help="Image size")

    parser.add_argument("--ckpt", default=None, help="path to the checkpoint file(manually specified)")

    return parser.parse_args()


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.clip_feature_dim = 512

        self.featurizer = get_backbone(args.backbone, self.clip_feature_dim).to(device)
        self.classifier = Classifier(self.clip_feature_dim, args.n_classes).to(device)
        self.model = nn.Sequential(
            self.featurizer,
            self.classifier
        )
        if self.args.ckpt is not None:
            self.checkpoint_path = self.args.ckpt
        else:
            self.checkpoint_path = join(self.args.model_path, self.args.dataset, f"{self.args.source[:2]}_to_{self.args.target[:2]}", "best_targetmodel.pth")
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        logging.info(f">>> Loading adapted model from: {self.checkpoint_path}")

        self.test_loader = data_helper.get_test_dataloader(args)
        logging.info("Dataset size: test %d" % (len(self.test_loader.dataset)))

    def do_test(self):
        self.model.eval()
        correct_predictions = 0
        total_samples = 0

        # 为VisDA数据集添加每个类别的准确率统计
        if self.args.dataset == "VisDA":
            class_correct = [0] * self.args.n_classes
            class_total = [0] * self.args.n_classes

        # 用于保存featurizer输出和标签的列表
        all_features = []
        all_labels = []

        for iter, ((data, class_l), d_idx) in tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc="Testing"):
            data, class_l = data.to(self.device), class_l.to(self.device)
            with torch.no_grad():
                # 获取featurizer的输出
                z = self.featurizer(data)
                outputs = self.classifier(z)
                pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            
            # 保存featurizer输出和标签
            all_features.append(z.cpu().numpy())
            all_labels.append(class_l.cpu().numpy())
            
            correct_predictions += (pred == class_l).sum().item()
            total_samples += class_l.size(0)

            # 为VisDA数据集统计每个类别的准确率
            if self.args.dataset == "VisDA":
                for i in range(class_l.size(0)):
                    label = class_l[i].item()
                    class_total[label] += 1
                    if pred[i] == class_l[i]:
                        class_correct[label] += 1

        # 合并所有特征和标签
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # 保存特征和标签到output_folder
        features_path = os.path.join(self.args.output_folder, 'features.npy')
        labels_path = os.path.join(self.args.output_folder, 'labels.npy')
        np.save(features_path, all_features)
        np.save(labels_path, all_labels)
        logging.info(f"Features saved to: {features_path}")
        logging.info(f"Labels saved to: {labels_path}")
        logging.info(f"Features shape: {all_features.shape}, Labels shape: {all_labels.shape}")

        accuracy = correct_predictions / total_samples
        logging.info(f'Task: {self.args.source} --> {self.args.target} Accuracy: {accuracy:.4f}')
        
        # 为VisDA数据集输出每个类别的准确率
        if self.args.dataset == "VisDA":
            logging.info("Per-class accuracy:")
            for i in range(self.args.n_classes):
                if class_total[i] > 0:
                    class_acc = class_correct[i] / class_total[i]
                    logging.info(f'  {self.args.classes[i]}: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})')
                else:
                    logging.info(f'  {self.args.classes[i]}: N/A (0 samples)')
        logging.info(">>>=====================<<<\n")


def GSFDA_test():
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

    args.output_folder = join(os.getcwd(), 'results', args.exp, args.backbone, args.dataset, f"{args.source}_to_{args.target}")
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

    logging.info(f">>> Test {args.source} --> {args.target}")

    trainer = Trainer(args, device)
    trainer.do_test()


if __name__ == "__main__":
    GSFDA_test()
