"""
Test the OD performance of the source ERM pre-trained model, then register it for further SFDA task
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
from batchgenerators.utilities.file_and_folder_operations import *
from data import data_helper
from model.model_factory import get_backbone, Classifier
from data.data_info import get_data_info


def get_args():
    parser = argparse.ArgumentParser(description="Test for Source Domains ERM Pre-train")
    # Experiment Name
    # (The experiment name of ERM pre-train stage)
    parser.add_argument("--exp", type=str, required=True)

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
    parser.add_argument("--src_classes", type=int, default=None, help="Number of source classes (ODA)")

    # Training Setting
    parser.add_argument("--data_path", default='./dataset', help="your data_path")
    parser.add_argument("--source", default=None, help="Specify single source domain")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")

    # Data Augmentation Setting
    parser.add_argument("--image_size", type=int, default=224, help="Image size")

    parser.add_argument("--regis", action="store_true", help="Whether to register the model")

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
        self.checkpoint_path = join(self.args.output_folder, f'{args.source}_best.pth')
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        logging.info(f">>> Loading source model from: {self.checkpoint_path}")

        if args.src_classes is not None and args.src_classes < args.n_classes:
            self.test_loader = data_helper.get_ODA_test_dataloader(args)
            logging.info("Using ODA-Setting test dataloader")
        else:
            self.test_loader = data_helper.get_test_dataloader(args)
        logging.info("Dataset size: OOD test %d" % (len(self.test_loader.dataset)))

    def do_test(self):
        self.model.eval()
        correct_predictions = 0
        total_samples = 0
        for iter, ((data, class_l), d_idx) in tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc="Testing"):
            data, class_l = data.to(self.device), class_l.to(self.device)
            with torch.no_grad():
                outputs = self.model(data)
            prob = torch.softmax(outputs, dim=1)
            pred = torch.argmax(prob, dim=1)
            ent = torch.sum(-prob * torch.log(prob + 1e-5), dim=1) / np.log(self.args.n_classes)
            ent = ent.detach().cpu().numpy()

            from sklearn.cluster import KMeans
            kmeans = KMeans(2, random_state=0, n_init='auto').fit(ent.reshape(-1,1))
            labels = kmeans.predict(ent.reshape(-1,1))
            idx = np.where(labels==1)[0]
            iidx = 0
            if ent[idx].mean() > ent.mean():
                iidx = 1
            pred[np.where(labels==iidx)[0]] = self.args.n_classes
            correct_predictions += (pred == class_l).sum().item()
            total_samples += class_l.size(0)

        accuracy = correct_predictions / total_samples
        logging.info(f'Task: {self.args.source} --> {self.args.target} Accuracy: {accuracy:.4f}')
        logging.info(">>>=====================<<<\n")

    def do_regis(self):
        copy_target_path = join(os.getcwd(), 'pretrain', f'ERM_ODA{self.args.src_classes}', self.args.backbone, self.args.dataset)
        maybe_mkdir_p(copy_target_path)
        shutil.copy(self.checkpoint_path,
                    join(copy_target_path, f'{self.args.source}_best.pth'))


def ERM_test():
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

    # 如果指定了source，只测试指定的域；否则测试所有域
    domains_to_test = [args.source] if args.source is not None else args.Domain_ID

    for domain in domains_to_test:
        args.source = domain
        for target in args.Domain_ID:
            if target == args.source:
                continue
            args.target = target

            args.output_folder = join(os.getcwd(), 'results', args.exp, args.backbone, args.dataset, args.source)
            if not os.path.exists(args.output_folder):
                raise ValueError(f"Path does not exist: {args.output_folder}, Please first run train_source.py")
                # args.output_folder = input(f"Please input the path of {args.target} ERM pre-train output results: ")
            else:
                print(">>> Using ERM pre-train model saved at: {}".format(args.output_folder))

            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(filename=args.output_folder + "/log.txt", level=logging.INFO,
                                format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

            logging.info("\n****************************")
            for key, value in vars(args).items():
                logging.info(f"{key}: {value}")
            logging.info("****************************\n")

            logging.info(f">>> Test source model on target domain: {args.source} --> {args.target}")

            trainer = Trainer(args, device)
            trainer.do_test()
            if args.regis:
                trainer.do_regis()


if __name__ == "__main__":
    ERM_test()
