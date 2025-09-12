import argparse
import sys
import logging
from tqdm import tqdm
import torch
import clip
from batchgenerators.utilities.file_and_folder_operations import *
from data import data_helper
from data.data_info import get_data_info


def get_args():
    parser = argparse.ArgumentParser(description="CLIP zero-shot inference")
    # Experiment Name
    parser.add_argument("--exp", type=str, default="CLIP_zs",
                        help="Determines where the results are saved, highly recommended to keep it different for each experiment")

    # Device Setting
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # Backbone Network Setting
    parser.add_argument("--CLIP_backbone", default="ViT-B/16",
                        help="CLIP model vision encoder arch, only ViT-B/16 | ViT-B/32 are supported")

    # Dataset Setting (Only --dataset needs to be determined, the others will be configured automatically)
    parser.add_argument("--dataset", "-d", default="Officehome")
    parser.add_argument("--Domain_ID", default=[])
    parser.add_argument("--classes", default=[])
    parser.add_argument("--n_classes", type=int, default=7, help="Number of classes")

    # Training Setting
    parser.add_argument("--data_path", default='./dataset', help="your data_path")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="batch size")

    # Data Augmentation Setting
    parser.add_argument("--image_size", type=int, default=224, help="image size")

    return parser.parse_args()


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.clip_model, _ = clip.load(self.args.CLIP_backbone, device=self.device, download_root='./pretrain/CLIP')
        self.text_feature_dim = 512

        self.text_inputs = torch.cat([clip.tokenize(f"a photo of a {c.replace('_', '')}") for c in args.classes]).to(device)
        self.test_loader = data_helper.get_test_dataloader(args)
        logging.info("Dataset size: OOD test %d" % (len(self.test_loader.dataset)))

    def do_testing(self):
        self.clip_model.eval()
        correct_predictions = 0
        total_samples = 0
        for iter, ((data, class_l), d_idx) in enumerate(tqdm(self.test_loader)):
            data, class_l = data.to(self.device), class_l.to(self.device)

            with torch.no_grad():
                logits_per_image, _ = self.clip_model(data, self.text_inputs)
                preds = torch.argmax(logits_per_image.softmax(dim=-1), dim=1)

            correct_predictions += (preds == class_l).sum().item()
            total_samples += class_l.size(0)

        accuracy = correct_predictions / total_samples

        logging.info("\n>>>=====================<<<")
        logging.info(f'Domain: {self.args.target} Accuracy: {accuracy:.4f}')
        logging.info(">>>=====================<<<\n")


def clip_inference():
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
    
    for domain in args.Domain_ID:
        args.target = domain

        args.output_folder = join(os.getcwd(), 'results', args.exp, args.CLIP_backbone.replace('/', '-'),
                                  args.dataset, args.target)
        if os.path.exists(args.output_folder):
            raise ValueError(f"Output path {args.output_folder} existed, please specify a new one by setting <--exp>")
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

        logging.info(f">>> Zero-shot inference on target domain: {args.target}")

        trainer = Trainer(args, device)
        trainer.do_testing()


if __name__ == "__main__":
    clip_inference()
