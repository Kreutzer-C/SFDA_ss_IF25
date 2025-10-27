import argparse
import sys
import logging
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import clip
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from batchgenerators.utilities.file_and_folder_operations import *
from utils.loss import *
from data import data_helper
from data.data_info import get_data_info


def get_args():
    parser = argparse.ArgumentParser(description="SFDA-IF25 single-source version")
    parser.add_argument("--exp", type=str, default="CLIP_Text_Feat_Visualization")

    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    parser.add_argument("--backbone", default="resnet50", help="target model arch, only resnet18 | resnet50 | resnet 101 are supported")
    parser.add_argument("--CLIP_backbone", default="ViT-B/16", help="CLIP model vision encoder arch, only ViT-B/16 | ViT-B/32 are supported")

    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'],
                        help='list of attention matrices where deploy LoRA')
    parser.add_argument('--position', type=str, default='all',
                        choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'],
                        help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling equals alpha/r, see LoRA paper')
    parser.add_argument('--dropout_rate', '-dr', default=0.1, type=float, help='dropout rate applied before the LoRA module')

    parser.add_argument("--dataset", "-d", default="Officehome")
    parser.add_argument("--Domain_ID", default=[])
    parser.add_argument("--classes", default=[])
    parser.add_argument("--n_classes", type=int, default=7, help="Number of classes")
    parser.add_argument("--val_size", type=float, default=0.1)

    parser.add_argument("--data_path", default='./dataset', help="your data_path")
    parser.add_argument("--source", "-src", default=None, help="Specify single source domain")
    parser.add_argument("--target", "-tgt", default=None, help="Specify single target domain")
    parser.add_argument("--lora_cp_path", default=None, help="Path to the LoRA checkpoint")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")

    parser.add_argument("--image_size", type=int, default=224, help="Image size")

    parser.add_argument("--mode", required=True, help="zs | st | ours | oracle")

    return parser.parse_args()


def get_clip_outputs(clip_model, image, prompt):
    with ((torch.amp.autocast(device_type="cuda", dtype=torch.float16))):
        image_feature = clip_model.encode_image(image)

        classify_prompt = [cp.replace('_', ' ') for cp in prompt]
        classify_token = clip.tokenize(classify_prompt).to(image.device)
        text_feature = clip_model.encode_text(classify_token)

    image_feature_norm = image_feature / image_feature.norm(dim=-1, keepdim=True)
    text_feature_norm = text_feature / text_feature.norm(dim=-1, keepdim=True)
    logit_scale = clip_model.logit_scale.data
    logit_scale = logit_scale.exp()
    outputs = (logit_scale * image_feature_norm @ text_feature_norm.T).to(torch.float32)
    return outputs, image_feature


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.clip_feature_dim = 512

        self.clip_model, _ = clip.load(self.args.CLIP_backbone, device=self.device, download_root='./pretrain/CLIP')
        if args.mode != 'zs':
            self.list_lora_layers = apply_lora(args, self.clip_model)
            print(len(self.list_lora_layers))
            load_lora(args, self.list_lora_layers, load_path=args.lora_cp_path)
            logging.info(f">>> Loading LoRA checkpoint from: {args.lora_cp_path}")
            mark_only_lora_as_trainable(self.clip_model)

        self.clip_model = self.clip_model.to(device)

        self.target_loader = data_helper.get_test_dataloader(args)
        logging.info("Dataset size: OD test(target) %d" % (len(self.target_loader.dataset)))

    def _do_epoch(self):
        self.clip_model.eval()
        all_features = []
        all_classes = []
        all_domains = []

        all_prompts = []
        for i in range(len(self.args.classes)):
            for j in range(len(self.args.Domain_ID)):
                c = self.args.classes[i]
                d = self.args.Domain_ID[j]
                prompt = f"a {d} for a {c}".replace("_", " ")
                all_prompts.append(prompt)
                all_classes.append(i)
                all_domains.append(j)

        with torch.no_grad():
            with ((torch.amp.autocast(device_type="cuda", dtype=torch.float16))):
                tokens = clip.tokenize(all_prompts).to(self.device)
                text_feature = self.clip_model.encode_text(tokens)

                all_features.append(text_feature.cpu().detach().numpy())

        return all_features, all_classes, all_domains

    def do_inference(self):
        all_features, all_classes, all_domains = self._do_epoch()

        epoch_feature = np.concatenate(all_features, axis=0)
        epoch_pred = np.array(all_classes)
        epoch_domain = np.array(all_domains)
        assert epoch_feature.shape[0] == epoch_pred.shape[0] == epoch_domain.shape[0]
        feature_save_path = join(self.args.output_folder, 'clip_feature')
        maybe_mkdir_p(feature_save_path)
        np.save(join(feature_save_path, "text_feature.npy"), epoch_feature)
        np.save(join(feature_save_path, "class.npy"), epoch_pred)
        np.save(join(feature_save_path, "domain.npy"), epoch_domain)
        all_features.clear()
        logging.info(f">>> Text features saved to: {feature_save_path}")
        logging.info(f">>> Class labels saved to: {feature_save_path}/class.npy")
        logging.info(f">>> Domain labels saved to: {feature_save_path}/domain.npy")


def inference():
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

    args.output_folder = os.path.join(os.getcwd(), 'results', args.dataset, f"{args.source}_to_{args.target}", args.exp)
    maybe_mkdir_p(args.output_folder)
    print("output results are saved at: {}".format(args.output_folder))

    if args.mode != 'zs':
        assert args.lora_cp_path is not None, "LoRA checkpoint path is required for non-zero-shot mode"
        assert os.path.isfile(args.lora_cp_path), "LoRA checkpoint file does not exist"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.output_folder + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    logging.info("\n****************************")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    logging.info("****************************\n")

    logging.info(">>>Inference on target domain:")
    logging.info(args.target)

    trainer = Trainer(args, device)
    trainer.do_inference()


if __name__ == "__main__":
    inference()