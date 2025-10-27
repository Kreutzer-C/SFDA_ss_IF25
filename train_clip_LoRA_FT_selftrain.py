import argparse
import sys
import logging
import wandb
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
import clip
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from batchgenerators.utilities.file_and_folder_operations import *
from utils.loss import *
from data import data_helper
from data.data_info import get_data_info


def get_args():
    parser = argparse.ArgumentParser(description="SFDA-IF25 single-source version")
    # Experiment Name
    parser.add_argument("--exp", type=str, default="CLIP-selftrain")

    # Device Setting
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    # Backbone Network Setting
    parser.add_argument("--CLIP_backbone", default="ViT-B/16", help="CLIP model vision encoder arch, only ViT-B/16 | ViT-B/32 are supported")

    # LoRA Setting
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'],
                        help='list of attention matrices where putting a LoRA')
    parser.add_argument('--position', type=str, default='all',
                        choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'],
                        help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--r', default=2, type=int, 
                        help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, 
                        help='scaling equals alpha/r, see LoRA paper')
    parser.add_argument('--dropout_rate', '-dr', default=0.1, type=float,
                        help='dropout rate applied before the LoRA module')

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
    parser.add_argument("--learning_rate_lora", "-lr2", type=float, default=1e-5, help="Learning rate")

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
    return outputs


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.clip_model, _ = clip.load(self.args.CLIP_backbone, device=self.device, download_root='./pretrain/CLIP')
        self.list_lora_layers = apply_lora(args, self.clip_model)
        self.clip_model = self.clip_model.to(device)
        print(len(self.list_lora_layers))
        mark_only_lora_as_trainable(self.clip_model)
        self.clip_feature_dim = 512

        self.classify_prompt = [f"a photo of a {c.replace('_', ' ')}" for c in self.args.classes]

        self.target_loader = data_helper.get_adapt_dataloader(args)
        self.test_loader = data_helper.get_test_dataloader(args)
        logging.info(f"Dataset size: OOD(target) {len(self.target_loader.dataset)}")

        self.optimizer_lora = optim.AdamW(get_lora_parameters(self.clip_model), lr=args.learning_rate_lora,
                                          betas=(0.9, 0.999))
        self.scheduler_lora = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_lora,
                                                                   args.epochs, eta_min=0.0)

        self.best_epoch_acc = 0.0
        self.writer = wandb.init(
            project="SFDA_ss_IF25",
            name=f"{args.exp}: {args.dataset}_{args.source}_to_{args.target}",
            tags=[args.dataset, args.source, args.target],
            config=vars(args)
        )


    def _do_epoch(self):
        scaler = torch.cuda.amp.GradScaler()
        self.clip_model.train()

        logging.info("\nDoing CLIP LoRA Tuning...")
        for iteration, ((data, class_l), _) in enumerate(self.target_loader):
            data, class_l = data.to(self.device), class_l.to(self.device)

            # Step1: Generate pseudo label
            self.clip_model.eval()
            with torch.no_grad():
                output = get_clip_outputs(self.clip_model, data, self.classify_prompt)
                p_logits = torch.softmax(output, dim=1)
                y_pseudo = torch.argmax(p_logits, dim=1)

            # Step2: Self-Training FT
            self.clip_model.train()
            self.optimizer_lora.zero_grad()
            y_pseudo = y_pseudo.detach().clone()
            anchor_prompt = [f"a {self.args.classes[i].replace('_',' ')}" for i in y_pseudo]

            with ((torch.amp.autocast(device_type="cuda", dtype=torch.float16))):
                image_feature = self.clip_model.encode_image(data)

                anchor_token = clip.tokenize(anchor_prompt).to(self.device)
                anchor_feature = self.clip_model.encode_text(anchor_token)  # [bs, 512]

            loss_lora = clip_contrastive_loss(image_feature, anchor_feature)

            scaler.scale(loss_lora).backward()
            scaler.step(self.optimizer_lora)
            scaler.update()

            wandb.log({
                'stage1_LoRA/loss_lora': loss_lora.item(),
                'stage1_LoRA/step': iteration + self.current_epoch * len(self.target_loader),
                'stage1_LoRA/lr_lora': self.optimizer_lora.param_groups[0]['lr']
            })


        self.clip_model.eval()
        correct_predictions = 0
        total_samples = 0
        for iteration, ((data, class_l), _) in enumerate(self.test_loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            with torch.no_grad():
                output = get_clip_outputs(self.clip_model, data, self.classify_prompt)
                p_logits = torch.softmax(output, dim=1)

            y_pred = torch.argmax(p_logits, dim=1)
            correct_predictions += (y_pred == class_l).sum().item()
            total_samples += class_l.size(0)

        accuracy = correct_predictions / total_samples

        wandb.log({
            'common/accuracy': accuracy,
            'common/step': self.current_epoch
        })
        return accuracy

    def do_training(self):
        self.iter_num = 0
        for self.current_epoch in tqdm(range(self.args.epochs)):
            accuracy = self._do_epoch()
            self.scheduler_lora.step()

            if accuracy > self.best_epoch_acc:
                self.best_epoch_acc = accuracy
                logging.info("\n*************NEW BEST!************")
                logging.info(f'Epoch: {self.current_epoch} Accuracy: {accuracy:.4f}')
                save_lora(self.args, self.list_lora_layers)
            else:
                logging.info(f'Epoch: {self.current_epoch} Accuracy: {accuracy:.4f}')


    def do_test(self):
        logging.info("\n>>>=====================<<<")
        print("Testing on OD domain")
        logging.info(f'Domain: {self.args.target} Accuracy: {self.best_epoch_acc:.4f}')
        logging.info(">>>=====================<<<\n")


def train():
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

    logging.info(f">>>Training {args.dataset} on source domain: {args.source}")
    logging.info(f">>>Adapting to target domain: {args.target}")

    trainer = Trainer(args, device)
    trainer.do_training()
    trainer.do_test()


if __name__ == "__main__":
    train()
