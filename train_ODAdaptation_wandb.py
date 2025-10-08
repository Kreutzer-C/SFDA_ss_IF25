import argparse
import os
import sys
import logging
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import wandb
import clip
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora
from batchgenerators.utilities.file_and_folder_operations import *

from data import data_helper
from data.data_info import get_data_info
from model.model_factory import get_backbone, Classifier
from model.CO_module import Collaboration_Module
from utils.text_prompts_lib import get_text_prompts
from utils.loss import *


def get_args():
    parser = argparse.ArgumentParser(description="SFDA-IF25 single-source version")
    # Experiment Name
    # (Determines where the results are saved, highly recommended to keep it different for each experiment)
    parser.add_argument("--exp", type=str, default="ODA")

    # Device Setting
    parser.add_argument("--GPU_num", default="0", help="specify which GPU(s) to be used")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    # Backbone Network Setting
    parser.add_argument("--backbone", default="resnet50", help="target model arch, only resnet18 | resnet50 | resnet 101 are supported")
    parser.add_argument("--CLIP_backbone", default="ViT-B/16", help="CLIP model vision encoder arch, only ViT-B/16 | ViT-B/32 are supported")

    # LoRA Setting
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'],
                        help='list of attention matrices where deploy LoRA')
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

    # Dataset Setting (Only --dataset and --tgt_classes need to be determined)
    parser.add_argument("--dataset", "-d", default="Officehome")
    parser.add_argument("--Domain_ID", default=[])
    parser.add_argument("--classes", default=[])
    parser.add_argument("--n_classes", type=int, default=7, help="Number of classes")
    parser.add_argument("--src_classes", type=int, default=25, help="Number of source classes")

    # Training Setting (two learning rates may need to be adjusted depending on the specific task)
    parser.add_argument("--data_path", default='./dataset', help="your data_path")
    parser.add_argument("--source", "-src", default=None, help="Specify single source domain")
    parser.add_argument("--target", "-tgt", default=None, help="Specify single target domain")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning_rate_tar", "-lr1", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--learning_rate_lora", "-lr2", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--src_init", action='store_true', help="Whether to initialize the prototype memory bank from source domain")

    # Data Augmentation Setting
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscale")

    # Trade-off Hyperparameters Setting
    parser.add_argument("--lamb1", default=1.0, type=float,
                        help='trade-off parameter of L_tc in L_CLIP')
    parser.add_argument("--lamb2", default=1.0, type=float,
                        help='trade-off parameter of L_tg in L_TMA')
    parser.add_argument("--zeta", default=10.0, type=float,
                        help='trade-off parameter of L_cls in L_MG')
    parser.add_argument("--alpha_decay", "-ad", default=0.5, type=float,
                        help='trade-off parameter of prototype bank updating')
    parser.add_argument("--dynamic_ad", action='store_true', 
                        help="Whether to use dynamic alpha decay")
    parser.add_argument("--K", default=3, type=int,
                        help='random choice times for constructing entangled prompts')
    parser.add_argument("--temperature", "-tau", default=0.07, type=float,
                        help='temperature in InfoNCE loss')

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
        self.classes = args.classes
        self.n_classes = args.n_classes
        self.lamb1 = args.lamb1
        self.lamb2 = args.lamb2
        self.zeta = args.zeta
        self.ad = args.alpha_decay
        self.K = args.K
        self.temperature = args.temperature
        self.clip_feature_dim = 512  # For ViT-B/16 or ViT-B/32

        # Dataloader
        self.target_loader = data_helper.get_ODA_adapt_dataloader(args)
        logging.info("Dataset size: OOD(target) %d" % (len(self.target_loader.dataset)))
        self.max_iter = args.epochs * len(self.target_loader)

        # Define VLM and apply LoRA
        self.clip_model, _ = clip.load(self.args.CLIP_backbone, device=self.device, download_root='./pretrain/CLIP')
        self.list_lora_layers = apply_lora(args, self.clip_model)
        if args.dataset == 'Terra':
            load_lora(args, self.list_lora_layers, load_path=args.lora_cp_path)
        self.clip_model = self.clip_model.to(device)
        print(len(self.list_lora_layers))
        mark_only_lora_as_trainable(self.clip_model)

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

        # Define CO-Module
        self.collaboration_module = Collaboration_Module(self.args.n_classes, device).to(device)

        # Define optimizers and schedulers
        self.optimizer_tar = optim.AdamW(self.model.parameters(), lr=args.learning_rate_tar,
                                         betas=(0.9, 0.999))
        self.optimizer_lora = optim.AdamW(get_lora_parameters(self.clip_model), lr=args.learning_rate_lora,
                                          betas=(0.9, 0.999))
        self.warmup_iters = int(self.max_iter * args.warmup_ratio)
        self.scheduler_tar = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_tar, 
            T_0=self.max_iter - self.warmup_iters + 1,  # 余弦退火周期
            T_mult=1,  # 周期倍数
            eta_min=0.0  # 最小学习率
        )
        self.scheduler_lora = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_lora, 
            T_0=self.max_iter - self.warmup_iters + 1,  # 余弦退火周期
            T_mult=1,  # 周期倍数
            eta_min=0.0  # 最小学习率
        )

        self.classify_prompt = [f"a photo of a {c.replace('_', ' ')}" for c in self.args.classes]
        self.cosine_sim_criterion = nn.CosineEmbeddingLoss()

        self.best_epoch_acc = 0.0
        self.best_vlm_acc = 0.0
        self.current_epoch = None

        wandb.init(
            project="SFDA_ss_IF25",
            name=f"{args.exp}: {args.dataset}_{args.source[:2]}_to_{args.target[:2]}",
            tags=[args.dataset, args.source, args.target, "ODA"],
            config=vars(args)
        )

    def initialize_prototype(self):
        self.featurizer.eval()
        self.classifier.eval()
        prototype_sum = torch.zeros(self.n_classes, self.n_classes).to(self.device)
        prototype_count = torch.zeros(self.n_classes).to(self.device)
        with torch.no_grad():
            logging.info("\nInitialize prototype memory...")
            for iteration, ((img, class_l), d_idx) in enumerate(tqdm(self.target_loader)):
                img, class_l = img.to(self.device), class_l.to(self.device)

                z = self.featurizer(img)
                p_src = torch.softmax(self.classifier(z), dim=1)
                y_src = torch.argmax(p_src, dim=1)
                for i in range(self.n_classes):
                    mask = (y_src == i)
                    if mask.any():
                        p_src_i = p_src[mask]
                        prototype_sum[:, i] += p_src_i.sum(dim=0)
                        prototype_count[i] += mask.sum()

            init_memory = torch.zeros_like(prototype_sum)  # [n_classes, n_classes]
            uniform = torch.full((self.n_classes,), 1.0 / self.n_classes).to(self.device)
            for i in range(self.n_classes):
                if prototype_count[i] > 0:
                    init_memory[:, i] = prototype_sum[:, i] / prototype_count[i]
                else:
                    init_memory[:, i] = uniform

        self.collaboration_module.initialize(init_memory)
        memory_image = self.collaboration_module.memory_bank.unsqueeze(0).detach().cpu().numpy()
        wandb.log({"Prototype_Matrix": wandb.Image(memory_image)}, step=0)

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
        scaler = torch.cuda.amp.GradScaler()

        # Stage1:
        logging.info("\nDoing VLM LoRA Tuning...")
        self.featurizer.eval()
        self.classifier.eval()
        correct_predictions = 0
        correct_predictions_mix = 0
        total_samples = 0
        for iteration, ((img, class_l), d_idx) in enumerate(tqdm(self.target_loader)):
            img, class_l = img.to(self.device), class_l.to(self.device)

            self.clip_model.eval()
            with torch.no_grad():
                z = self.featurizer(img)
                p_tar = torch.softmax(self.classifier(z), dim=1)
                out_vlm = get_clip_outputs(self.clip_model, img, self.classify_prompt)
                p_vlm = torch.softmax(out_vlm, dim=1)
                p_mix = self.collaboration_module(p_tar, p_vlm, self.ad)

            y_pred_mix = torch.argmax(p_mix, dim=1)
            anchor_prompt = []
            entangled_prompts = []
            for i in y_pred_mix:
                cls_name = self.args.classes[i].replace('_', ' ')
                anchor_prompt.append(f"a {cls_name}.")
                single_sample_entangled_prompt = get_text_prompts(cls_name, self.K)  # len = K
                entangled_prompts.extend(single_sample_entangled_prompt)  # len = bs*K
            # print(f"{anchor_prompt} || {entangled_prompts}")

            self.clip_model.train()
            self.optimizer_lora.zero_grad()
            with ((torch.amp.autocast(device_type="cuda", dtype=torch.float16))):
                image_feature = self.clip_model.encode_image(img)

                anchor_token = clip.tokenize(anchor_prompt).to(self.device)
                anchor_feature = self.clip_model.encode_text(anchor_token)  # (bs, 512)

                entangled_tokens = clip.tokenize(entangled_prompts).to(self.device)
                entangled_features = self.clip_model.encode_text(entangled_tokens)
                entangled_features = entangled_features.view(-1, self.K, self.clip_feature_dim)  # (bs,K,512)

            ab_sim_loss = absolute_sim_loss(anchor_feature, entangled_features)
            re_sim_loss = relative_sim_loss(entangled_features)
            loss_tc = ab_sim_loss + re_sim_loss

            loss_nce = clip_contrastive_loss(image_feature, anchor_feature, temperature=self.temperature)

            loss_clip = self.lamb1 * loss_tc + loss_nce

            scaler.scale(loss_clip).backward()
            scaler.step(self.optimizer_lora)
            scaler.update()

            iter_num = iteration + self.current_epoch * len(self.target_loader)
            self._update_learning_rate(self.optimizer_lora, self.scheduler_lora, self.args.learning_rate_lora, iter_num)
            wandb.log({
                'stage1_LoRA/loss_clip': loss_clip,
                'stage1_LoRA/loss_tc': loss_tc,
                'stage1_LoRA/loss_nce': loss_nce,
                'stage1_LoRA/lr_lora': self.optimizer_lora.param_groups[0]['lr'],
                'stage1_LoRA/step': iter_num
            })

            y_pred_vlm = torch.argmax(p_vlm, dim=1)
            correct_predictions += (y_pred_vlm == class_l).sum().item()
            correct_predictions_mix += (y_pred_mix == class_l).sum().item()
            total_samples += class_l.size(0)

        vlm_accuracy = correct_predictions / total_samples
        mix_accuracy = correct_predictions_mix / total_samples
        logging.info(f"\n >>> LoRA Tuning DONE. [VLM acc={vlm_accuracy:.4f}, Mix acc={mix_accuracy:.4f}]")
        wandb.log({
            'common/vlm_acc': vlm_accuracy,
            'common/mix_acc': mix_accuracy,
            'common/step': self.current_epoch
        })

        # Stage2:
        logging.info("\nDoing Target Model Adaptation...")
        self.clip_model.eval()
        self.featurizer.train()
        self.classifier.train()
        correct_predictions = 0
        total_samples = 0
        for iteration, ((img, class_l), d_idx) in enumerate(self.target_loader):
            img, class_l = img.to(self.device), class_l.to(self.device)

            self.optimizer_tar.zero_grad()
            z = self.featurizer(img)
            out_tar = self.classifier(z)
            p_tar = torch.softmax(out_tar, dim=1)

            with torch.no_grad():
                out_vlm = get_clip_outputs(self.clip_model, img, self.classify_prompt)
                p_vlm = torch.softmax(out_vlm, dim=1)

                p_mix = self.collaboration_module(p_tar, p_vlm, self.ad)
                y_pred_mix = torch.argmax(p_mix, dim=1)

                anchor_prompt = [f"a {self.args.classes[i].replace('_', ' ')}." for i in y_pred_mix]
                with ((torch.amp.autocast(device_type="cuda", dtype=torch.float16))):
                    anchor_token = clip.tokenize(anchor_prompt).to(self.device)
                    anchor_feature = self.clip_model.encode_text(anchor_token)

            loss_mi = IID_loss(p_tar, p_mix)
            loss_cls = (- p_mix * p_tar).sum(dim=1).mean()
            loss_tg = feature_dist_loss(anchor_feature.to(torch.float32), z)

            loss_tma = loss_mi + self.zeta * loss_cls + self.lamb2 * loss_tg

            loss_tma.backward()
            self.optimizer_tar.step()

            if iteration % (len(self.target_loader) // 3) == 0:
                logging.info("iter {}/{} loss_TMA: {:.6f} loss_mi: {:.6f} loss_cls: {:.6f} loss_tg: {:.6f}"
                             .format(iteration, len(self.target_loader),
                                     loss_tma.item(), loss_mi.item(), loss_cls.item(), loss_tg.item()))

            iter_num = iteration + self.current_epoch * len(self.target_loader)
            self._update_learning_rate(self.optimizer_tar, self.scheduler_tar, self.args.learning_rate_tar, iter_num)
            wandb.log({
                'stage2_TMA/loss_tma': loss_tma,
                'stage2_TMA/loss_mi': loss_mi,
                'stage2_TMA/loss_cls': loss_cls,
                'stage2_TMA/loss_tg': loss_tg,
                'stage2_TMA/lr_tar': self.optimizer_tar.param_groups[0]['lr'],
                'stage2_TMA/step': iter_num
            })

            y_pred_tar = torch.argmax(p_tar, dim=1)
            correct_predictions += (y_pred_tar == class_l).sum().item()
            total_samples += class_l.size(0)

        tar_accuracy = correct_predictions / total_samples
        wandb.log({
            'common/tar_acc': tar_accuracy,
            'common/step': self.current_epoch
        })  
        return tar_accuracy, vlm_accuracy

    def do_training(self):
        for self.current_epoch in tqdm(range(self.args.epochs)):
            if self.args.dynamic_ad:
                self.ad = asymptotic_gaussian_warmup(self.current_epoch, eta=self.args.epochs-1)
                wandb.log({
                    'common/alpha_decay': self.ad,
                    'common/step': self.current_epoch
                })

            tar_accuracy, vlm_accuracy = self._do_epoch()

            prototype_memory = self.collaboration_module.memory_bank.clone().unsqueeze(0).detach().cpu().numpy()
            wandb.log({
                "Prototype_Matrix": wandb.Image(prototype_memory),
                'common/step': self.current_epoch+1
            })

            if tar_accuracy >= self.best_epoch_acc:
                self.best_epoch_acc = tar_accuracy
                logging.info("\n*************NEW BEST!************")
                logging.info(f'Epoch: {self.current_epoch} Accuracy: {tar_accuracy:.4f} VLM: {vlm_accuracy:.4f}\n')
                save_lora(self.args, self.list_lora_layers)
                target_save_path = join(self.args.output_folder, f'best_targetmodel.pth')
                torch.save(self.model.state_dict(), target_save_path)
            else:
                logging.info(f'Epoch: {self.current_epoch} Accuracy: {tar_accuracy:.4f} VLM: {vlm_accuracy:.4f}\n')

            if vlm_accuracy >= self.best_vlm_acc:
                self.best_vlm_acc = vlm_accuracy

    def do_test(self):
        logging.info("\n>>>=====================<<<")
        logging.info(f">>>Testing SFDA task: {self.args.source} --> {self.args.target}")

        self.model.load_state_dict(torch.load(join(self.args.output_folder, f'best_targetmodel.pth')))
        self.model.eval()
        correct_predictions = 0
        total_samples = 0
        self.test_loader = data_helper.get_PDA_test_dataloader(self.args)

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
        
        args.checkpoint_path = join(os.getcwd(), 'pretrain', f'ERM_ODA{args.src_classes}', args.backbone, args.dataset, f"{args.source}_best.pth")
        args.lora_cp_path = join(os.getcwd(), 'pretrain', 'LORA', args.CLIP_backbone.replace('/', '-'),
                                 args.dataset, f"{args.source}", 'lora.pt')

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
        if args.src_init:
            trainer.initialize_prototype()
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
                
                args.checkpoint_path = join(os.getcwd(), 'pretrain', f'ERM_ODA{args.src_classes}', args.backbone, args.dataset, f"{args.source}_best.pth")
                args.lora_cp_path = join(os.getcwd(), 'pretrain', 'LORA', args.CLIP_backbone.replace('/', '-'),
                                        args.dataset, f"{args.source}", 'lora.pt')

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
                if args.src_init:
                    trainer.initialize_prototype()
                trainer.do_training()
                trainer.do_test()


if __name__ == '__main__':
    train_adaptation()
