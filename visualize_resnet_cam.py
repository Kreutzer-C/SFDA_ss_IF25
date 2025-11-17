import argparse
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from matplotlib import cm

from model.model_factory import get_backbone, Classifier

class_map = ["aircraft_carrier", "alarm_clock", "ant", "anvil", "asparagus", "axe",
                        "banana", "basket", "bathtub", "bear", "bee", "bird", "blackberry",
                        "blueberry", "bottlecap", "broccoli", "bus", "butterfly", "cactus",
                        "cake", "calculator", "camel", "camera", "candle", "cannon", "canoe",
                        "carrot", "castle", "cat", "ceiling_fan", "cello", "cell_phone", "chair",
                        "chandelier", "coffee_cup", "compass", "computer", "cow", "crab",
                        "crocodile", "cruise_ship", "dog", "dolphin", "dragon", "drums", "duck",
                        "dumbbell", "elephant", "eyeglasses", "feather", "fence", "fish",
                        "flamingo", "flower", "foot", "fork", "frog", "giraffe", "goatee",
                        "grapes", "guitar", "hammer", "helicopter", "helmet", "horse", "kangaroo",
                        "lantern", "laptop", "leaf", "lion", "lipstick", "lobster", "microphone",
                        "monkey", "mosquito", "mouse", "mug", "mushroom", "onion", "panda",
                        "peanut", "pear", "peas", "pencil", "penguin", "pig", "pillow",
                        "pineapple", "potato", "power_outlet", "purse", "rabbit", "raccoon",
                        "rhinoceros", "rifle", "saxophone", "screwdriver", "sea_turtle", "see_saw",
                        "sheep", "shoe", "skateboard", "snake", "speedboat", "spider", "squirrel",
                        "strawberry", "streetlight", "string_bean", "submarine", "swan", "table",
                        "teapot", "teddy-bear", "television", "The_Eiffel_Tower",
                        "The_Great_Wall_of_China", "tiger", "toe", "train", "truck", "umbrella",
                        "vase", "watermelon", "whale", "zebra"]


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet Grad-CAM 可视化脚本")
    parser.add_argument("--ckpt", required=True, help="ResNet 训练权重路径 (.pth/.pt)")
    parser.add_argument("--image", required=True, help="待可视化的原始图像路径")
    parser.add_argument("--output_dir", default="./results/CAM_Vis", help="CAM 叠加热力图输出文件夹")
    parser.add_argument("--backbone", default="resnet50",
                        choices=["resnet18", "resnet50", "resnet101"],
                        help="训练所用的 ResNet Backbone")
    parser.add_argument("--proj_dim", type=int, default=512, help="训练时 projector 的输出维度")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="分类类别数，若不指定则从 checkpoint 中自动推断")
    parser.add_argument("--target_class", type=int, default=None,
                        help="需要可视化的类别 id，默认取模型预测的最大概率类别")
    parser.add_argument("--image_size", type=int, default=224, help="模型输入图像尺寸")
    parser.add_argument("--alpha", type=float, default=0.4, help="叠加热力图的不透明度，取值 0~1")
    parser.add_argument("--device", default=None, help="运行设备，例如 cuda:0 或 cpu，默认自动检测")
    return parser.parse_args()


def infer_num_classes(state_dict: dict, proj_dim: int) -> int:
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and tensor.shape[1] == proj_dim:
            return tensor.shape[0]
    raise RuntimeError("无法从 checkpoint 推断分类类别数，请显式传入 --num_classes")


def load_checkpoint_state(ckpt_path: str, device: torch.device) -> dict:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        for possible_key in ["model", "model_state_dict", "state_dict"]:
            if possible_key in state and isinstance(state[possible_key], dict):
                state = state[possible_key]
                break
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint 格式不正确，未能解析 state_dict")
    return state


def build_model(backbone: str, proj_dim: int, num_classes: int) -> nn.Module:
    featurizer = get_backbone(backbone, proj_dim, pretrained=False)
    classifier = Classifier(proj_dim, num_classes)
    model = nn.Sequential(featurizer, classifier)
    return model


def get_target_layer(model: nn.Module) -> nn.Module:
    # model = Sequential(featurizer, classifier)
    featurizer = model[0]
    if isinstance(featurizer, nn.Sequential):
        encoder = featurizer[0]
        if isinstance(encoder, nn.Sequential):
            return encoder[-2]
    raise RuntimeError("未能定位 ResNet layer4 层，无法计算 CAM")


def prepare_image(image_path: str, image_size: int) -> Tuple[torch.Tensor, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor, image


def compute_grad_cam(model: nn.Module,
                     target_layer: nn.Module,
                     input_tensor: torch.Tensor,
                     target_class: Optional[int],
                     device: torch.device) -> Tuple[np.ndarray, int]:
    activations = {}
    gradients = {}

    def forward_hook(_, __, output):
        activations["value"] = output.detach()

    def backward_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_full_backward_hook(backward_hook)

    try:
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)
        outputs = model(input_tensor)
        if target_class is None:
            target_class = int(torch.argmax(outputs, dim=1).item())
        score = outputs[:, target_class]
        model.zero_grad()
        score.backward()

        if "value" not in activations or "value" not in gradients:
            raise RuntimeError("未能获取到 CAM 所需的特征或梯度")

        grads = gradients["value"]
        acts = activations["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, target_class
    finally:
        handle_fw.remove()
        handle_bw.remove()


def overlay_heatmap(original_image: Image.Image, heatmap: np.ndarray, alpha: float) -> Image.Image:
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(original_image.size, Image.BILINEAR)
    heatmap_np = np.array(heatmap_img) / 255.0
    colored = cm.get_cmap("jet")(heatmap_np)[..., :3]
    original_np = np.array(original_image) / 255.0
    overlay = np.clip(alpha * colored + (1 - alpha) * original_np, 0, 1)
    return Image.fromarray(np.uint8(overlay * 255))


def main():
    args = parse_args()
    device_str = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    state_dict = load_checkpoint_state(args.ckpt, device)
    num_classes = args.num_classes or infer_num_classes(state_dict, args.proj_dim)

    model = build_model(args.backbone, args.proj_dim, num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[警告] 以下权重缺失: {missing}")
    if unexpected:
        print(f"[警告] 以下权重在模型中未使用: {unexpected}")
    model.to(device)
    model.eval()

    target_layer = get_target_layer(model)

    input_tensor, original_image = prepare_image(args.image, args.image_size)
    heatmap, used_class = compute_grad_cam(model, target_layer, input_tensor, args.target_class, device)

    overlay = overlay_heatmap(original_image, heatmap, args.alpha)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image))
    overlay.save(output_path)
    print(f"Saved CAM heatmap for class [{class_map[used_class]}] to {output_path}")


if __name__ == "__main__":
    main()

