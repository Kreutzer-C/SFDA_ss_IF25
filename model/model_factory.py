import torch
import torch.nn as nn
import torchvision.models as models


def get_backbone(name, proj_dim, pretrained=True):
    if name == "resnet50":
        network = models.resnet50(weights = "ResNet50_Weights.DEFAULT" if pretrained else None)
        encoder = nn.Sequential(*list(network.children())[:-1])
        projector = nn.Linear(2048, proj_dim)
        network = nn.Sequential(
            encoder,
            nn.Flatten(),
            projector
        )
        return network

    elif name == "resnet101":
        network = models.resnet101(weights = "ResNet101_Weights.DEFAULT" if pretrained else None)
        encoder = nn.Sequential(*list(network.children())[:-1])
        projector = nn.Linear(2048, proj_dim)
        network = nn.Sequential(
            encoder,
            nn.Flatten(),
            projector
        )
        return network

    elif name == "resnet18":
        network = models.resnet18(weights = "ResNet18_Weights.DEFAULT" if pretrained else None)
        encoder = nn.Sequential(*list(network.children())[:-1])
        projector = nn.Linear(2048, proj_dim)
        network = nn.Sequential(
            encoder,
            nn.Flatten(),
            projector
        )
        return network

    else:
        raise ValueError(name)


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


if __name__ == "__main__":
    model = get_backbone('resnet50', proj_dim=512)
    print(model)
