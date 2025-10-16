import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class Collaboration_Module(nn.Module):
    def __init__(self, n_classes, device):
        super(Collaboration_Module, self).__init__()
        self.n_classes = n_classes
        self.memory_bank = torch.full((n_classes, n_classes), 1.0 / n_classes).to(device)
        self._norm_fact = 1 / math.sqrt(n_classes)

    def initialize(self, init_memory):
        assert init_memory.shape == (self.n_classes, self.n_classes)
        self.memory_bank = init_memory.clone()

    def class_prototype_atten(self, p_tar):
        # p_tar_norm = p_tar / p_tar.norm(dim=1, keepdim=True)
        # proto_norm = self.memory_bank / self.memory_bank.norm(dim=1, keepdim=True)
        proto = self.memory_bank.clone()
        atten = torch.softmax(torch.matmul(p_tar, proto.permute(1, 0)) * self._norm_fact, dim=1)  # N*C
        p_new = torch.matmul(atten, proto)
        return p_new

    def update_membank(self, p_tar, p_vlm, alpha):
        y_tar = torch.argmax(p_tar, dim=1)
        y_vlm = torch.argmax(p_vlm, dim=1)
        for i in range(self.n_classes):
            indices = (y_tar == i) & (y_vlm == i)

            if indices.any():
                selected_p_tar = p_tar[indices]
                selected_p_tar_mean = selected_p_tar.mean(dim=0)
                self.memory_bank[i, :] = alpha * self.memory_bank[i, :] + (1 - alpha) * selected_p_tar_mean

    @staticmethod
    def get_uncertainty(p):
        return -1.0 * p * torch.log(p + 1e-6)

    @staticmethod
    def gaussian_warmup(x, sigma=10):
        return 1 - 0.5 * np.exp(-(x ** 2) / (2 * sigma ** 2))

    def uncertainty_mixing(self, p_tar, p_vlm):
        u_tar = self.get_uncertainty(p_tar)
        u_vlm = self.get_uncertainty(p_vlm)

        eu_tar = torch.exp_(-u_tar)
        eu_vlm = torch.exp_(-u_vlm)

        p_mix = (eu_tar * p_tar + eu_vlm * p_vlm) / (eu_tar + eu_vlm)
        return p_mix

    def forward(self, p_tar, p_vlm, alpha, mode='full'):
        if mode == 'full':
            p_tar_new = self.class_prototype_atten(p_tar)
            self.update_membank(p_tar, p_vlm, alpha)
            p_mix = self.uncertainty_mixing(p_tar_new, p_vlm)

        elif mode == 'ent':
            p_mix = self.uncertainty_mixing(p_tar, p_vlm)
        
        elif mode == 'avg':
            p_mix = (p_tar + p_vlm) / 2
        
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return p_mix


if __name__ == '__main__':
    device = torch.device("cpu")
    cm = Collaboration_Module(4, device)
    print(cm.memory_bank)

    ini_m = torch.ones((4,4)).to(device)
    cm.initialize(ini_m)
    print(cm.memory_bank)

    # p = torch.tensor([[0.4, 0.1, 0.3, 0.2], [0.7, 0.1, 0.1, 0.1]])
    # u = cm.get_uncertainty(p)
    # print(u)
    # print(torch.exp_(-u))
    # print(cm.memory_bank)
    #
    # import math
    # sim = torch.softmax(torch.matmul(p, cm.memory_bank.permute(1, 0)) / math.sqrt(4), dim=1)
    # output = torch.matmul(sim, cm.memory_bank)
    # print(output)



