import numpy as np
import torch
import torch.nn.functional as F
import sys
from itertools import combinations


def absolute_sim_loss(af, ef):
    af = af.unsqueeze(1)  # [bs, 512] to [bs, 1, 512]
    af_expanded = af.expand(-1, ef.size(1), -1)  # [bs, 3, 512]
    cosine_sim = F.cosine_similarity(af_expanded, ef, dim=-1)  # [bs, 3]
    loss = 1 - cosine_sim
    return loss.mean()


def relative_sim_loss(ef):
    K = ef.size(1)

    indices = list(combinations(range(K), 2))
    losses = 0

    for i, j in indices:
        loss = 1 - F.cosine_similarity(ef[:, i, :], ef[:, j, :])
        losses += loss.mean()

    return losses / (K * (K - 1))


def feature_dist_loss(feature1, feature2, reduction='mean'):
    cosine_sim = F.cosine_similarity(feature1, feature2, dim=-1)
    if reduction == 'mean':
        return 1 - cosine_sim.sum() / cosine_sim.size(0)
    else:
        return 1 - cosine_sim


def clip_contrastive_loss(image_features, text_features, temperature=0.07):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    logits_per_image = torch.matmul(image_features, text_features.T) / temperature
    logits_per_text = logits_per_image.T

    batch_size = image_features.size(0)
    labels = torch.arange(batch_size).to(image_features.device)

    loss_image_to_text = F.cross_entropy(logits_per_image, labels)
    loss_text_to_image = F.cross_entropy(logits_per_text, labels)

    return (loss_image_to_text + loss_text_to_image) / 2


def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    # has had softmax applied
    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)
    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalize

    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_j) - lamb * torch.log(p_i))
    loss = loss.sum()

    return loss


def CORAL_loss(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)
    # source covariance
    tmp_s = torch.sum(source, dim=0, keepdim=True)
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)
    # target covariance
    tmp_t = torch.sum(target, dim=0, keepdim=True)
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)
    # frobenius norm
    loss = (cs - ct).pow(2).sum().sqrt()
    loss = loss / (4 * d * d)
    return loss


def asymptotic_gaussian_warmup(x, eta=20, epsilon=0.01, n=2):
    k = (-np.log(epsilon / 0.5)) ** (1 / n)
    return 1 - 0.5 * np.exp(-(k * x / eta) ** n)
