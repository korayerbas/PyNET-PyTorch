from torch import nn
import torch

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, sigmoid

class EarthMoversDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # input has dims: (Batch x Bins)
        bins = x.size(1)
        r = torch.arange(bins)
        s, t = torch.meshgrid(r, r)
        tt = t >= s

        cdf_x = torch.matmul(x, tt.float())
        cdf_y = torch.matmul(y, tt.float())

        return torch.sum(torch.square(cdf_x - cdf_y), dim=1)

class MutualInformationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p1, p2, p12):
        # input p12 has dims: (Batch x Bins x Bins)
        # input p1 & p2 has dims: (Batch x Bins)

        product_p = torch.matmul(torch.transpose(p1.unsqueeze(1), 1, 2), p2.unsqueeze(1)) + torch.finfo(p1.dtype).eps
        mi = torch.sum(p12 * torch.log(p12 / product_p + torch.finfo(p1.dtype).eps), dim=(1, 2))
        h = -torch.sum(p12 * torch.log(p12 + torch.finfo(p1.dtype).eps), dim=(1, 2))

        return 1 - (mi / h)

def np_to_torch(img, add_batch_dim=True):
    img = np.asarray(img)
    img = img.astype(np.float32).transpose((2, 0, 1))
    if add_batch_dim:
        img = img[np.newaxis, ...]
    img_torch = torch.from_numpy(img) / 255.0
    return img_torch


#def torch_to_np(tensor):
#    tensor = tensor.detach().squeeze().cpu().numpy()
#    if len(tensor.shape) < 3:
#        return tensor
#    else:
#        return tensor.transpose(1, 2, 0)

#def main():
#    # TODO: convert RGB to YUV space
#    # and sum loss for all channels

#    result = np_to_torch(Image.open("deep_hist.png").resize((460, 460)).convert("RGB"))
#    source = np_to_torch(Image.open("source.png").resize((460, 460)).convert("RGB"))
#    target = np_to_torch(Image.open("target.png").resize((460, 460)).convert("RGB"))

#    hist1 = SingleDimHistLayer()(source[:, 0])
#    hist2 = SingleDimHistLayer()(target[:, 0])
#    hist3 = SingleDimHistLayer()(result[:, 0])

#    print("emd: source - target", EarthMoversDistanceLoss()(hist1, hist2))
#    print("emd: target - result", EarthMoversDistanceLoss()(hist3, hist2))

    # we compare the differentiable histogram with the one produced by numpy
#    _, ax = plt.subplots(2)
#    ax[0].plot(hist1[0].cpu().numpy())
#    ax[1].plot(np.histogram(source[:, 0].view(-1).cpu().numpy(), bins=256)[0])
#    plt.show()

#    joint_hist1 = JointHistLayer()(source[:, 0], result[:, 0])
#    joint_hist2 = JointHistLayer()(target[:, 0], result[:, 0])
#    joint_hist_self = JointHistLayer()(source[:, 0], source[:, 0])

#    print("mi loss: source - result", MutualInformationLoss()(hist1, hist3, joint_hist1))
#    print("mi loss: target - result", MutualInformationLoss()(hist2, hist3, joint_hist2))
#    print("mi loss: source - source", MutualInformationLoss()(hist1, hist1, joint_hist_self))

def phi_k(x, L, W):
    return sigmoid((x + (L / 2)) / W) - sigmoid((x - (L / 2)) / W)

def compute_pj(x, mu_k, K, L, W):
    # we assume that x has only one channel already
    # flatten spatial dims
    x = x.reshape(x.size(0), 1, -1)
    x = x.repeat(1, K, 1)  # construct K channels

    # apply activation functions
    return phi_k(x - mu_k, L, W)

class HistLayerBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.K = 256
        self.L = 1 / self.K  # 2 / K -> if values in [-1,1] (Paper)
        self.W = self.L / 2.5

        self.mu_k = (self.L * (torch.arange(self.K) + 0.5)).view(-1, 1)

class SingleDimHistLayer(HistLayerBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N = x.size(1) * x.size(2)
        pj = compute_pj(x, self.mu_k, self.K, self.L, self.W)
        return pj.sum(dim=2) / N

class JointHistLayer(HistLayerBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        N = x.size(1) * x.size(2)
        p1 = compute_pj(x, self.mu_k, self.K, self.L, self.W)
        p2 = compute_pj(y, self.mu_k, self.K, self.L, self.W)
        return torch.matmul(p1, torch.transpose(p2, 1, 2)) / N

if __name__ == '__main__':
    main()
