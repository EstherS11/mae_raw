import torch
import torch.nn as nn
from torch.nn import functional as F


import math
import numbers


def focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 2,
        reduction: str = 'None') -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, probs, targets): # logits: [b, c, h, w] targets:[b, c, h, w]
        num = targets.size(0)
        smooth = 1e-8

        # probs = torch.sigmoid(logits)
        intersection = (probs * targets)
        score = (2*torch.sum(intersection,dim=(2,3)) + smooth)/(torch.sum(probs*probs, dim=(2,3))+torch.sum(targets*targets, dim=(2,3)) + smooth)
        loss = 1 - score.sum()/num

        return loss


class BCELosswithLogits(nn.Module):
    def __init__(self, pos_weight=1):
        super(BCELosswithLogits, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        # logits: [N, *], target: [N, *]
        logits = torch.sigmoid(logits)
        loss = - self.pos_weight * target * torch.log(logits) - \
               (1 - target) * torch.log(1 - logits)
        return loss

class BCELoss(nn.Module):
    def __init__(self, pos_weight=1, reduction='mean'):
        super(BCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, *], target: [N, *]
        # logits = torch.sigmoid(logits)
        loss = - self.pos_weight * target * torch.log(logits) - \
               (1 - target) * torch.log(1 - logits)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        if self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6, *args, **kwargs):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'




class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class MSE_blur_loss(nn.Module):
    """MSE_blur Loss (L2)"""

    def __init__(self, channels=3, kernel_size=5, sigma=1, dim=2):
        super(MSE_blur_loss, self).__init__()
        self.Gaussian_blur = GaussianSmoothing(channels, kernel_size, sigma, dim)
        self.kernel_size = kernel_size
        self.cri_pix = nn.MSELoss()

    def forward(self, x, y):
        pad_size = self.kernel_size // 2
        x = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        x = self.Gaussian_blur(x)

        pad_size = self.kernel_size // 2
        y = F.pad(y, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        y = self.Gaussian_blur(y)

        loss = self.cri_pix(x, y)

        return loss