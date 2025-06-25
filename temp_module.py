from __future__ import annotations

import numpy as np
import random
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from collections import defaultdict
from typing import cast, Optional
from torch import Tensor, nn
from torch.utils.data import Sampler
from torch.nn.utils.parametrize import register_parametrization
from sklearn.metrics.pairwise import euclidean_distances


# %% Ensure4d
class Ensure4d(nn.Module):
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x

    def __call__(self, x):
        return self.forward(x)


# %% smart_padding
def smart_padding(kernel_size):
    if kernel_size % 2 == 0:
        return kernel_size // 2 - 1, kernel_size // 2, 0, 0
    else:
        return kernel_size // 2, kernel_size // 2, 0, 0


# %% smart_dilate_padding
def smart_dilate_padding(kernel_size: tuple[int, int], dilation: tuple[int, int]) -> tuple[int, int]:
    """
    Calculate padding size for 'same' convolution with dilation.
    """
    padding_height = ((kernel_size[0] - 1) * dilation[0]) // 2
    padding_width = ((kernel_size[1] - 1) * dilation[1]) // 2
    return padding_height, padding_width


# %% MaxNormParametrize
class MaxNormParametrize(nn.Module):
    """
    Enforce a max‑norm constraint on the rows of a weight tensor via parametrization.
    """
    def __init__(self, max_norm: float = 1.0):
        super().__init__()
        self.max_norm = max_norm

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # normalize each "row" (dim=0 slice) to have at most self.max_norm L2-norm
        return X.renorm(p=2, dim=0, maxnorm=self.max_norm)


# %% MaxNorm
class MaxNorm(nn.Module):
    def __init__(self, max_norm_val=2.0, eps=1e-5):
        super().__init__()
        self.max_norm_val = max_norm_val
        self.eps = eps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        norm = X.norm(2, dim=0, keepdim=True)
        denom = norm.clamp(min=self.max_norm_val / 2)
        number = denom.clamp(max=self.max_norm_val)
        return X * (number / (denom + self.eps))

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:
        # Assuming the forward scales X by a factor s,
        # the right inverse would scale it back by 1/s.
        norm = X.norm(2, dim=0, keepdim=True)
        denom = norm.clamp(min=self.max_norm_val / 2)
        number = denom.clamp(max=self.max_norm_val)
        scale = number / (denom + self.eps)
        return X / scale


# %% Conv2dWithConstraint
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm
        # initialize the weights
        nn.init.xavier_uniform_(self.weight, gain=1)
        register_parametrization(self, "weight", MaxNormParametrize(self.max_norm))


# %% LinearWithConstraint
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super(LinearWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm
        register_parametrization(self, "weight", MaxNormParametrize(self.max_norm))


# %% SqueezeFinalOutput
class SqueezeFinalOutput(nn.Module):
    """
    Removes empty dimension at the end and potentially removes empty time dimension.
    It does not just use squeeze as we never want to remove the first dimension.

    Returns
    -------
    x: torch.Tensor
        squeezed tensor
    """

    def __init__(self):
        super().__init__()
        self.squeeze = Rearrange("b c t 1 -> b c t")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) drop feature dim
        x = self.squeeze(x)
        # 2) drop time dim if singleton
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        return x

# %% glorot_weight_zero_bias
def glorot_weight_zero_bias(model):
    """
    Initialize parameters of all modules by initializing weights with glorot uniform/xavier initialization,
    and setting biases to zero. Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight"):
            if "BatchNorm" in module.__class__.__name__:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

# %% CausalConv1d
class CausalConv1d(nn.Conv1d):
    """
    Parameters
    ----------
    in_channels : int
        Input channels.
    out_channels : int
        Output channels (number of filters).
    kernel_size : int
        Kernel size.
    dilation : int, optional
        Dilation (number of elements to skip within kernel multiplication).
        Default to 1.
    **kwargs :
        Other keyword arguments to pass to torch.nn.Conv1d, except for `padding`!!

    References
    ----------
    [1] https://discuss.pytorch.org/t/causal-convolution/3456/4
    [2] https://gist.github.com/paultsw/7a9d6e3ce7b70e9e2c61bc9287addefc
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        **kwargs,
    ):
        assert "padding" not in kwargs, (
            f"The padding parameter is controlled internally by {type(self).__name__} class. "
            f"You should not try to override this parameter."
        )

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
            **kwargs,
        )

    def forward(self, X):
        out = F.conv1d(
            X,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return out[..., : -self.padding[0]]


# %% CombinedConv
class CombinedConv(nn.Module):
    """
    Merged convolutional layer for temporal and spatial convolutions in Deep4/ShallowFBCSP.
    Numerically equivalent to the separate sequential approach, but this should be faster.

    Parameters
    ----------
    in_chans: int
        Number of EEG input channels.
    n_filters_time: int
        Number of temporal filters.
    filter_time_length: int
        Length of the temporal filter.
    n_filters_spat: int
        Number of spatial filters.
    bias_time: bool
        Whether to use bias in the temporal conv
    bias_spat: bool
        Whether to use bias in the spatial conv
    """

    def __init__(
            self,
            in_chans,
            n_filters_time=40,
            n_filters_spat=40,
            filter_time_length=25,
            bias_time=True,
            bias_spat=True,
    ):
        super().__init__()
        self.bias_time = bias_time
        self.bias_spat = bias_spat
        self.conv_time = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters_time,
            kernel_size=(filter_time_length, 1),
            bias=bias_time,
            stride=1
        )
        self.conv_spat = nn.Conv2d(
            in_channels=n_filters_time,
            out_channels=n_filters_spat,
            kernel_size=(1, in_chans),
            bias=bias_spat,
            stride=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Merge time and spat weights
        combined_weight = self.conv_time.weight * self.conv_spat.weight.permute(1, 0, 2, 3).sum(0).unsqueeze(1)
        bias = None
        calculated_bias: Optional[torch.Tensor] = None

        # Calculate bias terms
        if self.bias_time:
            time_bias = self.conv_time.bias
            assert time_bias is not None
            calculated_bias = self.conv_spat.weight.squeeze().sum(-1).mm(time_bias.unsqueeze(-1)).squeeze()

        if self.bias_spat:
            spat_bias = self.conv_spat.bias
            assert spat_bias is not None
            if calculated_bias is None:
                calculated_bias = spat_bias
            else:
                calculated_bias = calculated_bias + spat_bias

        bias = calculated_bias

        return F.conv2d(x, weight=combined_weight, bias=bias, stride=(1, 1))


# %% Expression
class Expression(nn.Module):
    """
    Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super().__init__()
        self.expression_fn = expression_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expression_fn(x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(self.expression_fn, "kwargs"):
            expression_str = "{:s} {:s}".format(self.expression_fn.func.__name__, str(self.expression_fn.kwargs))
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return self.__class__.__name__ + "(expression=%s) " % expression_str


# %% safe_log
def safe_log(x, eps: float = 1e-6) -> torch.Tensor:
    """Prevents :math:`log(0)` by using :math:`log(max(x, eps))`."""
    return torch.log(torch.clamp(x, min=eps))


# %% square
def square(x):
    return x * x


# %% SafeLog
class SafeLog(nn.Module):
    r"""
    Safe logarithm activation function module.
    :math:\text{SafeLog}(x) = \log\left(\max(x, \epsilon)\right)

    Parameters
    ----------
    eps: float, optional
        A small value to clamp the input tensor to prevent computing log(0) or log of negative numbers.
        Default is 1e-6.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x) -> Tensor:
        """
        Forward pass of the SafeLog module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        ----------
        torch.Tensor
            Output tensor after applying safe logarithm.
        """
        return safe_log(x=x, eps=self.epsilon)

    def extra_repr(self) -> str:
        eps_str = f"eps={self.epsilon}"
        return eps_str


# %% CausalConv1d
class MaxNormLinear(nn.Linear):
    """
    Linear layer with MaxNorm constraining on weights.

    Parameters
    ----------
    in_features: int
        Size of each input sample.
    out_features: int
        Size of each output sample.
    bias: bool, optional
        If set to ``False``, the layer will not learn an additive bias.
        Default: ``True``.

    References
    ----------
    [1] https://keras.io/api/layers/core_layers/dense/#dense-class
    [2] https://www.tensorflow.org/api_docs/python/tf/keras/constraints/MaxNorm
    [3] https://discuss.pytorch.org/t/how-to-correctly-implement-in-place-max-norm-constraint/96769
    """

    def __init__(
        self, in_features, out_features, bias=True, max_norm_val=2, eps=1e-5, **kwargs
    ):
        super().__init__(
            in_features=in_features, out_features=out_features, bias=bias, **kwargs
        )
        self._max_norm_val = max_norm_val
        self._eps = eps
        register_parametrization(self, "weight", MaxNorm(self._max_norm_val, self._eps))


# %% DiscriminativeAlignmentLoss
class ClassAlignmentLoss(nn.Module):
    def __init__(
            self,
            intra_mode: str = 'center',
            verbose: bool = False,
            alpha: float = 1.0,
            loss_ablation: str = 'type_1'
    ):
        super(ClassAlignmentLoss, self).__init__()
        assert intra_mode in ['euclidean', 'cosine', 'center'], \
            "intra_mode must be one of ['euclidean', 'cosine', 'center']"
        self.intra_mode = intra_mode
        self.verbose = verbose
        if loss_ablation == "type_1":
            self.alpha = alpha
            self.beta = 1
            self.gama = 1
        elif loss_ablation == "type_2":
            self.alpha = 0
            self.beta = 1
            self.gama = 1
        elif loss_ablation == "type_3":
            self.alpha = 1
            self.beta = 0
            self.gama = 1
        elif loss_ablation == "type_4":
            self.alpha = 0
            self.beta = 0
            self.gama = 1

    @staticmethod
    def inter_domain_center_loss(features_list, labels_list):
        loss = 0.0
        num_domains = len(features_list)
        num_classes = torch.cat(labels_list).unique()

        centers = []
        for features, labels in zip(features_list, labels_list):
            domain_centers = []
            for c in num_classes:
                class_mask = labels == c
                if class_mask.sum() == 0:
                    domain_centers.append(torch.zeros_like(features[0]))
                else:
                    domain_centers.append(features[class_mask].mean(dim=0))
            centers.append(torch.stack(domain_centers))

        for i in range(num_domains):
            for j in range(i + 1, num_domains):
                loss += torch.norm(centers[i] - centers[j], p=2) / len(num_classes)

        return loss / (num_domains * (num_domains - 1) / 2)

    def forward(self, domain_features: list[torch.Tensor], domain_labels: list[torch.Tensor]):
        intra_compactness = 0.0
        intra_separability = 0.0

        for f, l in zip(domain_features, domain_labels):
            if self.intra_mode == 'euclidean':
                diff = f.unsqueeze(1) - f.unsqueeze(0)
                dist_matrix = torch.norm(diff, dim=2, p=2)
            elif self.intra_mode == 'cosine':
                f = F.normalize(f, p=2, dim=1)
                sim_matrix = torch.matmul(f, f.T)
                dist_matrix = 1 - sim_matrix
            elif self.intra_mode == 'center':
                unique_labels = l.unique()
                centers = []
                for c in unique_labels:
                    class_mask = (l == c)
                    class_center = f[class_mask].mean(dim=0)
                    centers.append(class_center)
                centers = torch.stack(centers)
                label_to_center = {int(c.item()): i for i, c in enumerate(unique_labels)}
                center_indices = torch.tensor([label_to_center[int(i.item())] for i in l], device=l.device)
                compactness = torch.norm(f - centers[center_indices], dim=1).mean()

                if centers.size(0) >= 2:
                    dist_centers = torch.cdist(centers, centers, p=2)
                    separability = dist_centers[~torch.eye(centers.size(0), dtype=torch.bool, device=f.device)].mean()
                else:
                    separability = torch.tensor(0.0, device=f.device)

                intra_compactness += compactness
                intra_separability += separability
                continue

            same_mask = l.unsqueeze(0) == l.unsqueeze(1)
            diff_mask = ~same_mask
            eye_mask = ~torch.eye(len(l), dtype=torch.bool, device=l.device)
            same_mask &= eye_mask
            diff_mask &= eye_mask

            compactness = dist_matrix[same_mask].mean()
            separability = dist_matrix[diff_mask].mean()

            intra_compactness += compactness
            intra_separability += separability

        intra_domain_loss = (self.beta * intra_compactness - self.alpha * intra_separability) / len(domain_features)

        inter_domain_loss = self.inter_domain_center_loss(domain_features, domain_labels)
        total_loss = self.gama * intra_domain_loss + inter_domain_loss

        if self.verbose:
            print(f"[ClassAlignmentLoss]")
            print(f"  Compactness     : {intra_compactness.item():.4f}")
            print(f"  Separability    : {intra_separability.item():.4f}")
            print(f"  Intra-domain    : {intra_domain_loss.item():.4f}")
            print(f"  Inter-domain    : {inter_domain_loss.item():.4f}")
            print(f"  Total           : {total_loss.item():.4f}")

        return total_loss