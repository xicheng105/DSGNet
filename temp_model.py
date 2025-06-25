from __future__ import annotations

import warnings
import math
import torch
import numpy as np

from collections import OrderedDict
from typing import Dict, Iterable, Optional
from docstring_inheritance import NumpyDocstringInheritanceInitMeta
from torchinfo import ModelStatistics, summary
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from mne.utils import warn
from einops.layers.torch import Rearrange

from modules import Ensure4d, smart_padding, Conv2dWithConstraint, LinearWithConstraint, SqueezeFinalOutput
from modules import glorot_weight_zero_bias, smart_dilate_padding, CausalConv1d, MaxNormLinear, CombinedConv
from modules import Expression, SafeLog, square


# %% models base
# This module was modified from:
# https://github.com/braindecode/braindecode/blob/master/braindecode/models/base.py#L35

def deprecated_args(obj, *old_new_args):
    out_args = []
    for old_name, new_name, old_val, new_val in old_new_args:
        if old_val is None:
            out_args.append(new_val)
        else:
            warnings.warn(
                f"{obj.__class__.__name__}: {old_name!r} is depreciated. Use {new_name!r} instead."
            )
            if new_val is not None:
                raise ValueError(
                    f"{obj.__class__.__name__}: Both {old_name!r} and {new_name!r} were specified."
                )
            out_args.append(old_val)
    return out_args

class EEGModuleMixin(metaclass=NumpyDocstringInheritanceInitMeta):
    """
    Mixin class for all EEG models in braindecode.

    Parameters
    ----------
    n_outputs : int
        Number of outputs of the model. This is the number of classes in the case of classification.
    n_electrodes : int
        Number of EEG electrodes.
    electrodes_info : list of dict
        Information about each EEG electrode. This should be filled with "info["chs"]".
        Refer to: class:`mne.Info` for more details.
    n_timepoints : int
        Number of sampling points of the input window.
    input_window_seconds : float
        Length of the input window in seconds.
    sfreq : float
        Sampling frequency of the EEG recordings.

    Raises
    ------
    ValueError: If some input signal-related parameters are not specified and cannot be inferred.

    Notes
    -----
    If some input signal-related parameters are not specified, there will be an attempt to infer
    them from the other parameters.
    """

    def __init__(
            self,
            n_outputs: Optional[int] = None,
            n_electrodes: Optional[int] = None,
            electrodes_info = None,
            n_timepoints: Optional[int] = None,
            input_window_seconds: Optional[float] = None,
            sfreq: Optional[float] = None
    ):
        if n_electrodes is not None and electrodes_info is not None and len(electrodes_info) != n_electrodes:
            raise ValueError(f"{n_electrodes=} different from {electrodes_info=} length")
        if (
            n_timepoints is not None
            and input_window_seconds is not None
            and sfreq is not None
            and n_timepoints != int(input_window_seconds * sfreq)
        ):
            raise ValueError(f"{n_timepoints=} different from {input_window_seconds=} * {sfreq=}")

        self._input_window_seconds = input_window_seconds
        self._electrodes_info = electrodes_info
        self._n_outputs = n_outputs
        self._n_electrodes = n_electrodes
        self._n_timepoints = n_timepoints
        self._sfreq = sfreq

        super().__init__()

    @property
    def n_outputs(self) -> int:
        if self._n_outputs is None:
            raise ValueError("n_outputs not specified.")
        return self._n_outputs

    @property
    def n_electrodes(self) -> int:
        if self._n_electrodes is None and self._electrodes_info is not None:
            return len(self._electrodes_info)
        elif self._n_electrodes is None:
            raise ValueError("n_electrodes could not be inferred. Either specify n_electrodes or electrodes_info.")
        return self._n_electrodes

    @property
    def electrodes_info(self) -> list[str]:
        if self._electrodes_info is None:
            raise ValueError("electrodes_info not specified.")
        return self._electrodes_info

    @property
    def n_timepoints(self) -> int:
        if (
            self._n_timepoints is None
            and self._input_window_seconds is not None
            and self._sfreq is not None
        ):
            return int(self._input_window_seconds * self._sfreq)
        elif self._n_timepoints is None:
            raise ValueError(
                "n_timepoints could not be inferred. Either specify n_timepoints or input_window_seconds and sfreq."
            )
        return self._n_timepoints

    @property
    def sfreq(self) -> float:
        if (
            self._sfreq is None
            and self._input_window_seconds is not None
            and self._n_timepoints is not None
        ):
            return float(self._n_timepoints / self._input_window_seconds)
        elif self._sfreq is None:
            raise ValueError(
                "sfreq could not be inferred. Either specify sfreq or input_window_seconds and n_timepoints."
            )
        return self._sfreq

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return 1, self.n_electrodes, self.n_timepoints

    def get_output_shape(self) -> tuple[int, ...]:
        """
        Returns
        -------
        output_shape: tuple[int, ...]
            shape of the network output for `batch_size==1` (1, ...)
        :return:
        """

        with torch.inference_mode():
            try:
                return tuple(
                    self.forward(
                        torch.zeros(
                            self.input_shape,
                            dtype=next(self.parameters()).dtype,
                            device=next(self.parameters()).device
                        )
                    ).shape
                )
            except RuntimeError as exc:
                if str(exc).endswith(
                    "Output size is too small, Kernel size can't be greater than actual input size."
                ):
                    msg = (
                        "During model prediction RuntimeError was thrown showing that at some "
                        f"layer `{str(exc).split('.')[-1]}` (see above in the stacktrace). "
                        "This could be caused by providing too small `n_timepoints`/`input_window_seconds`. "
                        f"Model may require longer chunks of signal in the input than {self.input_shape}."
                    )
                    raise ValueError(msg) from exc
                raise exc

    mapping: Optional[Dict[str, str]] = None

    def load_state_dict(self, state_dict, *args, **kwargs):
        mapping = self.mapping if self.mapping else {}
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k in mapping:
                new_state_dict[mapping[k]] = v
            else:
                new_state_dict[k] = v

        return super().load_state_dict(new_state_dict, *args, **kwargs)

    def to_dense_prediction_model(self, axis: tuple[int, ...] | int = (2, 3)) -> None:
        """
        Transform a sequential model with strides to a model that outputs dense predictions by
        removing the strides and instead inserting dilation's. Modifies model in-place.

        Parameters
        ----------
        axis: int or (int,int)
            Axis to transform (in terms of intermediate output axes)
            can either be 2, 3, or (2,3).

        Notes
        -----
        Does not yet work correctly for average pooling.
        Prior to version 0.1.7, there had been a bug that could move strides backwards one layer.
        """

        if not hasattr(axis, "__iter__"):
            axis = (axis,)
        assert all([ax in [2, 3] for ax in axis]), "Only 2 and 3 allowed for axis"  # type: ignore[union-attr]
        axis = np.array(axis) - 2
        stride_so_far = np.array([1, 1])
        for module in self.modules():
            if hasattr(module, "dilation"):
                assert module.dilation == 1 or (module.dilation == (1, 1)), (
                    "Dilation should equal 1 before conversion, maybe the model is already converted?"
                )
                new_dilation = [1, 1]
                for ax in axis:
                    new_dilation[ax] = int(stride_so_far[ax])
                module.dilation = tuple(new_dilation)
            if hasattr(module, "stride"):
                if not hasattr(module.stride, "__len__"):
                    module.stride = (module.stride, module.stride)
                stride_so_far *= np.array(module.stride)
                new_stride = list(module.stride)
                for ax in axis:  # type: ignore[union-attr]
                    new_stride[ax] = 1
                module.stride = tuple(new_stride)

    def get_torchinfo_statistics(
            self,
            col_names: Optional[Iterable[str]] = (
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
            ),
            row_settings: Optional[Iterable[str]] = ("var_names", "depth"),
    ) -> ModelStatistics:
        """Generate table describing the model using torchinfo.summary.

        Parameters
        ----------
        col_names : tuple, optional
            Specify which columns to show in the output, see torchinfo for details, by default
            ("input_size", "output_size", "num_params", "kernel_size")
        row_settings : tuple, optional
             Specify which features to show in a row, see torchinfo for details, by default
             ("var_names", "depth")

        Returns
        -------
        torchinfo.ModelStatistics
            ModelStatistics generated by torchinfo.summary.
        """
        return summary(
            self,
            input_size=(1, self.n_electrodes, self.n_timepoints),
            col_names=col_names,
            row_settings=row_settings,
            verbose=0,
        )

    def __str__(self) -> str:
        return str(self.get_torchinfo_statistics())

# %% DDANet
class DDANet(EEGModuleMixin, nn.Module):

    def __init__(
            self,
            n_timepoints: Optional[int] = None,
            n_electrodes: Optional[int] = None,
            n_outputs: Optional[int] = None,
            # Global convolution parameters.
            seize: int = 2,
            depth: int = 4,
            stride: int = 1,
            n_windows: int = 6,
            kernel_size: int = 4,
            dropout_rate: float = 0.25,
            model_ablation: str = "both_branch",    # [temporal_branch, spectral_branch, both_branch]
            activation: nn.Module = nn.ELU,
            # Other ways to construct the signal related parameters
            electrodes_info: Optional[list[Dict]] = None,
            input_window_seconds=None,
            sfreq=None,
            **kwargs
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_electrodes=n_electrodes,
            electrodes_info=electrodes_info,
            n_timepoints=n_timepoints,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq
        )
        del n_outputs, n_electrodes, electrodes_info, n_timepoints, input_window_seconds, sfreq

        self.n_windows = n_windows
        self.seize = seize
        self.depth = depth
        self.stride = stride
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.model_ablation = model_ablation

        self.ensure_dimension = Ensure4d()
        self.dimension_shuffle = Rearrange("b e t 1 -> b 1 e t")

        self.temporal_conv_block = _LC_Block(
            n_timepoints=self.n_timepoints,
            n_electrodes=self.n_electrodes,
            filter_1=8,
            kernel_length=64,
            pool_mode='average'
        )

        self.spectral_conv_block = _LC_Block(
            n_timepoints=self.n_timepoints,
            n_electrodes=self.n_electrodes,
            filter_1=16,
            kernel_length=96,
            pool_mode='max'
        )

        c_temporal, t_temporal, c_spectral, t_spectral = self.get_temporal_spectral_shape(
            dummy_input_shape=(1, self.n_electrodes, self.n_timepoints)
        )

        self.temporal_se_blocks = nn.ModuleList([
            _SE_Block(in_features=c_temporal, seize=self.seize, SE_dimension="temporal")
            for _ in range(self.n_windows)
        ])

        self.spectral_se_blocks = nn.ModuleList([
            _SE_Block(in_features=t_spectral, seize=self.seize, SE_dimension="spectral")
            for _ in range(self.n_windows)
        ])

        self.temporal_dila_cas_conv_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    CausalConv1d(
                        in_channels=c_temporal,
                        out_channels=c_temporal,
                        kernel_size=self.kernel_size,
                        dilation=d + 1
                    ),
                    nn.BatchNorm1d(num_features=c_temporal)
                ) for d in range(self.depth)
            ]) for _ in range(self.n_windows)
        ])

        self.spectral_dila_cas_conv_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    CausalConv1d(
                        in_channels=t_spectral,
                        out_channels=t_spectral,
                        kernel_size=self.kernel_size,
                        dilation=d + 1
                    ),
                    nn.BatchNorm1d(num_features=t_spectral)
                ) for d in range(self.depth)
            ]) for _ in range(self.n_windows)
        ])

        if self.model_ablation == "temporal_branch":
            in_features = self.n_windows * c_temporal * t_temporal
        elif self.model_ablation == "spectral_branch":
            in_features = self.n_windows * c_spectral * t_spectral
        else:
            in_features = self.n_windows * (c_temporal * t_temporal + c_spectral * t_spectral)

        self.remodel_layer = MaxNormLinear(
            in_features=in_features,
            out_features=100,
            max_norm_val=0.25
        )
        self.full_connected_layer = MaxNormLinear(
            in_features=100,
            out_features=self.n_outputs,
            max_norm_val=0.25
        )

    def forward(self, x):
        x = self.ensure_dimension(x)
        x = self.dimension_shuffle(x)

        # temporal branch
        x_temporal = self.temporal_conv_block(x)
        B_temporal, C_temporal, T_temporal = x_temporal.shape
        temporal_sw_concat = []
        for j in range(self.n_windows):
            start_point = j * self.stride
            end_point = T_temporal - (self.n_windows - j - 1) * self.stride
            x_temporal_windows = x_temporal[:, :, start_point:end_point]  # (B, C, L)
            x_temporal_SE = self.temporal_se_blocks[j](x_temporal_windows)
            x_temporal_last_block = x_temporal_SE
            for d in range(self.depth):
                conv = self.temporal_dila_cas_conv_blocks[j][d]
                out = conv(x_temporal_last_block)
                out = self.activation()(out)
                out = F.dropout(out, p=self.dropout_rate, training=self.training)
                x_temporal_last_block = self.activation()(out + x_temporal_last_block)
            temporal_sw_concat.append(x_temporal_last_block.flatten(start_dim=1))

        # spectral branch
        x_spectral = self.spectral_conv_block(x)
        B_spectral, C_spectral, T_spectral = x_spectral.shape
        spectral_sw_concat = []
        for j in range(self.n_windows):
            start_point = j * self.stride
            end_point = C_spectral - (self.n_windows - j - 1) * self.stride
            x_spectral_windows = x_spectral[:, start_point:end_point, :]  # (B, L, T)
            x_spectral_SE = self.spectral_se_blocks[j](x_spectral_windows)
            x_spectral_SE = x_spectral_SE.permute(0, 2, 1)  # (B, T, L)
            x_spectral_last_block = x_spectral_SE
            for d in range(self.depth):
                conv = self.spectral_dila_cas_conv_blocks[j][d]
                out = conv(x_spectral_last_block)
                out = self.activation()(out)
                out = F.dropout(out, p=self.dropout_rate, training=self.training)
                x_spectral_last_block = self.activation()(out + x_spectral_last_block)
            spectral_sw_concat.append(x_spectral_last_block.flatten(start_dim=1))

        # concatenate
        temporal_concat_tensor = torch.cat(temporal_sw_concat, dim=1)
        spectral_concat_tensor = torch.cat(spectral_sw_concat, dim=1)
        if self.model_ablation == "temporal_branch":
            x_concatenated = temporal_concat_tensor
        elif self.model_ablation == "spectral_branch":
            x_concatenated = spectral_concat_tensor
        else:
            x_concatenated = torch.cat([temporal_concat_tensor, spectral_concat_tensor], dim=1)
        x_remodel = self.remodel_layer(x_concatenated)
        x_out = self.full_connected_layer(x_remodel)

        return x_out

    def get_temporal_spectral_shape(self, dummy_input_shape):
        device = next(self.parameters()).device
        dummy_input = torch.zeros(*dummy_input_shape).to(device)

        with torch.no_grad():
            dummy_input = self.ensure_dimension(dummy_input)
            dummy_input = self.dimension_shuffle(dummy_input)

            x_temporal = self.temporal_conv_block(dummy_input)
            B_temporal, C_temporal, T_temporal = x_temporal.shape
            x_temporal_windows = x_temporal[:, :, 0:T_temporal - (self.n_windows - 1) * self.stride]

            x_spectral = self.spectral_conv_block(dummy_input)
            B_spectral, C_spectral, T_spectral = x_spectral.shape
            x_spectral_windows = x_spectral[:, 0:C_spectral - (self.n_windows - 1) * self.stride, :]

            return x_temporal_windows.shape[1], x_temporal_windows.shape[-1], x_spectral_windows.shape[1], x_spectral_windows.shape[-1]


class _LC_Block(nn.Module):
    def __init__(
            self,
            n_timepoints: Optional[int] = None,
            n_electrodes: Optional[int] = None,
            n_outputs: Optional[int] = None,
            filter_1: int = 8,
            depth: int = 2,
            filter_2: Optional[int | None] = None,
            kernel_length: int = 64,
            pool_mode: str = "average",
            depthwise_kernel_length: Optional[int] = None,
            pool1_kernel_size: Optional[int] = None,
            pool2_kernel_size: Optional[int] = None,
            dropout_rate: float = 0.25,
            activation: nn.Module = nn.ELU
    ):
        super().__init__()

        self.n_timepoints = n_timepoints
        self.n_electrodes = n_electrodes
        self.n_outputs = n_outputs
        self.kernel_length = kernel_length
        self.pool_mode = pool_mode
        self.filter_1 = filter_1
        self.depth = depth
        if filter_2 is None:
            filter_2 = self.filter_1 * self.depth
        if pool1_kernel_size is None:
            pool1_kernel_size = self.kernel_length // 16
        if pool2_kernel_size is None:
            pool2_kernel_size = self.kernel_length // 8
        if depthwise_kernel_length is None:
            depthwise_kernel_length = self.kernel_length // 4
        self.filter_2 = filter_2
        self.pool1_kernel_size = pool1_kernel_size
        self.pool2_kernel_size = pool2_kernel_size
        self.depthwise_kernel_length = depthwise_kernel_length
        self.dropout_rate = dropout_rate
        self.activation = activation

        pool_class = dict(max=nn.MaxPool2d, average=nn.AvgPool2d)[self.pool_mode]

        self.conv_block_1 = nn.Sequential(
            nn.ZeroPad2d(smart_padding(self.kernel_length)),
            nn.Conv2d(
                in_channels=1,
                out_channels=self.filter_1,
                kernel_size=(1, self.kernel_length),
                bias=False,
                padding=0
            ),
            nn.BatchNorm2d(self.filter_1)
        )

        self.conv_block_2 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=self.filter_1,
                out_channels=self.filter_2,
                kernel_size=(self.n_electrodes, 1),
                max_norm=1,
                bias=False,
                groups=self.filter_1
            ),
            nn.BatchNorm2d(self.filter_2),
            activation(),
            pool_class(kernel_size=(1, self.pool1_kernel_size)),
            nn.Dropout(self.dropout_rate)
        )

        self.conv_block_3 = nn.Sequential(
            nn.ZeroPad2d(smart_padding(self.depthwise_kernel_length)),
            nn.Conv2d(
                in_channels=self.filter_2,
                out_channels=self.filter_2,
                kernel_size=(1, self.depthwise_kernel_length),
                bias=False,
                groups=self.filter_2,
                padding=0
            ),
            nn.Conv2d(
                in_channels=self.filter_2,
                out_channels=self.filter_2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(self.filter_2),
            activation(),
            pool_class(kernel_size=(1, self.pool2_kernel_size)),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = x.squeeze(2)

        return x


class _SE_Block(nn.Module):
    def __init__(
            self,
            in_features: int,
            seize: int = 2,
            activation_1: nn.Module = nn.ReLU,
            activation_2: nn.Module = nn.Sigmoid,
            SE_dimension: str = "temporal",
    ):
        super().__init__()
        self.SE_dimension = SE_dimension
        self.activation_1 = activation_1
        self.activation_2 = activation_2
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.squeeze_fc1 = nn.Linear(in_features, seize, bias=False)
        self.squeeze_fc2 = nn.Linear(seize, in_features, bias=False)

    def forward(self, x):
        if self.SE_dimension == 'temporal':
            x_pooled = self.adaptive_pool(x).squeeze(-1)  # (B, C)
        else:
            x_perm = x.permute(0, 2, 1)  # (B, T, C)
            x_pooled = self.adaptive_pool(x_perm).squeeze(-1)  # (B, T)

        out = self.activation_1()(self.squeeze_fc1(x_pooled))
        out = self.activation_2()(self.squeeze_fc2(out)).unsqueeze(-1)

        if self.SE_dimension == 'temporal':
            return x * out  # (B, C, T)
        else:
            return x * out.permute(0, 2, 1)  # (B, C, T)


# %% Dual-branch Subject-aligned Generalization Network, DSGNet
class DSGNet(EEGModuleMixin, nn.Module):

    def __init__(
            self,
            n_timepoints: Optional[int] = None,
            n_electrodes: Optional[int] = None,
            n_outputs: Optional[int] = None,
            # Global convolution parameters.
            seize: int = 2,
            depth: int = 4,
            stride: int = 1,
            n_windows: int = 6,
            kernel_size: int = 4,
            dropout_rate: float = 0.25,
            model_ablation: str = "type_1",
            activation: nn.Module = nn.ELU,
            # Other ways to construct the signal related parameters
            electrodes_info: Optional[list[Dict]] = None,
            input_window_seconds=None,
            sfreq=None,
            **kwargs
    ):
        super().__init__(
            n_outputs=n_outputs,
            n_electrodes=n_electrodes,
            electrodes_info=electrodes_info,
            n_timepoints=n_timepoints,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq
        )
        del n_outputs, n_electrodes, electrodes_info, n_timepoints, input_window_seconds, sfreq

        self.n_windows = n_windows
        self.seize = seize
        self.depth = depth
        self.stride = stride
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.model_ablation = model_ablation

        self.ensure_dimension = Ensure4d()
        self.dimension_shuffle = Rearrange("b e t 1 -> b 1 e t")

        self.temporal_conv_block = _LC_Block(
            n_timepoints=self.n_timepoints,
            n_electrodes=self.n_electrodes,
            filter_1=8,
            kernel_length=64,
            pool_mode='average'
        )

        self.spectral_conv_block = _LC_Block(
            n_timepoints=self.n_timepoints,
            n_electrodes=self.n_electrodes,
            filter_1=16,
            kernel_length=96,
            pool_mode='max'
        )

        c_temporal, t_temporal, c_spectral, t_spectral = self.get_temporal_spectral_shape(
            dummy_input_shape=(1, self.n_electrodes, self.n_timepoints)
        )

        self.temporal_se_blocks = nn.ModuleList([
            _SE_Block(in_features=c_temporal, seize=self.seize, SE_dimension="temporal")
            for _ in range(self.n_windows)
        ])

        self.spectral_se_blocks = nn.ModuleList([
            _SE_Block(in_features=t_spectral, seize=self.seize, SE_dimension="spectral")
            for _ in range(self.n_windows)
        ])

        self.temporal_dila_cas_conv_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    CausalConv1d(
                        in_channels=c_temporal,
                        out_channels=c_temporal,
                        kernel_size=self.kernel_size,
                        dilation=d + 1
                    ),
                    nn.BatchNorm1d(num_features=c_temporal)
                ) for d in range(self.depth)
            ]) for _ in range(self.n_windows)
        ])

        self.spectral_dila_cas_conv_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    CausalConv1d(
                        in_channels=t_spectral,
                        out_channels=t_spectral,
                        kernel_size=self.kernel_size,
                        dilation=d + 1
                    ),
                    nn.BatchNorm1d(num_features=t_spectral)
                ) for d in range(self.depth)
            ]) for _ in range(self.n_windows)
        ])

        if self.model_ablation == "type_2":
            in_features = c_temporal * t_temporal + c_spectral * t_spectral
        elif self.model_ablation == "type_3":
            in_features = self.n_windows * c_temporal * t_temporal
        else:
            in_features = self.n_windows * (c_temporal * t_temporal + c_spectral * t_spectral)
        self.full_connected_layer = MaxNormLinear(
            in_features=in_features,
            out_features=self.n_outputs,
            max_norm_val=0.25
        )

    def forward(self, data_list):
        data = torch.cat([d.to(torch.float32) for d in data_list], dim=0)
        domain_sizes = [d.shape[0] for d in data_list]

        x = self.ensure_dimension(data)
        x = self.dimension_shuffle(x)

        # temporal branch
        x_temporal = self.temporal_conv_block(x)
        B_temporal, C_temporal, T_temporal = x_temporal.shape
        temporal_sw_concat = []
        for j in range(self.n_windows):
            start_point = j * self.stride
            end_point = T_temporal - (self.n_windows - j - 1) * self.stride
            x_temporal_windows = x_temporal[:, :, start_point:end_point]  # (B, C, L)
            x_temporal_SE = self.temporal_se_blocks[j](x_temporal_windows)
            if self.model_ablation == "type_1":
                x_temporal_last_block = x_temporal_windows
            else:
                x_temporal_last_block = x_temporal_SE
            for d in range(self.depth):
                conv = self.temporal_dila_cas_conv_blocks[j][d]
                out = conv(x_temporal_last_block)
                out = self.activation()(out)
                out = F.dropout(out, p=self.dropout_rate, training=self.training)
                x_temporal_last_block = self.activation()(out + x_temporal_last_block)
            temporal_sw_concat.append(x_temporal_last_block.flatten(start_dim=1))

        # spectral branch
        x_spectral = self.spectral_conv_block(x)
        B_spectral, C_spectral, T_spectral = x_spectral.shape
        spectral_sw_concat = []
        for j in range(self.n_windows):
            start_point = j * self.stride
            end_point = C_spectral - (self.n_windows - j - 1) * self.stride
            x_spectral_windows = x_spectral[:, start_point:end_point, :]  # (B, L, T)
            x_spectral_SE = self.spectral_se_blocks[j](x_spectral_windows)
            x_spectral_SE = x_spectral_SE.permute(0, 2, 1)  # (B, T, L)
            if self.model_ablation == "type_1":
                x_spectral_last_block = x_spectral_windows.permute(0, 2, 1)
            else:
                x_spectral_last_block = x_spectral_SE
            for d in range(self.depth):
                conv = self.spectral_dila_cas_conv_blocks[j][d]
                out = conv(x_spectral_last_block)
                out = self.activation()(out)
                out = F.dropout(out, p=self.dropout_rate, training=self.training)
                x_spectral_last_block = self.activation()(out + x_spectral_last_block)
            spectral_sw_concat.append(x_spectral_last_block.flatten(start_dim=1))

        # concatenate
        if self.model_ablation in ["type_1", "type_4"]:
            temporal_concat_tensor = torch.cat(temporal_sw_concat, dim=1)
            spectral_concat_tensor = torch.cat(spectral_sw_concat, dim=1)
            x_concatenated = torch.cat([temporal_concat_tensor, spectral_concat_tensor], dim=1)
        elif self.model_ablation == "type_2":
            x_concatenated = torch.cat([x_temporal.flatten(start_dim=1), x_spectral.flatten(start_dim=1)], dim=1)
        else:
            x_concatenated = torch.cat(temporal_sw_concat, dim=1)
        # x_remodel = self.remodel_layer(x_concatenated)
        x_out = self.full_connected_layer(x_concatenated)

        start = 0
        dist_features = []
        for size in domain_sizes:
            dist_features.append(x_concatenated[start:start + size])
            start += size

        return x_out, dist_features

    def get_temporal_spectral_shape(self, dummy_input_shape):
        device = next(self.parameters()).device
        dummy_input = torch.zeros(*dummy_input_shape).to(device)

        with torch.no_grad():
            dummy_input = self.ensure_dimension(dummy_input)
            dummy_input = self.dimension_shuffle(dummy_input)

            x_temporal = self.temporal_conv_block(dummy_input)
            B_temporal, C_temporal, T_temporal = x_temporal.shape

            x_spectral = self.spectral_conv_block(dummy_input)
            B_spectral, C_spectral, T_spectral = x_spectral.shape

            if self.model_ablation == "type_2":
                return C_temporal, T_temporal, C_spectral, T_spectral
            else:
                x_temporal_windows = x_temporal[:, :, 0:T_temporal - (self.n_windows - 1) * self.stride]
                x_spectral_windows = x_spectral[:, 0:C_spectral - (self.n_windows - 1) * self.stride, :]

                return x_temporal_windows.shape[1], x_temporal_windows.shape[-1], x_spectral_windows.shape[1], x_spectral_windows.shape[-1]

