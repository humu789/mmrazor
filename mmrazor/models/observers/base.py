# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
from torch.ao.quantization.observer import UniformQuantizationObserverBase

from mmrazor.models.utils import pot_quantization, sync_tensor

# from mmengine.model import BaseModule


class BaseObserver(UniformQuantizationObserverBase):
    """Modified torch quantization observer.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
            reference model spec.
        qscheme: Quantization scheme to be used.
        reduce_range: Reduces the range of the quantized data type by 1 bit.
            This is sometimes required to avoid instruction overflow.
        quant_min: Minimum quantization value. If unspecified, it will follow
            the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow
            the 8-bit setup.
        ch_axis (int, optional): Channel axis index. Defaults to -1.
        is_pot_scale (bool, optional):  Indicate whether scale is power of two.
            Defaults to False.
        eps: Epsilon value for float32.
            Defaults to `torch.finfo(torch.float32).eps`.
    """

    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self,
                 dtype=torch.quint8,
                 qscheme=torch.per_tensor_affine,
                 reduce_range=False,
                 quant_min=None,
                 quant_max=None,
                 ch_axis=-1,
                 is_pot_scale=False,
                 factory_kwargs=None,
                 eps=torch.finfo(torch.float32).eps) -> None:
        super().__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                         factory_kwargs, eps)
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer('min_val',
                             torch.tensor(float('inf'), **factory_kwargs))
        self.register_buffer('max_val',
                             torch.tensor(float('-inf'), **factory_kwargs))
        self.ch_axis = ch_axis
        self.is_pot_scale = is_pot_scale

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        if self.is_pot_scale:
            scale = pot_quantization(scale)
        return scale, zero_point

    @torch.jit.export
    def extra_repr(self):
        return 'min_val={}, max_val={} ch_axis={} is_pot_scale={}'.format(
            self.min_val, self.max_val, self.ch_axis, self.is_pot_scale)

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        self.min_val.copy_(torch.tensor(float('inf')))
        self.max_val.copy_(torch.tensor(float('-inf')))
