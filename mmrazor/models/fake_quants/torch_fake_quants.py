# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from typing import List

import torch.ao.quantization.fake_quantize as torch_fake_quant_src

from mmrazor.registry import MODELS


def register_torch_fake_quants() -> List[str]:
    """Register fake_quants in ``torch.ao.quantization.fake_quantize`` to the
    ``MODELS`` registry.

    Returns:
        List[str]: A list of registered fake_quants' name.
    """
    torch_fake_quants = []
    for module_name in dir(torch_fake_quant_src):
        if module_name.startswith('__') or module_name.startswith('_') or \
                                            module_name.startswith('default'):
            continue
        _fake_quant = getattr(torch_fake_quant_src, module_name)
        if inspect.isclass(_fake_quant) and issubclass(
                _fake_quant, torch_fake_quant_src.FakeQuantizeBase):
            if MODELS.get(module_name) is None:
                MODELS.register_module(module=_fake_quant)
                torch_fake_quants.append(module_name)
    return torch_fake_quants


TORCH_fake_quants = register_torch_fake_quants()
# TORCH_fake_quants including:
# FakeQuantize
# FakeQuantizeBase
# FixedQParamsFakeQuantize
# FusedMovingAvgObsFakeQuantize
