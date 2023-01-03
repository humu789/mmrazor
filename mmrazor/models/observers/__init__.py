# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseObserver
from .lsq import LSQMinMaxObserver, LSQPerChannelMinMaxObserver
from .torch_observers import register_torch_observers

__all__ = ['BaseObserver', 'register_torch_observers', 'LSQMinMaxObserver',
           'LSQPerChannelMinMaxObserver']
