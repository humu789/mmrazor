# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
import torch.nn as nn

from mmrazor.models.task_modules import CustomTracer, UntracedMethodRegistry, build_graphmodule
from mmrazor.testing import ConvBNReLU
from mmrazor.structures.quantization import BackendConfigs, QConfigHander


class testCustomTracer(TestCase):

    def test_init(self):
        tracer = CustomTracer()
        assert tracer.skipped_methods.__len__() == 0

    def test_trace(self):
        tracer = CustomTracer()
        model = ConvBNReLU(3, 3, norm_cfg=dict(type='BN'))
        graph = tracer.trace(model)  # noqa: F841

    def test_auto_skip_call_module(self):
        pass

    def test_auto_skip_call_method(self):
        pass

    def test_configurable_skipped_methods(self):
        pass


class testUntracedMethodRgistry(TestCase):

    def test_init(self):
        self.assertEqual(len(UntracedMethodRegistry.method_dict), 0)

    def test_add_method(self):
        pass


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.fc = nn.Linear(3, 4)

    def _get_predictions(self, x):
        # something can not be traced
        pass

    def forward(self, x):
        x = self.fc(x)
        self._get_predictions(x)
        return x


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.stem_layer = nn.Sequential(
            nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3), nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = Head()

    def forward(self, x):
        x = self.stem_layer(x)
        x = self.maxpool(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.head(x)
        return x


class TestBuildGraphModule(TestCase):

    def setUp(self):
        self.tracer = CustomTracer(
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_predictions'
            ])

    def test_build_graph_module(self):
        pass
