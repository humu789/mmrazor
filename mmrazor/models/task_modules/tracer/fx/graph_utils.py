# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any, List, Tuple

import torch.fx
from torch.ao.quantization.fake_quantize import FakeQuantizeBase


def _get_attrs(target: torch.nn.Module, attr: str) -> Any:
    """Get the attribute from target.

    Args:
        target (torch.nn.Module): Get the attribute from target module.
        attr (str): The target attribute.

    Returns:
        Any: The target attribute.
    """

    attrs: List[str] = attr.split('.')

    for att in attrs:
        target = getattr(target, att, None)
    return target


def del_fakequant_before_target(prepared_model: torch.fx.GraphModule,
                                target_patterns: Tuple,
                                inplace: bool = True) -> torch.fx.GraphModule:
    """Delete useless fakequant before nodes whose target attribute
    (node.target) is in `target_patterns`.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        target_patterns (tuple): Fakequants before nodes whose target attribute
            (node.target) is in `target_patterns` will be deleted.
        inplace (bool): Can optionally do the operation in-place. Defaults to
            True.

    Returns:
        GraphModule: Prepared standalone module after deletion.
    """

    def recursive_find_erased_nodes(node):
        """Find FakeQuant before target node recursively.

        Examples:
            head_fc = self.head.fc(activation_post_process_87);  \
                activation_post_process_87 = None
            activation_post_process_88 = \
                self.activation_post_process_88(head_fc);  head_fc = None
            head = self.head
            _get_loss = head._get_loss(activation_post_process_88,
                data_samples);  \
                head = activation_post_process_88 = data_samples = None
            return _get_loss

        node                       |           node.args
        --------------------
        output                     | (_get_loss, )
        _get_loss                  | (head, activation_post_process_88,
                                        data_samples)
        head                       | ()
        activation_post_process_88 | (head_fc, )
        data_samples               | (None, )
        """
        if node is None:
            return
        if node.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, node.target), FakeQuantizeBase):
            nodes_to_erase.append(node)
            return
        for prev_node in node.args:
            if isinstance(prev_node, torch.fx.Node):
                recursive_find_erased_nodes(prev_node)
        for prev_node in node.kwargs.values():
            if isinstance(prev_node, torch.fx.Node):
                recursive_find_erased_nodes(prev_node)
        return

    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if node.target in target_patterns:
            nodes_to_erase: List[torch.fx.Node] = []
            recursive_find_erased_nodes(node)
            for to_erase in nodes_to_erase:
                to_erase.replace_all_uses_with(to_erase.args[0])
                new_graph.erase_node(to_erase)
                delattr(prepared_model, to_erase.target)
    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def del_fakequant_after_target(prepared_model: torch.fx.GraphModule,
                               target_patterns: Tuple,
                               inplace: bool = True) -> torch.fx.GraphModule:
    """Delete useless fakequant after nodes whose target attribute
    (node.target) is in `target_patterns`.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        target_patterns (tuple): Fakequants after nodes whose target attribute
            (node.target) is in `target_patterns` will be deleted.
        inplace (bool): Can optionally do the operation in-place. Defaults to
            True.

    Returns:
        GraphModule: Prepared standalone module after deletion.
    """
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)

    target_nodes = []
    for node in new_graph.nodes:
        if node.target in target_patterns:
            target_nodes.append(node)

    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, node.target), FakeQuantizeBase):
            assert len(node.args) == 1
            prev_node = node.args[0]
            if prev_node not in target_nodes:
                continue
            node.replace_all_uses_with(prev_node)
            new_graph.erase_node(node)
            delattr(prepared_model, node.target)
    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def del_fakequant_before_module(prepared_model: torch.fx.GraphModule,
                                module_patterns: Tuple,
                                inplace: bool = True) -> torch.fx.GraphModule:
    """Delete useless fakequant before modules whose type are in
    `module_patterns`.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        target_patterns (tuple): Fakequants before modules whose type is in
            `module_patterns` will be deleted.
        inplace (bool): Can optionally do the operation in-place.
            Defaults to True.

    Returns:
        GraphModule: Prepared standalone module after deletion.
    """
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, node.target), module_patterns):
            to_erase = node.args[0]
            if not (to_erase.op == 'call_module' and isinstance(
                    _get_attrs(prepared_model, to_erase.target),
                    FakeQuantizeBase)):
                continue
            if len(to_erase.users) > 1:
                continue
            to_erase.replace_all_uses_with(to_erase.args[0])
            new_graph.erase_node(to_erase)
            delattr(prepared_model, to_erase.target)
    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def del_fakequant_after_module(prepared_model: torch.fx.GraphModule,
                               module_patterns: Tuple,
                               inplace: bool = True) -> torch.fx.GraphModule:
    """Delete useless fakequant after modules whose type are in
    `module_patterns`.

    Args:
        prepared_model (GraphModule): Prepared standalone module.
        target_patterns (tuple): Fakequants after modules whose type is in
            `module_patterns` will be deleted.
        inplace (bool): Can optionally do the operation in-place.
            Defaults to True.

    Returns:
        GraphModule: Prepared standalone module after deletion.
    """
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    target_nodes = []
    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, node.target), module_patterns):
            target_nodes.append(node)

    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, node.target), FakeQuantizeBase):
            assert len(node.args) == 1
            prev_node = node.args[0]
            if prev_node not in target_nodes:
                continue
            node.replace_all_uses_with(prev_node)
            new_graph.erase_node(node)
            delattr(prepared_model, node.target)
    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model
