###########################################################################################
# Compilation utilities for MACE
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This code is pulled from the MACE repository, which is distributed under the MIT License
###########################################################################################
# Taken From:
# GitHub: https://github.com/ACEsuit/mace
# ArXiV: https://arxiv.org/pdf/2206.07697
# Date: August 27, 2024  |  12:37 (EST)
###########################################################################################
# NOTE: This file relates heavily to speedups in the MACE architecture which can be done by
#       compiling correctly. As a result, much of this code can be commented and MACE will
#       still function correctly. Specifically, only simplify_if_compile() and it's imports
#       are required. However, for optimal performance, it is recommended to keep the entire
#       file as-is.
###########################################################################################

from contextlib import contextmanager
from functools import wraps
from typing import Callable, Tuple

try:
    import torch._dynamo as dynamo
except ImportError:
    dynamo = None
from e3nn import get_optimization_defaults, set_optimization_defaults
from torch import autograd, nn
from torch.fx import symbolic_trace

ModuleFactory = Callable[..., nn.Module]
TypeTuple = Tuple[type, ...]


@contextmanager
def disable_e3nn_codegen():
    """Context manager that disables the legacy PyTorch code generation used in e3nn."""
    init_val = get_optimization_defaults()["jit_script_fx"]
    set_optimization_defaults(jit_script_fx=False)
    yield
    set_optimization_defaults(jit_script_fx=init_val)


def prepare(func: ModuleFactory, allow_autograd: bool = True) -> ModuleFactory:
    """Function transform that prepares a MACE module for torch.compile
    Args:
        func (ModuleFactory): A function that creates an nn.Module
        allow_autograd (bool, optional): Force inductor compiler to inline call to
            `torch.autograd.grad`. Defaults to True.
    Returns:
        ModuleFactory: Decorated function that creates a torch.compile compatible module
    """
    if allow_autograd:
        dynamo.allow_in_graph(autograd.grad)
    elif dynamo.allowed_functions.is_allowed(autograd.grad):
        dynamo.disallow_in_graph(autograd.grad)

    @wraps(func)
    def wrapper(*args, **kwargs):
        with disable_e3nn_codegen():
            model = func(*args, **kwargs)

        model = simplify(model)
        return model

    return wrapper


_SIMPLIFY_REGISTRY = set()


def simplify_if_compile(module: nn.Module) -> nn.Module:
    """Decorator to register a module for symbolic simplification

    The decorated module will be simplifed using `torch.fx.symbolic_trace`.
    This constrains the module to not have any dynamic control flow, see:

    https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing

    Args:
        module (nn.Module): the module to register

    Returns:
        nn.Module: registered module
    """
    _SIMPLIFY_REGISTRY.add(module)
    return module


# This is a different type of simplify() function than provided in e3nn. In e3nn, simplify()
# combines irreps, while this one simplifies the module for symbolic tracing.
def simplify(module: nn.Module) -> nn.Module:
    """Recursively searches for registered modules to simplify with
    `torch.fx.symbolic_trace` to support compiling with the PyTorch Dynamo compiler.

    Modules are registered with the `simplify_if_compile` decorator and

    Args:
        module (nn.Module): the module to simplify

    Returns:
        nn.Module: the simplified module
    """
    simplify_types = tuple(_SIMPLIFY_REGISTRY)

    for name, child in module.named_children():
        if isinstance(child, simplify_types):
            traced = symbolic_trace(child)
            setattr(module, name, traced)
        else:
            simplify(child)

    return module
