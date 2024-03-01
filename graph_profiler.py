from enum import Enum
from typing import Dict
import torch
import torch.fx as fx
from typing import Dict, Any

class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"

class GraphProfiler(fx.Interpreter):

    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)

    def meta_run(self, *args) -> Any:
        args_iter = iter(args)
        for n in self.module.graph.nodes:
            if n.op == OP.PLACEHOLDER:
                self.env[n] = next(args_iter)
        args = None
        return self.run([])

    def run(self, *args, initial_env: Dict[fx.Node, Any] | None = None, enable_io_processing: bool = True) -> torch.Any:
        return super().run(*args, initial_env=initial_env, enable_io_processing=enable_io_processing)
    
    def run_node(self, n: fx.Node) -> Any:

        result = super().run_node(n)

        return result