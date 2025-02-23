"""
In the initial phase, our primary task is to construct a comprehensive computational graph. This
graph will encapsulate all operations from the forward, backward, and optimizer steps within a
single iteration. The nodes within this graph symbolize individual operations, while the edges
represent the dependencies between input and output data. The profiler's job is :
    1. Collecting data on the computation time and memory usage of each operator when the
    graph operations are executed in topological order.
        # TODO ask if these have to be disaggregated by type?
    2. Categorizing the inputs and outputs of each operation as a parameter, gradient,
    activation, optimizer state, or other types.
    3. Conducting static data analysis on activations by documenting the first and last use of
    each activation during the forward and backward passes.
    4. Generating a peak memory breakdown graph using the collected statistics.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Any, final
import torch
import torch.fx as fx

class OP(str, Enum):
    CALL_FUNCTION = "call_function"
    CALL_MODULE = "call_module"
    CALL_METHOD = "call_method"
    GET_ATTR = "get_attr"
    OUTPUT = "output"
    PLACEHOLDER = "placeholder"


class NodeType(Enum):
    """
    NodeType is a enum that records the type of the tensors in the graph.
    """

    PARAM = 0
    ACT = 1
    GRAD = 2
    OTHER = 3


@dataclass
class ProfilerStatistics:
    time_per_run: int

# This is an example graph_profiler that extends the fx.Interpreter class, it
# will perform graph execution by running the graph node by node.


class GraphProfiler(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, garbage_collect_values: bool = True):
        super().__init__(module, garbage_collect_values)
        
        # Fields for statistics
        self.num_runs: int = 0
        self.total_time_elapsed = 0     # in ms
        self.all_aggregated_stats: ProfilerStatistics = None

        # You should perform the static analysis of the graph here. In
        # particular you might want to find the intermediate
        # nodes/activations/feature_maps in the graph that will be defined as
        # those nodes which are not parameters (not placeholder node types) but
        # are created during the forward pass and are also used in the backward
        # pass for computation.

        # The boundary between the forward pass and backward pass can be
        # identified by locating the node '%sep : [num_users=1] =
        # call_function[target=torch.ops.separator.sep.default]' which will
        # define the end of the forward pass. You will see the loss function
        # after thsi operation and then you will encounter a node named,
        # '%sep_backward : [num_users=1] =
        # call_function[target=torch.ops.separator.sep_backward.default]'. This
        # node marks the beginning of the backward pass.

        # For these intermediate nodes in the graph, you will record their last
        # use in the forward pass and their first use in the backward pass.

        # The parameters of the models are the placeholder (input) nodes of the
        # graph. Note that not all the placeholder nodes of the graph are
        # parameters. The optimizer's states and the input mini-batch are also
        # placeholder nodes that given as inputs to the graph.

        # The parameters and gradients of the model can be otained using the
        # optimizer node's arguments. The optimizer node can be identified by
        # the node '%_fused_adam : [num_users=3] =
        # call_function[target=torch.ops.aten._fused_adam.default]'.
        # The argument at position 0 is the list of parameter nodes, while the
        # argument at position 1 is the list of gradient nodes.

        # Printing the input nodes, node users and node names.

        for node in self.module.graph.nodes:
            print("Node name: ", node.name)
            print("Node type: ", node.op)
            print("Node target: ", node.target)
            print("Input to this node", node.all_input_nodes)
            print("Users of this node: ", node.users)

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True
    ) -> Any:
        
        self.num_runs += 1
        return super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

    def run_node(self, n: fx.Node) -> Any:
        # If you are in the backward pass region and one of the feature maps 'x'
        # was swapped out, and if node 'n' will use this feature map 'x' as one
        # of its inputs then you swap 'x' back to the GPU memory here.

        # TODO ask why start/end here and not including swap-node
        # Measuring time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = super().run_node(n)
        end.record()
        torch.cuda.current_stream().synchronize()
        self.total_time_elapsed += start.elapsed_time(end)
        # TODO ask about memory? Do we want to do it per step or not

        # If you are in the forward pass region and if the current node 'n' is
        # the last user of a feature map 'x', then it should be swapped out to
        # the CPU memory here.

        return result

    def aggregate_stats(self) -> None:
        
        self.all_aggregated_stats = ProfilerStatistics(
            time_per_run=(self.total_time_elapsed / self.num_runs)
        )

        # might need more elaborate agregation for nodes once we've actually profiled them

    def print_stats(self) -> None:
        # can make some appropriate data structure and flush
        if self.all_aggregated_stats is None:
            print("No statistics available yet. Try aggregating.")
        else:
            print(self.all_aggregated_stats)

    def reset_stats(self) -> None:
        # The statistics must be cleared out after x warm-up iterations and
        # reset before the actual measurement begins.
        self.num_runs = 0
        self.total_time_elapsed = 0
        self.all_aggregated_stats = None
