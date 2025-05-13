"""
In the initial phase, our primary task is to construct a comprehensive computational graph. This
graph will encapsulate all operations from the forward, backward, and optimizer steps within a
single iteration. The nodes within this graph symbolize individual operations, while the edges
represent the dependencies between input and output data. The profiler's job is :
    1. Collecting data on the computation time and memory usage of each operator when the
    graph operations are executed in topological order.
    2. Categorizing the inputs and outputs of each operation as a parameter, gradient,
    activation, optimizer state, or other types.
    3. Conducting static data analysis on activations by documenting the first and last use of
    each activation during the forward and backward passes.
    4. Generating a peak memory breakdown graph using the collected statistics.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Tuple, Any, final
import torch
import torch.fx as fx
import json

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

@dataclass 
class MemoryStatistics:
    cpu_memory_usages: List[int] = field(default_factory=list)
    gpu_memory_usages: List[int] = field(default_factory=list)
    peak_memory_usage: int = 0
    param_memory_usages: List[int] = field(default_factory=list)
    activation_memory_usages: List[int] = field(default_factory=list)
    grad_memory_usages: List[int] = field(default_factory=list)
    other_memory_usages: List[int] = field(default_factory=list)


@dataclass
class NodeAttributes:
    node_type: NodeType = None
    rank: int = None
    gtype_is_forward: bool = None
    run_time: float = 0.
    swap_time: float = 0.
    peak_mem: int = 0
    active_mem: int = 0
    inactive_time: float = 0
    last_fw_access: fx.Node = None
    first_bw_access: fx.Node = None
    last_bw_access: fx.Node = None


def calculate_memory_usage(tensor: torch.Tensor) -> int:
    """
    Calculate the memory usage of a tensor in bytes.
    """
    if tensor is None:
        return 0
    return tensor.element_size() * tensor.nelement()

class GraphProfiler(fx.Interpreter):
    def __init__(self, module: fx.GraphModule, to_swap: bool = False, garbage_collect_values: bool = True, verbose=False):
        super().__init__(module, garbage_collect_values)
        self.verbose = verbose
        
        # Fields for statistics
        self.num_runs: int = 0
        self.total_time_elapsed = 0     # in ms
        self.memory_stats = MemoryStatistics()

        self.in_forward_pass = True
        self.rank_of_backward = None
        self.all_aggregated_stats: ProfilerStatistics = None
        self.all_nodes_info: Dict[fx.Node, NodeAttributes] = {}

        rank = 0
        in_forward = True

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

        # First pass: for forward/backward uses, for later use in classification
        for node in self.module.graph.nodes:
            rank += 1 # idk what rank we want here / where it's relevant

            if self._is_backward_start(node):
                # beginning of backward pass
                in_forward = False
                self.rank_of_backward = rank

            if in_forward:
                # add self
                if node not in self.all_nodes_info:
                    self.all_nodes_info[node] = NodeAttributes(
                        node_type=None, 
                        rank=rank, 
                        gtype_is_forward=True,
                        last_fw_access = None,
                        first_bw_access = None,
                        last_bw_access = None
                    )
                else:
                    print("Unexpected error: node already in all_nodes_info", node.name, flush=True)
                    self.all_nodes_info[node].rank = rank
                
                # update other nodes info
                for input_node in node.all_input_nodes:
                    self.all_nodes_info[input_node].last_fw_access = node
            else:
                # add self
                if node not in self.all_nodes_info:
                    self.all_nodes_info[node] = NodeAttributes(
                        node_type=None, 
                        rank=rank, 
                        gtype_is_forward=False,
                        last_fw_access = None,
                        first_bw_access = None,
                        last_bw_access = None
                    )
                else:
                    # don't update anything else but warn
                    print("Unexpected error: node already in all_nodes_info during backwards pass", node.name, flush=True)
                
                # update other nodes' info for usage
                for input_node in node.all_input_nodes:
                    if self.all_nodes_info[input_node].first_bw_access is None:
                        self.all_nodes_info[input_node].first_bw_access = node
                    self.all_nodes_info[input_node].last_bw_access = node

        # Second pass to correctly classify node types
        found_optimizer = False
        param_nodes_to_update = []
        grad_nodes_to_update = []
        for node in self.module.graph.nodes:
            if node not in self.all_nodes_info:
                print("Unexpected error: node not in all_nodes_info", node.name, flush=True)
                continue
            else:
                self.all_nodes_info[node].node_type = self._initial_categorize_node(node)

            if '_fused_adam' in node.name and node.target == torch.ops.aten._fused_adam.default:
                found_optimizer = True
                print("Found optimizer node during initialization", flush=True)
                param_nodes_to_update = node.args[0]
                grad_nodes_to_update = node.args[1]

        # final classification should be with optimizer node information
        if not found_optimizer:
            print("Did not find optimizer node during initialization", flush=True)
        for input_node in param_nodes_to_update:
            self.all_nodes_info[input_node].node_type = NodeType.PARAM
        for output_node in grad_nodes_to_update:
            self.all_nodes_info[output_node].node_type = NodeType.GRAD

        if self.verbose:
            # print out everything needed
            for node in self.all_nodes_info:
                # print(f"Node: ", node.name, self.all_nodes_info[node], flush=True)
                if self.all_nodes_info[node].node_type == NodeType.ACT:
                    print(f"Activation node: ", node.name, self.all_nodes_info[node], flush=True)
                if self.all_nodes_info[node].node_type == NodeType.OTHER:
                    print(f"Other node: ", node.name, self.all_nodes_info[node], flush=True)

    def _initial_categorize_node(self, node: fx.Node) -> NodeType:
        if node.op == 'placeholder':
            # most likely be a parameter, but can be optimixer state or input mini-batch as well
            return NodeType.PARAM

        node_name = node.name.lower()
        guaranteed_activation_names = []
        other_feature_map_names = ['relu', 'gelu', 'sigm', 'tanh', 'softmax','conv', 'pool', 'aten', 'mm', 'fc', 'lin', 'batchnorm', 'dropout', 'mul', 'sub', 'add', 'view', 'getitem', 'transpos']
        # TODO look through printouts more carefully to ensure this is the best (are there some other rules?)
        if any(guaranteed_activation_name in node_name for guaranteed_activation_name in guaranteed_activation_names):
            return NodeType.ACT
        elif any(feature_map_name in node_name for feature_map_name in other_feature_map_names):
            # this might be a feature map but we must also determine if it has appropriate 'use cases' in the graph
            # either must be created in forward pass and used in backward pass, or an in-place operation
            if (self.all_nodes_info[node].gtype_is_forward) and ((self.all_nodes_info[node].last_fw_access is None and self.all_nodes_info[node].first_bw_access is None) or (self.all_nodes_info[node].first_bw_access is not None)):
                return NodeType.ACT
            else:
                if self.verbose:
                    print("False positive for activation node", node.name, self.all_nodes_info[node], flush=True)
                return NodeType.OTHER
        elif 'tag_grad' in node_name:
            return NodeType.GRAD
        else:
            return NodeType.OTHER

    def _is_backward_start(self, node: fx.Node) -> bool:
        node_name = node.name.lower()
        return 'sep_backward' in node_name

    def run(
        self,
        *args,
        initial_env: Dict[fx.Node, Any] | None = None,
        enable_io_processing: bool = True
    ) -> Any:
        
        self.num_runs += 1
        self.in_forward_pass = True
        return super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

    def run_node(self, n: fx.Node) -> Any:
        if self._is_backward_start(n):
            self.in_forward_pass = False

        # If you are in the backward pass region and one of the feature maps 'x'
        # was swapped out, and if node 'n' will use this feature map 'x' as one
        # of its inputs then you swap 'x' back to the GPU memory here.

        # Measuring time
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = super().run_node(n)
        end.record()
        torch.cuda.current_stream().synchronize()
        node_time = start.elapsed_time(end)
        self.all_nodes_info[n].run_time += node_time
        self.total_time_elapsed += node_time

        # Measure and update memory
        self._update_memory_stats()
        

        # If you are in the forward pass region and if the current node 'n' is
        # the last user of a feature map 'x', then it should be swapped out to
        # the CPU memory here.


        return result

    def _is_on_gpu(self, node: fx.Node | torch.Tensor) -> bool:
        if isinstance(node, fx.Node):
            node = self.env[node]
        return node.device.type != 'cpu'


    def _swap_out(self, node: fx.Node) -> None:
        # not sure if i should be doing swapping AND recomputation or just recomputation
        return None

    def _swap_in(self, node: fx.Node) -> None:
        # not sure if i should be swapping AND recomputation or just recomputation
        return None



    def _update_memory_stats(self) -> None:
        # Choose to not do on a specific node because not scalable to other operations but this is not efficient

        # Cumulative memory stats to update
        cpu_memory_usage = 0
        gpu_memory_usage = 0
        # TODO make sure this is only their GPU memory usage....
        param_memory_usage = 0
        activation_memory_usage = 0
        grad_memory_usage = 0
        other_memory_usage = 0
        # Populate by iterating over each node
        # A quicker approach might call this function for each node, using knowledge about the graph to figure out memory changes
        for node in self.env:
            node_tensor = self.env[node]
            node_category = self.all_nodes_info[node].node_type

            if torch.is_tensor(node_tensor):
                mem_usage = calculate_memory_usage(node_tensor)
                # update node_info
                self.all_nodes_info[node].active_mem = mem_usage
                if mem_usage > self.all_nodes_info[node].peak_mem:
                    self.all_nodes_info[node].peak_mem = mem_usage
                # update memory stats
                if self._is_on_gpu(node_tensor):
                    gpu_memory_usage += mem_usage
                    if node_category == NodeType.PARAM:
                        param_memory_usage += mem_usage
                    elif node_category == NodeType.ACT:
                        activation_memory_usage += mem_usage
                    elif node_category == NodeType.GRAD:
                        grad_memory_usage += mem_usage
                    elif node_category == NodeType.OTHER:
                        other_memory_usage += mem_usage
                else:
                    cpu_memory_usage += mem_usage
        # Update memory stats
        self.memory_stats.peak_memory_usage = max(gpu_memory_usage + cpu_memory_usage, self.memory_stats.peak_memory_usage)
        self.memory_stats.cpu_memory_usages.append(cpu_memory_usage)
        self.memory_stats.gpu_memory_usages.append(gpu_memory_usage)
        self.memory_stats.param_memory_usages.append(param_memory_usage)
        self.memory_stats.activation_memory_usages.append(activation_memory_usage)
        self.memory_stats.grad_memory_usages.append(grad_memory_usage)
        self.memory_stats.other_memory_usages.append(other_memory_usage)


    def aggregate_stats(self) -> None:
        self.all_aggregated_stats = ProfilerStatistics(
            time_per_run=(self.total_time_elapsed / self.num_runs)
        )

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
        self.memory_stats = MemoryStatistics()
        self.in_forward_pass = True

    def dump_stats(self, path: str = None) -> None:
        if path is None:
            path = 'results/default_graph_profiler_stats.json'
        with open(path, 'w') as f:
            dict_to_dump = asdict(self.memory_stats)
            dict_to_dump['rank_of_backward'] = self.rank_of_backward
            dict_to_dump['avg_time_per_run'] = self.all_aggregated_stats.time_per_run
            json.dump(dict_to_dump, f)
        print(f"Dumped graph profiler stats to {path}", flush=True)
        
