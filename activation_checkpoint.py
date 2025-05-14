import torch
import torch.nn as nn
import torch.fx as fx
from typing import Dict, List, Tuple, Set
from torch.fx.experimental.proxy_tensor import make_fx
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from graph_tracer import SEPFunction
from graph_prof import GraphProfiler, NodeAttributes
from recomp import RecompDecision, RecompAttributes


# We define a custom function that takes in two weight matrices that require
# gradients to be computed and an input data matrix. The function returns the
# gradients of the weight matrices with respect to the loss (sum in our
# example). NOTE: The custom function mimics a simple two layer liner neural
# network with relu activation functions and a sum loss function.
def custom_fn(w1: torch.Tensor, w2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    z = torch.mm(w1, x)
    z = nn.functional.relu(z)
    z = torch.mm(z, w2)
    z = nn.functional.relu(z)
    z = z.sum()
    z = SEPFunction.apply(z)
    z.backward()
    return w1.grad, w2.grad


def replace_subsequent_uses_of(
    graph: fx.Graph, old_node: fx.Node, new_node: fx.Node
) -> None:
    old_node_users = old_node.users
    for node in reversed(graph.nodes):
        if node == new_node:
            break
        if node in old_node_users:
            node.replace_input_with(old_node, new_node)


def remove_detach_nodes(gm: fx.GraphModule) -> fx.GraphModule:
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.detach.default:
            input_node = node.all_input_nodes[0]
            node.replace_all_uses_with(input_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()
    return gm


def get_name_to_node_map(gm: fx.GraphModule) -> Dict[str, fx.Node]:
    name_to_node = {}
    for node in gm.graph.nodes:
        name_to_node[node.name] = node
    return name_to_node

# def _determine_nodes_to_recompute(gm: fx.GraphModule, name_to_node: Dict[str, fx.Node], all_nodes_info: Dict[fx.Node, NodeAttributes]
#                                   ) -> Tuple[List[fx.Node], List[List[fx.Node]], List[fx.Node]]:
#     # returns nodes_to_recompute, nodes_required_to_recompute, first_back_access
#     #  where all lists have the same length

#     # In this example we are going to recompute one of the relu activations for the
#     # backward pass instead of saving it. We know from our custom function
#     # that we have 2 intermeidate nodes: ['relu', 'relu_1']

#     # So the intermediate node to recompute is: ['relu'] and
#     # intermediate nodes to checkpoint (retain) are: ['relu_1']

#     # Nodes required to recompute 'relu' are ['w1_1', 'x_1']
#     # First back use is at node 't'

#     # NOTE: For your project, you will use GraphProfiler to identify the
#     # intermediate nodes, their first back access, last forward access and
#     # then MuTWO's algorithm to select the intermediate 'nodes_to_recompute' and
#     # checkpoint (retain). The 'nodes_required_to_recompute' any of the
#     # intermediate nodes MUST be a subset of the placeholder nodes and the
#     # intermediate nodes that are checkpointed.

#     # NOTE: we cannot directly use 'mm' to recompute 'relu' since 'mm' is not an
#     # intermediate node that is retained (checkpointed).

#     nodes_to_recompute = [name_to_node["relu"]]
#     nodes_required_to_recompute_nodes = [[name_to_node["w1_1"], name_to_node["x_1"]]]
#     first_back_accesses = [name_to_node["t"]]
#     return nodes_to_recompute, nodes_required_to_recompute_nodes, first_back_accesses

# def topo_sort(nodes):
#     seen = set()
#     sorted_nodes = []

#     def visit(n):
#         if n in seen:
#             return
#         seen.add(n)
#         for arg in n.all_input_nodes:
#             visit(arg)
#         sorted_nodes.append(n)

#     for node in nodes:
#         visit(node)

#     return sorted_nodes


def activation_checkpointing(gm: fx.GraphModule, profile_node_info: Dict[fx.Node, NodeAttributes], recomp_decision: RecompDecision, verbose: bool = False) -> fx.GraphModule:
    # NOTE: You need to create the function for your project and call it inside
    # the graph_transformation function after performing graph profiling.

    name_to_node = get_name_to_node_map(gm)

    for rp, rp_attrs in recomp_decision.recomp_nodes.items():
        # Obtain a sub-graph that recomputes the required nodes
        recompute_subgraph = _extract_graph_with_inputs_outputs(
            joint_graph=gm.graph,
            inputs=list(rp_attrs.recomp_srcs),
            outputs=[rp],
        )
        if verbose:
            print("Extracted recomputation sub-graph: ")
            recompute_subgraph.print_tabular()

        # Insert the nodes of the new sub-graph in the old graph before the first
        # backward access of the node to be recomputed.
        with gm.graph.inserting_before(profile_node_info[rp].first_bw_access):
            for n in recompute_subgraph.nodes:
                if n.op == "placeholder" or n.op == "output":
                    continue

                # Copy the nodes of the new sub-graph to old graph and transform its
                # inputs to match the old-graph inputs. The arg_transform function
                # will pass the input arguments of the new node and will expect a
                # mapping to the nodes of the old graph.
                new_node = gm.graph.node_copy(
                    n, arg_transform=lambda arg: name_to_node[arg.name]
                )

                if n.name in [rp.name]:
                    old_node = name_to_node[n.name]
                    # Replace all the uses of the old node with new recomputation node
                    replace_subsequent_uses_of(
                        gm.graph, old_node=old_node, new_node=new_node
                    )
                # Add the new node to our name to node mapping
                name_to_node[n.name] = new_node

    gm.graph.lint()
    gm.recompile()
    return gm


if __name__ == "__main__":
    # Create two weight matrices that require gradients and one input data matrix
    w1 = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    w2 = torch.randn(2048, 512, device="cuda", requires_grad=True)
    x = torch.randn(1024, 2048, device="cuda")

    # Create a graph module by tracing the the custom function with the given inputs
    graph_module = make_fx(custom_fn)(w1, w2, x)
    graph_module = remove_detach_nodes(graph_module)
    print("Original graph of custom fn (fwd+bwd): ")
    graph_module.graph.print_tabular()

    # Obtain the gradients of (w1, w2) using x as input to the traced function
    # NOTE: We have already captured the backward operations during tracing
    # hence we are executing in no grad mode
    with torch.no_grad():
        old_grads = graph_module(w1, w2, x)

    # Obtain the graph profiling information
    graph_profiler = GraphProfiler(graph_module)
    # with torch.no_grad():
    #     for _ in range(2):
    #         graph_profiler.run()
    #     graph_profiler.reset_stats()

    #     for _ in range(1):
    #         graph_profiler.run()
    #     graph_profiler.aggregate_stats()
    # TODO this does not work we have some placeholder error in the profiler

    # Calculate recomp decision
    peak_mem = graph_profiler.memory_stats.peak_memory_usage
    recomp = RecompDecision(graph_module, graph_profiler.all_nodes_info)
    recomp.determine_recomp_nodes(peak_mem, 0.9*peak_mem)

    # Apply the activation checkpointing algorithm (check new node 'relu_2')
    new_graph_module = activation_checkpointing(graph_module, graph_profiler.all_nodes_info, recomp)
    print("Modified graph of custom fn (fwd+bwd+activation_checkpointing): ")
    new_graph_module.graph.print_tabular()

    # Obtain the gradients of (w1, w2) using x as input to the activation
    # checkpointed function to recalculate them
    with torch.no_grad():
        new_grads = new_graph_module(w1, w2, x)

    # Verify that gradients produced with activation checkpointing equal the
    # ones obtained earlier with no optimization.
    print("Result verification")
    for old_grad, new_grad in zip(old_grads, new_grads):
        print(torch.allclose(old_grad, new_grad))
