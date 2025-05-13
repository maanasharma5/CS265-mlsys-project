import torch.fx as fx
from typing import Any, Dict, List, Tuple, Set
from graph_prof import GraphProfiler, NodeAttributes, NodeType, OP
from dataclasses import dataclass, field, asdict
from time import time

# Recomputation Decision without Swapping-optimized Scheduling -- as described in https://www.youtube.com/watch?v=GBzar0GQrJo
# Largely from Appendix A.4 of MuTWO paper

"""
As a reminder:

class NodeType(Enum):

    PARAM = 0
    ACT = 1
    GRAD = 2
    OTHER = 3

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
"""

@dataclass 
class RecompAttributes:
    memory_usage: int = 0
    recomp_srcs: Set[fx.Node] = field(default_factory=set)
    recomp_cnt: int = 0
    recomp_time: float = 0.0



class RecompDecision:
    def __init__(self, gm: fx.GraphModule, profile_node_info: Dict[fx.Node, NodeAttributes]):
        print("Initializing RecompDecision", flush=True)
        self.gm = gm
        self.profile_node_info = profile_node_info

        self._src_cache: Dict[fx.Node, Set[fx.Node]] = {}

        self.recomp_nodes: Dict[fx.Node, RecompAttributes] = {}

        self.candidate_nodes: Dict[fx.Node, RecompAttributes] = {}
        intermediate_nodes = [node for node in gm.graph.nodes if self.profile_node_info[node].node_type == NodeType.ACT] # initialize with all activation (intermediate) nodes
        print(f"Found {len(intermediate_nodes)} intermediate nodes", flush=True)
        for node in intermediate_nodes:
            start = time()
            attrs = RecompAttributes()
            attrs.memory_usage = self.profile_node_info[node].peak_mem
            attrs.recomp_srcs = self._find_srcs(node)
            attrs.recomp_time = sum(self.profile_node_info[src].run_time for src in attrs.recomp_srcs) + self.profile_node_info[node].run_time
            attrs.recomp_cnt = 0
            self.candidate_nodes[node] = attrs
            end = time()
            print(f"Found {len(attrs.recomp_srcs)} srcs for node {node.name} in {end - start:.4f} seconds", flush=True)

        print(f"Finished initializing RecompDecision with {len(self.candidate_nodes)} candidate nodes", flush=True)

    def _find_srcs(self, node: fx.Node) -> Set[fx.Node]:
        if node in self._src_cache:
            return self._src_cache[node]

        srcs: Set[fx.Node] = set()
        for src in node.all_input_nodes:
            if src.op in [OP.PLACEHOLDER]:
                srcs.add(src)
            # if src is a parameter, we don't need to recompute it
            if src in self.candidate_nodes.keys():
                srcs.add(src)
            else:
                srcs.update(self._find_srcs(src))

        self._src_cache[node] = srcs
        return srcs

    # def _find_srcs(self, node: fx.Node) -> Set[fx.Node]:
    #     """
    #     DFS with memoisation.  `retain` = activations we already agreed to keep.
    #     """

    #     LEAF_OPS = {"placeholder", "get_attr"}

    #     if node in self._src_cache:
    #         return self._src_cache[node]

    #     frontier: Set[fx.Node] = set()
    #     for inp in node.all_input_nodes:
    #         if (inp.op in LEAF_OPS) or (inp in self.candidate_nodes.keys()):     # the *only* stopping rule
    #             frontier.add(inp)
    #         else:
    #             frontier.update(self._find_srcs(inp))

    #     self._src_cache[node] = frontier
    #     return frontier




    def _choose_max_recomp_ratio(self) -> fx.Node:
        max_recomp_ratio = -1
        max_node = None
        for node, attrs in self.candidate_nodes.items():
            # choosing to compute each time for correctness in case of lagging updates
            total_recomp_time = attrs.recomp_time * attrs.recomp_cnt
            recomp_ratio = attrs.memory_usage / (total_recomp_time) if total_recomp_time > 0 else 0
            if recomp_ratio > max_recomp_ratio:
                max_recomp_ratio = recomp_ratio
                max_node = node
        return max_node

    def _update_recomps_after_choice(self, cand: fx.Node) -> int:
        # NOTE assumes that cand information is still in candidate_nodes
        # Algorithm E
        cand_recomp_cnt = 1
        for recomped_node, recomped_node_attrs in self.recomp_nodes.items():
            cand_node_attrs = self.candidate_nodes[cand]
            if cand in recomped_node_attrs.recomp_srcs:
                recomped_node_attrs.recomp_srcs.remove(cand)
                recomped_node_attrs.recomp_srcs.update(cand_node_attrs.recomp_srcs)
                recomped_node_attrs.recomp_time += cand_node_attrs.recomp_time
                cand_recomp_cnt += 1
        return cand_recomp_cnt

    def _update_candidates_after_choice(self, t: fx.Node) -> None:
        # Algorithm F
        # slightly different than stated because i am not maintaing total_recomp_time separately

        # NOTE assumes that t is in recomp_nodes already -- if not we need to change line 110 as well

        t_attrs = self.recomp_nodes[t]

        for cand_node, cand_attrs in self.candidate_nodes.items():
            # case a
            if t in cand_attrs.recomp_srcs:
                cand_attrs.recomp_srcs.remove(t)
                cand_attrs.recomp_srcs.update(t_attrs.recomp_srcs)
                # accumulate cand_recomp_time (different than stated algo)
                cand_attrs.recomp_time += t_attrs.recomp_time
                cand_attrs.recomp_cnt = sum(1 for recomped_node_attrs in self.recomp_nodes.values() if cand_node in recomped_node_attrs.recomp_srcs)
    
            # case b
            if cand_node in t_attrs.recomp_srcs:
                # different than stated in algo
                cand_attrs.recomp_cnt += t_attrs.recomp_cnt

            # always need to update recompute ratio, but omitted because we don't keep that info separately

    def determine_recomp_nodes(self, consumed_memory: int, max_memory_capacity: int) -> None:
        # Updates internal state of RecompDecision object so we should not call this method with contradictory arguments
        # probably should have chosen a different abstraction to this process than a class
        # get all the information from self.recomp_nodes

        while (len(self.candidate_nodes) > 0 and consumed_memory > max_memory_capacity):
            # Algorithm D
            cand = self._choose_max_recomp_ratio()
            if cand is None:
                break
            # keep in candidate_nodes and get appropriate information
            cand_recomp_cnt = self._update_recomps_after_choice(cand)

            # update information and remove from candidates
            cand_attrs = self.candidate_nodes.pop(cand)
            cand_attrs.recomp_cnt = cand_recomp_cnt

            # add to recompute nodes
            self.recomp_nodes[cand] = cand_attrs
            self._update_candidates_after_choice(cand)
            consumed_memory -= self.recomp_nodes[cand].memory_usage

            print(f"Choosing to recompute node {cand.name}", flush=True)


                
  

    




    
