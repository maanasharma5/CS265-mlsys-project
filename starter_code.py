import os
import logging
from typing import Any
import torch.multiprocessing as mp
import torch
import torch.nn as nn
from graph_tracer import compile, SEPFunction
from graph_profiler import GraphProfiler
import torch.fx as fx


class DummyModel(nn.Module):
    def __init__(self, layers: int, dim: int):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x):
        return self.mod(x)

class WrappedDummyModel(nn.Module):
    def __init__(self, layers: int, dim: int):
        super().__init__()
        self.mod = DummyModel(layers, dim)
    def forward(self, x):
        return SEPFunction.apply(self.mod(x))

def train_step(
    model: torch.nn.Module, optim: torch.optim.Optimizer, batch: torch.Tensor
):
    out: torch.Tensor = model(batch)
    out.sum().backward()
    optim.step()
    optim.zero_grad()

def graph_transformation(gm:fx.GraphModule, args:Any):

    print(gm.graph)

    graph_profiler = GraphProfiler(gm)
    with torch.no_grad():
        graph_profiler.run(*args)

    return gm


def experiment():
    logging.getLogger().setLevel(logging.DEBUG)
    torch.manual_seed(20)
    batch_size = 100
    layers = 10
    dim = 100
    num_iters = 5
    model = WrappedDummyModel(dim=dim, layers=layers).cuda()
    batch = torch.randn(batch_size, dim).cuda()
    optim = torch.optim.Adam(
        model.parameters(), lr=0.01, foreach=False, fused=True, capturable=True
    )

    compiled_fn = compile(train_step, graph_transformation)
    compiled_fn(model, optim, batch)

if __name__ == "__main__":
    experiment()

