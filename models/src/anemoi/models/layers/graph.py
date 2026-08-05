# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import defaultdict

import einops
import torch
from torch import Tensor
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import NodeStorage


class TrainableTensor(nn.Module):
    """Trainable Tensor Module."""

    def __init__(self, tensor_size: int, trainable_size: int) -> None:
        """Initialize TrainableTensor."""
        super().__init__()

        if trainable_size > 0:
            trainable = nn.Parameter(
                torch.empty(
                    tensor_size,
                    trainable_size,
                ),
            )
            nn.init.constant_(trainable, 0)
        else:
            trainable = None
        self.register_parameter("trainable", trainable)

    def forward(self, x: Tensor, batch_size: int) -> Tensor:
        latent = [einops.repeat(x, "e f -> (repeat e) f", repeat=batch_size)]
        if self.trainable is not None:
            latent.append(einops.repeat(self.trainable.to(x.device), "e f -> (repeat e) f", repeat=batch_size))
        return torch.cat(
            latent,
            dim=-1,  # feature dimension
        )


class NamedNodesAttributes(nn.Module):
    """Named Nodes Attributes information.

    Attributes
    ----------
    num_nodes : dict[str, int]
        Number of nodes for each group of nodes.
    attr_ndims : dict[str, int]
        Total dimension of node attributes (non-trainable + trainable) for each group of nodes.
    trainable_tensors : nn.ModuleDict
        Dictionary of trainable tensors for each group of nodes.

    Methods
    -------
    get_coordinates(self, name: str) -> Tensor
        Get the coordinates of a set of nodes.
    forward( self, name: str, batch_size: int) -> Tensor
        Get the node attributes to be passed trough the graph neural network.
    """

    num_nodes: dict[str, int]
    attr_ndims: dict[str, int]
    trainable_tensors: dict[str, TrainableTensor]

    def __init__(
        self,
        trainable_parameters: dict[str, int],
        graph_data: HeteroData,
        node_attributes: dict[str, list[str]] | None = None,
    ) -> None:
        """Initialize NamedNodesAttributes."""
        super().__init__()

        trainable_parameters = defaultdict(int, trainable_parameters)
        # Names of the graph node attributes to expose to the model, per group of nodes. Empty by
        # default, which reproduces the coordinates-and-trainable-params-only behaviour exactly.
        self.node_attribute_names = defaultdict(list, node_attributes or {})

        self.define_fixed_attributes(graph_data, trainable_parameters)

        self.trainable_tensors = nn.ModuleDict()
        for nodes_name, nodes in graph_data.node_items():
            self.register_coordinates(nodes_name, nodes.x)
            self.register_features(nodes_name, nodes)
            self.register_tensor(nodes_name, trainable_parameters[nodes_name])

    @staticmethod
    def get_attribute_width(nodes: NodeStorage, attr_name: str) -> int:
        """Number of feature columns a graph node attribute contributes."""
        assert attr_name in nodes, (
            f"Node attribute '{attr_name}' was requested but is not present on these nodes. "
            f"Available: {sorted(k for k in nodes.keys() if not k.startswith('_'))}"
        )
        values = nodes[attr_name]
        return 1 if values.ndim == 1 else values.shape[-1]

    def define_fixed_attributes(self, graph_data: HeteroData, trainable_parameters: dict[str, int]) -> None:
        """Define fixed attributes."""
        nodes_names = list(graph_data.node_types)
        self.num_nodes = {nodes_name: graph_data[nodes_name].num_nodes for nodes_name in nodes_names}
        self.attr_ndims = {
            nodes_name: 2 * graph_data[nodes_name].x.shape[1]
            + trainable_parameters[nodes_name]
            + sum(
                self.get_attribute_width(graph_data[nodes_name], attr_name)
                for attr_name in self.node_attribute_names[nodes_name]
            )
            for nodes_name in nodes_names
        }

    def register_coordinates(self, name: str, node_coords: Tensor) -> None:
        """Register coordinates."""
        sin_cos_coords = torch.cat([torch.sin(node_coords), torch.cos(node_coords)], dim=-1)
        self.register_buffer(f"latlons_{name}", sin_cos_coords, persistent=True)

    def get_coordinates(self, name: str) -> Tensor:
        """Return original coordinates."""
        sin_cos_coords = getattr(self, f"latlons_{name}")
        ndim = sin_cos_coords.shape[1] // 2
        sin_values = sin_cos_coords[:, :ndim]
        cos_values = sin_cos_coords[:, ndim:]
        return torch.atan2(sin_values, cos_values)

    def register_features(self, name: str, nodes: NodeStorage) -> None:
        """Register the graph node attributes selected for this group of nodes.

        Nothing is registered when no attribute is requested, so that the feature vector is
        byte-for-byte what it was before this was added.

        Normalisation is the graph's responsibility, via the ``norm`` argument of the node
        attribute builders: raw values such as elevation in metres would otherwise dominate the
        sin/cos coordinates, which lie in [-1, 1].
        """
        attr_names = self.node_attribute_names[name]
        if not attr_names:
            return

        columns = []
        for attr_name in attr_names:
            values = nodes[attr_name].float()
            columns.append(values.unsqueeze(-1) if values.ndim == 1 else values)

        self.register_buffer(f"features_{name}", torch.cat(columns, dim=-1), persistent=True)

    def register_tensor(self, name: str, num_trainable_params: int) -> None:
        """Register a trainable tensor."""
        self.trainable_tensors[name] = TrainableTensor(self.num_nodes[name], num_trainable_params)

    def forward(self, name: str, batch_size: int) -> Tensor:
        """Returns the node attributes to be passed trough the graph neural network.

        It includes the coordinates, the trainable parameters and any selected node attributes,
        concatenated in that order. Appending the attributes last keeps ``[latlons, trainable]``
        as an unchanged prefix, so migrating a checkpoint across this change is a matter of
        padding weight columns at the end rather than reordering them.
        """
        latlons = getattr(self, f"latlons_{name}")
        latent = self.trainable_tensors[name](latlons, batch_size)

        features = getattr(self, f"features_{name}", None)
        if features is None:
            return latent

        return torch.cat(
            [latent, einops.repeat(features, "e f -> (repeat e) f", repeat=batch_size)],
            dim=-1,  # feature dimension
        )
