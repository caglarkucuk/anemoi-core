# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import torch
from torch_geometric.data.storage import NodeStorage

from anemoi.graphs.nodes.attributes.base_attributes import BaseNodeAttribute
from anemoi.graphs.nodes.attributes.base_attributes import BooleanBaseNodeAttribute

LOGGER = logging.getLogger(__name__)

MESH_LEVEL_KEY = "_node_levels"


def _get_node_levels(nodes: NodeStorage, cls_name: str) -> torch.Tensor:
    """Read the refinement level recorded by an adaptive mesh node builder."""
    assert MESH_LEVEL_KEY in nodes, (
        f"{cls_name} requires nodes built by AdaptiveTriNodes; "
        f"'{MESH_LEVEL_KEY}' not found in nodes of type '{nodes.get('node_type')}'."
    )
    return nodes[MESH_LEVEL_KEY].squeeze()


class MeshLevel(BaseNodeAttribute):
    """Refinement level of each node of an adaptive icosahedral mesh.

    The level is set by the node builder but stored privately, so it would be dropped by
    ``GraphCreator.clean()``. This attribute republishes it, both for inspection and so
    that MeshLevelMask can key off it.

    Methods
    -------
    get_raw_values(self, nodes)
        Return the refinement level of each node.
    """

    def __init__(self, norm: str | None = None, dtype: str = "int16") -> None:
        super().__init__(norm, dtype)

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        return _get_node_levels(nodes, self.__class__.__name__)


class MeshLevelMask(BooleanBaseNodeAttribute):
    """Boolean mask selecting the nodes of an adaptive mesh at a given refinement level.

    Intended to drive per-level encoder and decoder edge builders: a fixed number of nearest
    neighbours is only meaningful when the hidden node density is uniform relative to the data,
    so a mesh with several resolutions needs one edge builder per level, each with its own
    number of neighbours or cut-off radius.

    Attributes
    ----------
    level : int | list[int]
        Refinement level(s) to select.

    Methods
    -------
    get_raw_values(self, nodes)
        Return the mask of nodes at the requested level(s).
    """

    def __init__(self, level: int | list[int]) -> None:
        super().__init__()
        self.levels = [level] if isinstance(level, int) else list(level)
        assert len(self.levels) > 0, f"{self.__class__.__name__} requires at least one level."

    def get_raw_values(self, nodes: NodeStorage, **kwargs) -> torch.Tensor:
        node_levels = _get_node_levels(nodes, self.__class__.__name__)
        mask = torch.isin(node_levels, torch.tensor(self.levels, device=node_levels.device))
        if not mask.any():
            LOGGER.warning("%s selected no nodes for level(s) %s.", self.__class__.__name__, self.levels)
        return mask
