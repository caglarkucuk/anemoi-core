# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Iterable

import networkx as nx
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.generate.masks import AreaMaskBuilder
from anemoi.graphs.generate.transforms import cartesian_to_latlon_rad
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian_np
from anemoi.graphs.generate.utils import get_coordinates_ordering

LOGGER = logging.getLogger(__name__)


def create_tri_nodes(
    resolution: int, area_mask_builder: AreaMaskBuilder | None = None
) -> tuple[nx.DiGraph, np.ndarray, list[int]]:
    """Creates a global mesh from a refined icosahedron.

    This method relies on the trimesh python library.

    Parameters
    ----------
    resolution : int
        Level of mesh resolution to consider.
    area_mask_builder : AreaMaskBuilder
        AreaMaskBuilder with the cloud of points to limit the mesh area, by default None.

    Returns
    -------
    graph : networkx.Graph
        The specified graph (only nodes) sorted by latitude and longitude.
    coords_rad : np.ndarray
        The node coordinates (not ordered) in radians.
    node_ordering : list[int]
        Order of the node coordinates to be sorted by latitude and longitude.
    """
    coords_rad = get_latlon_coords_icosphere(resolution)

    node_ordering = get_coordinates_ordering(coords_rad)

    if area_mask_builder is not None:
        area_mask = area_mask_builder.get_mask(coords_rad)
        area_mask = area_mask.cpu().numpy()
        node_ordering = node_ordering[area_mask[node_ordering]]

    # Creates the graph, with the nodes sorted by latitude and longitude.
    nx_graph = create_nx_graph_from_tri_coords(coords_rad, node_ordering)

    return nx_graph, coords_rad, list(node_ordering)


def create_stretched_tri_nodes(
    base_resolution: int,
    lam_resolution: int,
    area_mask_builder: AreaMaskBuilder | None = None,
) -> tuple[nx.DiGraph, np.ndarray, list[int]]:
    """Creates a global mesh with 2 levels of resolution.

    The base resolution is used to define the nodes outside the Area Of Interest (AOI),
    while the lam_resolution is used to define the nodes inside the AOI.

    Parameters
    ---------
    base_resolution : int
        Global resolution level.
    lam_resolution : int
        Local resolution level.
    area_mask_builder : AreaMaskBuilder
        Builder used to generate the Area Of Interest (AOI) mask that limits the mesh area.

    Returns
    -------
    nx_graph : nx.DiGraph
        The graph with the added nodes.
    coords_rad : np.ndarray
        The node coordinates (not ordered) in radians.
    node_ordering : list[int]
        Order of the node coordinates to be sorted by latitude and longitude.
    """
    assert area_mask_builder is not None, "AOI mask builder must be provided to build refined grid."
    # Get the low resolution nodes
    base_coords_rad = get_latlon_coords_icosphere(base_resolution)

    # Define the low resolution outside AOI mask
    base_area_mask = ~area_mask_builder.get_mask(base_coords_rad)
    base_area_mask = base_area_mask.cpu().numpy()

    # Get the high resolution nodes
    coords_rad = get_latlon_coords_icosphere(lam_resolution)

    # Get the node ordering for all high resolution nodes
    node_ordering = get_coordinates_ordering(coords_rad)

    # Define the high resolution inside AOI mask
    lam_area_mask = area_mask_builder.get_mask(coords_rad)
    lam_area_mask = lam_area_mask.cpu().numpy()

    # Pad the low resolution ~(AOI mask) to match the high resolution AOI mask
    base_area_mask_padded = np.pad(base_area_mask, (0, len(lam_area_mask) - len(base_area_mask)), mode="constant")

    # Define the final mask (low resolution outside AOI | high resolution inside AOI )
    area_mask = np.logical_or(base_area_mask_padded, lam_area_mask)

    # Redefine the node ordering to include final node selection
    node_ordering = node_ordering[area_mask[node_ordering]]

    # Creates the graph, with the nodes sorted by latitude and longitude.
    nx_graph = create_nx_graph_from_tri_coords(coords_rad, node_ordering)

    return nx_graph, coords_rad, list(node_ordering)


def get_num_vertices(resolution: int) -> int:
    """Number of vertices of an icosphere at the given refinement level."""
    return 10 * 4**resolution + 2


def get_birth_levels(num_vertices: int) -> np.ndarray:
    """Refinement level at which each vertex of an icosphere first appears.

    ``trimesh.creation.icosphere`` produces strictly nested vertex arrays: the vertices of
    the level-r icosphere are the first ``10 * 4**r + 2`` vertices of the level-(r+1) one,
    in the same order. A vertex index therefore determines the coarsest mesh containing it,
    without any geometry being involved.

    Parameters
    ----------
    num_vertices : int
        Number of vertices of the finest icosphere considered.

    Returns
    -------
    np.ndarray of shape (num_vertices, )
        The refinement level at which each vertex is introduced.
    """
    # 20 levels is far beyond anything that fits in memory, so this never truncates.
    thresholds = np.array([get_num_vertices(r) for r in range(20)])
    return np.searchsorted(thresholds, np.arange(num_vertices), side="right").astype(np.int16)


def get_mean_spacing_km(resolution: int) -> float:
    """Mean distance between neighbouring vertices of an icosphere, in kilometres."""
    return float(np.sqrt(4 * np.pi * EARTH_RADIUS**2 / get_num_vertices(resolution)))


def enforce_level_gradation(
    target_levels: np.ndarray,
    coords_rad: np.ndarray,
    min_level: int,
    max_level: int,
    buffer: int = 1,
) -> np.ndarray:
    """Grow a collar of intermediate resolution around every refined region.

    A vertex introduced at level r is the midpoint of a level-(r-1) edge, so it has exactly
    two parents at level r-1 or coarser. If the target level drops by more than one between
    a vertex and its parents, those parents are discarded and the vertex is left hanging off
    the mesh with a degree of 2. Making the target level field 1-Lipschitz with respect to
    mesh adjacency rules that out: every kept vertex is guaranteed to keep both parents.

    Parameters
    ----------
    target_levels : np.ndarray of shape (num_vertices, )
        Requested refinement level for each vertex.
    coords_rad : np.ndarray of shape (num_vertices, 2)
        Vertex coordinates, in radians.
    min_level : int
        Coarsest level of the mesh. Levels are never lowered below this.
    max_level : int
        Finest level of the mesh.
    buffer : int, optional
        Width of each collar, in cells of the coarser level. 1 is the minimum that guarantees
        well-formedness; larger values soften the transition, by default 1.

    Returns
    -------
    np.ndarray of shape (num_vertices, )
        Target levels satisfying the gradation constraint.
    """
    assert buffer >= 1, "A buffer of at least 1 cell is required to keep the mesh well-formed."

    levels = np.clip(target_levels, min_level, max_level).astype(np.int16)
    vertices_xyz = latlon_rad_to_cartesian_np(coords_rad)

    # Walk from the finest level down, dilating each refined region by one collar at a time.
    for level in range(max_level, min_level, -1):
        refined = np.where(levels >= level)[0]
        if len(refined) == 0:
            continue

        # Chord length equivalent to `buffer` cells of the next coarser mesh.
        radius_km = buffer * get_mean_spacing_km(level - 1)
        chord = 2 * np.sin(radius_km / (2 * EARTH_RADIUS))

        # Only vertices in the axis-aligned bounding box of the refined region, grown by the
        # collar width, can be reached. Restricting the tree to those keeps the cost tied to
        # the size of the refined region rather than to the resolution of the whole sphere,
        # which matters because the finest icosphere can hold tens of millions of vertices.
        refined_xyz = vertices_xyz[refined]
        lower, upper = refined_xyz.min(axis=0) - chord, refined_xyz.max(axis=0) + chord
        candidates = np.flatnonzero(np.all((vertices_xyz >= lower) & (vertices_xyz <= upper), axis=1))

        neighbours = cKDTree(vertices_xyz[candidates]).query_ball_point(refined_xyz, r=chord, workers=-1)
        collar = candidates[np.unique(np.concatenate(neighbours))] if len(neighbours) else np.array([], dtype=int)
        levels[collar] = np.maximum(levels[collar], level - 1)
        LOGGER.debug(
            "Gradation at level %d: %d refined vertices grew a collar of %d vertices (%.1f km, %d candidates).",
            level,
            len(refined),
            len(collar),
            radius_km,
            len(candidates),
        )

    return levels


def create_adaptive_tri_nodes(
    base_resolution: int,
    max_resolution: int,
    level_field=None,
    area_mask_builder: AreaMaskBuilder | None = None,
    aoi_resolution: int | None = None,
    gradation_buffer: int = 1,
) -> tuple[nx.DiGraph, np.ndarray, list[int], np.ndarray]:
    """Creates a global mesh whose resolution varies from point to point.

    Generalises :func:`create_stretched_tri_nodes` from two resolutions to a target level
    field over the sphere. Each vertex of the level-``max_resolution`` icosphere is kept if
    it is introduced no later than the level requested at its own location::

        keep(v)  <=>  birth_level(v) <= target_level(v)

    A two-valued target level field reproduces :func:`create_stretched_tri_nodes` exactly.

    Parameters
    ---------
    base_resolution : int
        Coarsest resolution level, applied wherever nothing finer is requested.
    max_resolution : int
        Finest resolution level the target level field may request.
    level_field : BaseLevelField, optional
        Source of additional refinement, layered on top of the AOI floor. A level field only
        ever raises the resolution: the level actually used is the larger of the two, so a
        field may return 0 wherever it has nothing to ask for. When None, the mesh is the
        two-resolution stretched grid of :func:`create_stretched_tri_nodes`.
    area_mask_builder : AreaMaskBuilder, optional
        Builder used to generate the Area Of Interest (AOI) mask.
    aoi_resolution : int, optional
        Resolution floor inside the AOI, by default `max_resolution`. Lowering it leaves room
        for a level field to refine selectively rather than covering the whole AOI.
    gradation_buffer : int, optional
        Width of the transition collars, in cells, by default 1.

    Returns
    -------
    nx_graph : nx.DiGraph
        The graph with the added nodes.
    coords_rad : np.ndarray
        The node coordinates (not ordered) in radians.
    node_ordering : list[int]
        Order of the node coordinates to be sorted by latitude and longitude.
    node_levels : np.ndarray
        Refinement level of each node, in the order given by `node_ordering`.
    """
    assert base_resolution <= max_resolution, "The base resolution cannot exceed the maximum resolution."

    coords_rad = get_latlon_coords_icosphere(max_resolution)
    node_ordering = get_coordinates_ordering(coords_rad)
    birth_levels = get_birth_levels(len(coords_rad))

    aoi_resolution = max_resolution if aoi_resolution is None else aoi_resolution

    target_levels = np.full(len(coords_rad), base_resolution, dtype=np.int16)
    if area_mask_builder is not None:
        area_mask = area_mask_builder.get_mask(coords_rad).cpu().numpy()
        target_levels[area_mask] = aoi_resolution

    if level_field is None:
        # A two-valued field is 1-Lipschitz only in the trivial case, and gradation here would
        # silently change the mesh. Skip it so this path stays a drop-in for the stretched grid.
        assert area_mask_builder is not None, "An AOI mask builder is required when no level field is given."
    else:
        # A level field only ever refines, so that it composes with the AOI floor.
        target_levels = np.maximum(target_levels, level_field.get_levels(coords_rad)).astype(np.int16)
        target_levels = enforce_level_gradation(
            target_levels, coords_rad, base_resolution, max_resolution, gradation_buffer
        )

    keep = birth_levels <= target_levels
    node_ordering = node_ordering[keep[node_ordering]]
    LOGGER.info(
        "Adaptive mesh: kept %d of %d vertices; level histogram %s",
        len(node_ordering),
        len(coords_rad),
        dict(zip(*np.unique(birth_levels[node_ordering], return_counts=True))),
    )

    # Creates the graph, with the nodes sorted by latitude and longitude.
    nx_graph = create_nx_graph_from_tri_coords(coords_rad, node_ordering)

    return nx_graph, coords_rad, list(node_ordering), birth_levels[node_ordering]


def get_latlon_coords_icosphere(resolution: int) -> np.ndarray:
    """Get the latitude and longitude coordinates (in radians) of the icosphere.

    Parameters
    ----------
    resolution : int
        The resolution level of the icosphere.

    Returns
    -------
    np.ndarray
        The latitude and longitude coordinates, in radians, of the icosphere.
    """
    sphere = trimesh.creation.icosphere(subdivisions=resolution, radius=1.0)
    coords_rad = cartesian_to_latlon_rad(sphere.vertices)
    return coords_rad


def create_nx_graph_from_tri_coords(coords_rad: np.ndarray, node_ordering: np.ndarray) -> nx.DiGraph:
    """Creates the networkx graph from the coordinates and the node ordering."""
    graph = nx.DiGraph()
    for i, coords in enumerate(coords_rad[node_ordering]):
        node_id = node_ordering[i]
        graph.add_node(node_id, hcoords_rad=coords)

    assert list(graph.nodes.keys()) == list(node_ordering), "Nodes are not correctly added to the graph."
    assert graph.number_of_nodes() == len(node_ordering), "The number of nodes must be the same."
    return graph


def add_1_hop_edges(
    nodes_coords_rad,
    node_resolutions: list[int],
    edge_resolutions: list[int],
    node_ordering: list[int],
    area_mask_builder: AreaMaskBuilder | None = None,
) -> np.ndarray:
    """Adds edges for x_hops = 1 relying on trimesh only."""

    hop_1_edges = []

    # Loop over the edge_resolutions to get edges at all refinement levels
    for subdivisions in edge_resolutions:
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
        LOGGER.debug("Adding %d unmasked 1-hop edges for resolution %d", sphere.edges.shape[0], subdivisions)
        hop_1_edges.append(sphere.edges)

    # Concatenate all edges from different resolutions and transpose to get shape (2, num_edges)
    multiscale_edges = np.transpose(np.concatenate(hop_1_edges, axis=0), (1, 0))

    # Map the edges to the node ordering
    #  - Calculate the total (unmasked) number of nodes
    unmasked_nodes = cartesian_to_latlon_rad(
        trimesh.creation.icosphere(subdivisions=max(node_resolutions), radius=1.0).vertices
    )
    assert np.all(
        (nodes_coords_rad.cpu().numpy() - unmasked_nodes[node_ordering]) == 0
    ), "Node coordates do not match coordinates used for multi-scale edge building"
    LOGGER.debug("unmasked_nodes shape[0]: %d", unmasked_nodes.shape[0])
    if area_mask_builder is not None:
        # Take care of the edges with start- or end-point outside the mask
        inverse_ordering = np.full(unmasked_nodes.shape[0], -1, dtype=int)
    else:
        inverse_ordering = np.full(unmasked_nodes.shape[0], 0, dtype=int)

    # Update the start- and end indexes according to the node ordering
    inverse_ordering[node_ordering] = np.arange(len(node_ordering))
    updated_edges = inverse_ordering[multiscale_edges]
    valid_edges_mask = np.all(updated_edges >= 0, axis=0)

    # Select only those edges requested by the mask
    multiscale_edges = updated_edges[:, valid_edges_mask]
    LOGGER.debug("multiscale_edges_shape: %s", multiscale_edges.shape)

    return multiscale_edges


def add_edges_to_nx_graph(
    graph: nx.DiGraph,
    resolutions: list[int],
    x_hops: int = 1,
    area_mask_builder: AreaMaskBuilder | None = None,
) -> nx.DiGraph:
    """Adds the edges to the graph.

    This method adds multi-scale connections to the existing graph. The corresponfing nodes or vertices
    are defined by an isophere at the different esolutions (or refinement levels) specified.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to add the edges. It should correspond to the mesh nodes, without edges.
    resolutions : list[int]
        Levels of mesh refinement levels to consider.
    x_hops : int, optional
        Number of hops between 2 nodes to consider them neighbours, by default 1.
    area_mask_builder : AreaMaskBuilder
        NearestNeighbors with the cloud of points to limit the mesh area, by default None.

    Returns
    -------
    graph : nx.DiGraph
        The graph with the added edges.
    """
    assert x_hops > 0, "x_hops == 0, graph would have no edges ..."

    graph_vertices = np.array([graph.nodes[i]["hcoords_rad"] for i in sorted(graph.nodes)])
    tree = BallTree(graph_vertices, metric="haversine")

    # Build the multi-scale connections
    for resolution in resolutions:
        # Define the coordinates of the isophere vertices at specified 'resolution' level
        r_sphere = trimesh.creation.icosphere(subdivisions=resolution, radius=1.0)
        r_vertices_rad = cartesian_to_latlon_rad(r_sphere.vertices)

        # Limit area of mesh points.
        if area_mask_builder is not None:
            area_mask = area_mask_builder.get_mask(r_vertices_rad)
            area_mask = area_mask.cpu().numpy()
            valid_nodes = np.where(area_mask)[0]
        else:
            valid_nodes = None

        node_neighbours = get_neighbours_within_hops(r_sphere, x_hops, valid_nodes=valid_nodes)

        _, vertex_mapping_index = tree.query(r_vertices_rad, k=1)
        neighbour_pairs = create_node_neighbours_list(graph, node_neighbours, vertex_mapping_index)
        LOGGER.debug("Adding %d edges for resolution %d", len(neighbour_pairs), resolution)
        graph.add_edges_from(neighbour_pairs)
    return graph


def get_neighbours_within_hops(
    tri_mesh: trimesh.Trimesh, x_hops: int, valid_nodes: list[int] | None = None
) -> dict[int, set[int]]:
    """Get the neigbour connections in the graph.

    Parameters
    ----------
    tri_mesh : trimesh.Trimesh
        The mesh to consider.
    x_hops : int
        Number of hops between 2 nodes to consider them neighbours.
    valid_nodes : list[int], optional
        List of valid nodes to consider, by default None. It is useful to consider only a subset of the nodes to save
        computation time.

    Returns
    -------
    neighbours : dict[int, set[int]]
        A list with the neighbours for each vertex. The element at position 'i' correspond to the neighbours to the
        i-th vertex of the mesh.
    """
    edges = tri_mesh.edges_unique

    if valid_nodes is not None:
        edges = edges[np.isin(tri_mesh.edges_unique, valid_nodes).all(axis=1)]
    else:
        valid_nodes = list(range(len(tri_mesh.vertices)))
    graph = nx.from_edgelist(edges)

    neighbours = {
        i: set(nx.ego_graph(graph, i, radius=x_hops, center=False) if i in graph else []) for i in valid_nodes
    }

    return neighbours


def add_neigbours_edges(
    graph: nx.Graph,
    node_idx: int,
    neighbour_indices: Iterable[int],
    self_loops: bool = False,
    vertex_mapping_index: np.ndarray | None = None,
) -> nx.Graph:
    """Adds the edges of one node to its neighbours.

    Parameters
    ----------
    graph : nx.Graph
        The graph.
    node_idx : int
        The node considered.
    neighbour_indices : list[int]
        The neighbours of the node.
    self_loops : bool, optional
        Whether is supported to add self-loops, by default False.
    vertex_mapping_index : np.ndarray, optional
        Index to map the vertices from the refined sphere to the original one, by default None.

    Returns
    -------
    nx.Graph
        The graph with the added edges.
    """
    graph_nodes_idx = list(sorted(graph.nodes))
    for neighbour_idx in neighbour_indices:
        if not self_loops and node_idx == neighbour_idx:  # no self-loops
            continue

        if vertex_mapping_index is not None:
            # Use the same method to add edge in all spheres
            node_neighbour = graph_nodes_idx[vertex_mapping_index[neighbour_idx][0]]
            node = graph_nodes_idx[vertex_mapping_index[node_idx][0]]
        else:
            node_neighbour = graph_nodes_idx[neighbour_idx]
            node = graph_nodes_idx[node_idx]

        # add edge to the graph
        if node in graph and node_neighbour in graph:
            graph.add_edge(node_neighbour, node)

    return graph


def create_node_neighbours_list(
    graph: nx.Graph,
    node_neighbours: dict[int, set[int]],
    vertex_mapping_index: np.ndarray | None = None,
    self_loops: bool = False,
) -> list[tuple]:
    """Preprocesses the dict of node neighbours.

    Parameters:
    -----------
    graph: nx.Graph
        The graph.
    node_neighbours: dict[int, set[int]]
        dictionairy with key: node index and value: set of neighbour node indices
    vertex_mapping_index: np.ndarry
        Index to map the vertices from the refined sphere to the original one, by default None.
    self_loops: bool
        Whether is supported to add self-loops, by default False.

    Returns:
    --------
    list: tuple
        A list with containing node neighbour pairs in tuples
    """
    graph_nodes_idx = list(sorted(graph.nodes))

    if vertex_mapping_index is None:
        vertex_mapping_index = np.arange(len(graph.nodes)).reshape(len(graph.nodes), 1)

    neighbour_pairs = [
        (graph_nodes_idx[vertex_mapping_index[node_neighbour][0]], graph_nodes_idx[vertex_mapping_index[node][0]])
        for node, neighbours in node_neighbours.items()
        for node_neighbour in neighbours
        if node != node_neighbour or (self_loops and node == node_neighbour)
    ]

    return neighbour_pairs
