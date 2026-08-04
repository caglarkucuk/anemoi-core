# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from anemoi.graphs import EARTH_RADIUS
from anemoi.graphs.generate.transforms import latlon_rad_to_cartesian_np

LOGGER = logging.getLogger(__name__)

GRAVITY = 9.80665  # m s-2, to turn surface geopotential into elevation


class BaseLevelField(ABC):
    """Target refinement level requested at each point of the sphere.

    Consumed by AdaptiveTriNodes to decide which vertices of a refined icosahedron to keep.
    Implementations only have to answer "how fine should the mesh be here"; clipping to the
    resolution range and enforcing gradation are handled by the caller, so a level field may
    return any integer values it likes.
    """

    @abstractmethod
    def get_levels(self, coords_rad: np.ndarray) -> np.ndarray:
        """Target refinement level at each coordinate.

        Parameters
        ----------
        coords_rad : np.ndarray of shape (num_vertices, 2)
            Latitude and longitude of the query points, in radians.

        Returns
        -------
        np.ndarray of shape (num_vertices, )
            Requested refinement level at each point.
        """
        ...


class BoundingBoxLevelField(BaseLevelField):
    """Uniform refinement inside a latitude/longitude box.

    Mostly useful for testing and for hand-authored control experiments, where a mesh with a
    known, trivially describable refinement pattern is wanted.

    Attributes
    ----------
    level : int
        Refinement level requested inside the box.
    background_level : int
        Refinement level requested outside the box.
    latlon_bbox : list[float]
        Bounding box as [min_lat, min_lon, max_lat, max_lon], in degrees.
    """

    def __init__(self, level: int, background_level: int, latlon_bbox: list[float]) -> None:
        assert len(latlon_bbox) == 4, "latlon_bbox must be [min_lat, min_lon, max_lat, max_lon]."
        assert level >= background_level, "The level inside the box cannot be coarser than the background."
        self.level = level
        self.background_level = background_level
        self.latlon_bbox = latlon_bbox

    def get_levels(self, coords_rad: np.ndarray) -> np.ndarray:
        min_lat, min_lon, max_lat, max_lon = np.deg2rad(self.latlon_bbox)
        lat, lon = coords_rad[:, 0], coords_rad[:, 1]
        inside = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
        LOGGER.info("%s: %d of %d vertices inside the box.", self.__class__.__name__, inside.sum(), len(inside))
        return np.where(inside, self.level, self.background_level)


class TerrainComplexityLevelField(BaseLevelField):
    r"""Refinement driven by unresolved orography.

    Implements the standard adaptive-mesh feature indicator: refine from level r to r+1
    wherever the terrain varies too much *within a level-r cell* to be represented by a single
    node. Complexity is measured as the standard deviation of surface elevation over the data
    nodes falling in each cell,

    .. math::
        C_r(\text{cell}) = \mathrm{std}\{\, h(x) : x \in \text{cell} \,\},

    and level r+1 is requested wherever :math:`C_r > \tau_r`.

    Measuring the spread at the cell scale is what makes the criterion scale-consistent: the
    same threshold means the same thing at every level, and the test naturally stops refining
    once the terrain is resolved. Note this is deliberately *not* the ``sdor`` field, which is
    a sub-grid standard deviation and therefore means "within 31 km" in an ERA5 dataset but
    "within 2.5 km" in a kilometre-scale one -- stitching the two across a cutout boundary
    would bias refinement towards the regional block for a purely artefactual reason.

    The field only ever *requests* refinement: it returns 0 where no threshold is met, so that
    it composes with the resolution floors set by AdaptiveTriNodes rather than overriding them.

    Attributes
    ----------
    dataset : str | dict
        Anemoi dataset holding the orography, normally the same one used for the data nodes.
    thresholds : dict[int, float]
        Maps a level r to the elevation standard deviation, in metres, above which a level-r
        cell is refined to r+1.
    quantiles : dict[int, float]
        Alternative to `thresholds`: maps a level r to a quantile of the observed distribution
        of :math:`C_r`, so that e.g. 0.8 refines the most complex fifth of the populated cells.
    variable : str
        Name of the surface geopotential variable, by default "z".
    min_points_per_cell : int
        Cells containing fewer data nodes than this are left unrefined, since their standard
        deviation is not meaningful.
    cache_path : str | Path
        Optional npz file used to persist the per-data-node result between graph builds.
    """

    def __init__(
        self,
        dataset: str | dict,
        thresholds: dict[int, float] | None = None,
        quantiles: dict[int, float] | None = None,
        variable: str = "z",
        min_points_per_cell: int = 4,
        min_neighbours_to_keep: int = 2,
        min_neighbours_to_fill: int = 5,
        cleanup_passes: int = 1,
        neighbour_radius_factor: float = 1.3,
        cache_path: str | Path | None = None,
    ) -> None:
        assert (thresholds is None) != (quantiles is None), "Provide exactly one of 'thresholds' or 'quantiles'."
        self.dataset = dataset
        self.thresholds = {int(k): float(v) for k, v in thresholds.items()} if thresholds else None
        self.quantiles = {int(k): float(v) for k, v in quantiles.items()} if quantiles else None
        self.variable = variable
        self.min_points_per_cell = min_points_per_cell
        self.min_neighbours_to_keep = min_neighbours_to_keep
        self.min_neighbours_to_fill = min_neighbours_to_fill
        self.cleanup_passes = cleanup_passes
        self.neighbour_radius_factor = neighbour_radius_factor
        self.cache_path = Path(cache_path) if cache_path else None

        self._node_xyz: np.ndarray | None = None
        self._node_levels: np.ndarray | None = None
        self._tree: cKDTree | None = None
        self.diagnostics: dict[int, dict] = {}
        # Per-data-node complexity and refinement decision at each level, kept for plotting.
        self.node_complexity: dict[int, np.ndarray] = {}
        self.node_refined: dict[int, np.ndarray] = {}

    @property
    def levels(self) -> list[int]:
        """Levels at which a refinement test is applied, coarsest first."""
        return sorted((self.thresholds or self.quantiles).keys())

    def _read_elevation(self) -> tuple[np.ndarray, np.ndarray]:
        """Elevation and node coordinates from the dataset."""
        from anemoi.datasets import open_dataset

        LOGGER.info("%s: reading '%s' for orography.", self.__class__.__name__, self.variable)
        ds = open_dataset(self.dataset, select=self.variable)

        # Orography is static, so any valid step will do -- but a cutout over a dataset with
        # gaps can have a missing step 0, which raises rather than returning anything.
        missing = getattr(ds, "missing", set()) or set()
        index = next((i for i in range(len(ds)) if i not in missing), None)
        assert index is not None, f"{self.__class__.__name__}: the dataset has no valid time step."
        if index != 0:
            LOGGER.info("%s: step 0 is missing, reading orography from step %d.", self.__class__.__name__, index)

        elevation = np.asarray(ds[index]).squeeze() / GRAVITY
        coords_rad = np.deg2rad(np.stack([ds.latitudes, ds.longitudes], axis=-1))
        LOGGER.info(
            "%s: %d nodes, elevation %.0f to %.0f m (mean %.0f m).",
            self.__class__.__name__,
            len(elevation),
            elevation.min(),
            elevation.max(),
            elevation.mean(),
        )
        return elevation, coords_rad

    def _cell_adjacency(self, cell_xyz: np.ndarray, spacing_km: float):
        """Ring-1 adjacency between cells, as a symmetric sparse matrix.

        Built over the populated cells only, which is a small set compared with the full
        icosphere, and by radius rather than from the mesh faces so that it does not depend on
        having the finest icosphere in memory.
        """
        from scipy.sparse import csr_matrix

        radius = 2 * np.sin(self.neighbour_radius_factor * spacing_km / (2 * EARTH_RADIUS))
        pairs = cKDTree(cell_xyz).query_pairs(r=radius, output_type="ndarray")
        ones = np.ones(len(pairs), dtype=np.int8)
        n_cells = len(cell_xyz)
        adjacency = csr_matrix((ones, (pairs[:, 0], pairs[:, 1])), shape=(n_cells, n_cells))
        return adjacency + adjacency.T

    def _clean_refinement(self, refine: np.ndarray, cell_xyz: np.ndarray, spacing_km: float) -> np.ndarray:
        """Remove isolated refined cells and fill pinholes in otherwise refined regions.

        Thresholding a complexity field cell by cell yields a coherent core plus scattered
        single cells and small holes. Both are bad for the mesh: an isolated refined cell spawns
        level-(r+1) vertices whose only surviving neighbours are their two parents, and a pinhole
        punches a low-connectivity ring through the interior of a refined region.

        These are rank filters over the ring-1 neighbourhood rather than strict binary erosion
        and dilation, because the number of neighbours is not constant on an icosphere -- the
        twelve pentagonal vertices have five -- and a "how many of my neighbours agree" rule
        degrades gracefully where a structuring element would not.

        Parameters
        ----------
        refine : np.ndarray of shape (num_populated_cells, )
            Boolean refinement decision per populated cell.
        cell_xyz : np.ndarray of shape (num_populated_cells, 3)
            Cell centres as unit vectors.
        spacing_km : float
            Mean spacing of the cell mesh, used to size the neighbourhood.

        Returns
        -------
        np.ndarray
            The cleaned refinement decision.
        """
        if self.min_neighbours_to_keep <= 0 and self.min_neighbours_to_fill <= 0:
            return refine

        adjacency = self._cell_adjacency(cell_xyz, spacing_km)
        cleaned = refine.copy()

        for _ in range(self.cleanup_passes):
            neighbours_refined = adjacency @ cleaned.astype(np.int32)

            if self.min_neighbours_to_keep > 0:
                dropped = cleaned & (neighbours_refined < self.min_neighbours_to_keep)
                cleaned = cleaned & ~dropped
                neighbours_refined = adjacency @ cleaned.astype(np.int32)
            else:
                dropped = np.zeros_like(cleaned)

            if self.min_neighbours_to_fill > 0:
                filled = ~cleaned & (neighbours_refined >= self.min_neighbours_to_fill)
                cleaned = cleaned | filled
            else:
                filled = np.zeros_like(cleaned)

            LOGGER.debug("  cleanup pass: dropped %d isolated, filled %d holes", dropped.sum(), filled.sum())
            if not dropped.any() and not filled.any():
                break

        return cleaned

    @staticmethod
    def _cell_std(values: np.ndarray, cell_of: np.ndarray, num_cells: int) -> tuple[np.ndarray, np.ndarray]:
        """Standard deviation of `values` within each cell, and the cell populations."""
        counts = np.bincount(cell_of, minlength=num_cells)
        total = np.bincount(cell_of, weights=values, minlength=num_cells)
        total_sq = np.bincount(cell_of, weights=values**2, minlength=num_cells)

        with np.errstate(invalid="ignore", divide="ignore"):
            mean = total / counts
            variance = total_sq / counts - mean**2
        # Cancellation can push an exactly-flat cell marginally negative.
        return np.sqrt(np.clip(variance, 0.0, None)), counts

    def fit(self) -> None:
        """Compute the requested refinement level at each data node."""
        if self._node_levels is not None:
            return

        if self.cache_path is not None and self.cache_path.exists():
            LOGGER.info("%s: loading cached levels from %s.", self.__class__.__name__, self.cache_path)
            cached = np.load(self.cache_path)
            self._node_xyz, self._node_levels = cached["node_xyz"], cached["node_levels"]
            self._tree = cKDTree(self._node_xyz)
            return

        from anemoi.graphs.generate.tri_icosahedron import get_latlon_coords_icosphere
        from anemoi.graphs.generate.tri_icosahedron import get_mean_spacing_km

        elevation, coords_rad = self._read_elevation()
        self._node_xyz = latlon_rad_to_cartesian_np(coords_rad)
        requested = np.zeros(len(elevation), dtype=np.int16)

        for level in self.levels:
            # Assign every data node to its nearest level-r vertex: that is the level-r cell.
            cell_vertices = latlon_rad_to_cartesian_np(get_latlon_coords_icosphere(level))
            _, cell_of = cKDTree(cell_vertices).query(self._node_xyz, k=1, workers=-1)

            complexity, counts = self._cell_std(elevation, cell_of, len(cell_vertices))
            populated = counts >= self.min_points_per_cell

            if self.thresholds is not None:
                threshold = self.thresholds[level]
            else:
                threshold = float(np.quantile(complexity[populated], self.quantiles[level]))

            refine = populated & (complexity > threshold)
            raw_refined = int(refine.sum())

            # Tidy the refined region before it reaches the mesh: the raw threshold leaves
            # isolated cells and pinholes, which become poorly connected mesh nodes.
            populated_idx = np.flatnonzero(populated)
            cleaned = self._clean_refinement(
                refine[populated_idx], cell_vertices[populated_idx], get_mean_spacing_km(level)
            )
            refine = np.zeros_like(refine)
            refine[populated_idx] = cleaned

            requested = np.where(refine[cell_of], np.maximum(requested, level + 1), requested).astype(np.int16)

            # Project the per-cell quantities back onto the data nodes so they can be mapped.
            self.node_complexity[level] = np.where(populated[cell_of], complexity[cell_of], np.nan)
            self.node_refined[level] = refine[cell_of]

            self.diagnostics[level] = {
                "cell_spacing_km": get_mean_spacing_km(level),
                "threshold_m": threshold,
                "populated_cells": int(populated.sum()),
                "refined_cells_raw": raw_refined,
                "refined_cells": int(refine.sum()),
                "cleanup_delta": int(refine.sum()) - raw_refined,
                "refined_fraction": float(refine.sum() / max(populated.sum(), 1)),
                "complexity_percentiles_m": np.percentile(complexity[populated], [50, 75, 90, 95, 99]).tolist(),
            }
            LOGGER.info(
                "%s: level %d (%.1f km cells) threshold %.1f m -> %d of %d populated cells refined "
                "(%.1f%%); cleanup %+d cells from %d raw.",
                self.__class__.__name__,
                level,
                get_mean_spacing_km(level),
                threshold,
                refine.sum(),
                populated.sum(),
                100 * refine.sum() / max(populated.sum(), 1),
                int(refine.sum()) - raw_refined,
                raw_refined,
            )

        self._node_levels = requested
        self._tree = cKDTree(self._node_xyz)

        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.cache_path, node_xyz=self._node_xyz, node_levels=self._node_levels)
            LOGGER.info("%s: cached levels to %s.", self.__class__.__name__, self.cache_path)

    def get_levels(self, coords_rad: np.ndarray) -> np.ndarray:
        """Requested refinement level at each query point, by nearest data node."""
        self.fit()
        LOGGER.info("%s: mapping %d query points onto the data nodes.", self.__class__.__name__, len(coords_rad))
        _, nearest = self._tree.query(latlon_rad_to_cartesian_np(coords_rad), k=1, workers=-1)
        return self._node_levels[nearest]
