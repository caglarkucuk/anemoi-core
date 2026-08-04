# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import logging
from typing import Literal
from typing import Optional

from pydantic import Field

from anemoi.graphs.schemas.normalise import ImplementedNormalisationSchema
from anemoi.utils.schemas import BaseModel

LOGGER = logging.getLogger(__name__)


class PlanarAreaWeightSchema(BaseModel):
    target_: Literal[
        "anemoi.graphs.nodes.attributes.PlanarAreaWeights",
        "anemoi.graphs.nodes.attributes.UniformWeights",
        "anemoi.graphs.nodes.attributes.CosineLatWeightedAttribute",
        "anemoi.graphs.nodes.attributes.IsolatitudeAreaWeights",
    ] = Field(..., alias="_target_")
    "Implementation of the area of the nodes as the weights from anemoi.graphs.nodes.attributes."
    norm: ImplementedNormalisationSchema = Field(example="unit-max")
    "Normalisation of the weights."


class MaskedPlanarAreaWeightsSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.attributes.MaskedPlanarAreaWeights"] = Field(..., alias="_target_")
    "Implementation of the area of the nodes as the weights from anemoi.graphs.nodes.attributes."
    mask_node_attr_name: str = Field(examples="cutout_mask")
    "Attribute name to mask the area weights."
    norm: ImplementedNormalisationSchema = Field(example="unit-max")
    "Normalisation of the weights."


class SphericalAreaWeightSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.attributes.SphericalAreaWeights"] = Field(..., alias="_target_")
    "Implementation of the 3D area of the nodes as the weights from anemoi.graphs.nades.attributes."
    norm: ImplementedNormalisationSchema = Field(example="unit-max")
    "Normalisation of the weights."
    fill_value: float = Field(example=0)
    "Value to fill the empty regions."


class CutOutMaskSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.attributes.CutOutMask", "anemoi.graphs.nodes.attributes.LimitedAreaMask"] = (
        Field(..., alias="_target_")
    )
    "Implementation of the area masks from anemoi.graphs.nodes.attributes."


class GridsMaskSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.attributes.GridsMask"] = Field(..., alias="_target_")
    "Implementation of the grids mask from anemoi.graphs.nodes.attributes."
    grids: list[int] | int = Field(examples=[0, [0]])
    "Position of the grids to consider as True."


class NonmissingAnemoiDatasetVariableSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.attributes.NonmissingAnemoiDatasetVariable"] = Field(..., alias="_target_")
    (
        "Implementation of a mask from the nonmissing values of a anemoi-datasets variable "
        "from anemoi.graphs.nodes.attributes."
    )
    variable: str
    "The anemoi-datasets variable to use."


class MeshLevelSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.attributes.MeshLevel"] = Field(..., alias="_target_")
    "Refinement level of each node of an adaptive icosahedral mesh, from anemoi.graphs.nodes.attributes."
    norm: Optional[ImplementedNormalisationSchema] = None
    "Normalisation of the levels. Defaults to none, keeping the raw level index."


class MeshLevelMaskSchema(BaseModel):
    target_: Literal["anemoi.graphs.nodes.attributes.MeshLevelMask"] = Field(..., alias="_target_")
    "Mask selecting the nodes of an adaptive mesh at given refinement levels, from anemoi.graphs.nodes.attributes."
    level: int | list[int] = Field(examples=[10, [10, 11]])
    "Refinement level(s) to select."


SingleAttributeSchema = (
    PlanarAreaWeightSchema
    | MaskedPlanarAreaWeightsSchema
    | SphericalAreaWeightSchema
    | CutOutMaskSchema
    | GridsMaskSchema
    | NonmissingAnemoiDatasetVariableSchema
    | MeshLevelSchema
    | MeshLevelMaskSchema
)


class BooleanOperationSchema(BaseModel):
    target_: Literal[
        "anemoi.graphs.nodes.attributes.BooleanNot",
        "anemoi.graphs.nodes.attributes.BooleanAndMask",
        "anemoi.graphs.nodes.attributes.BooleanOrMask",
    ] = Field(..., alias="_target_")
    "Implementation of boolean masks from anemoi.graphs.nodes.attributes"
    masks: str | SingleAttributeSchema | list[str | SingleAttributeSchema]


NodeAttributeSchemas = SingleAttributeSchema | BooleanOperationSchema
