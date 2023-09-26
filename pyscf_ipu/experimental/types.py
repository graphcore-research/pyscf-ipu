# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Tuple

from jaxtyping import Array, Float, Int

Float3 = Float[Array, "3"]
Float3xNxN = Float[Array, "3 N N"]
Float3xNxNxNxN = Float[Array, "3 N N N N"]
FloatNx3 = Float[Array, "N 3"]
FloatN = Float[Array, "N"]
FloatNxN = Float[Array, "N N"]
FloatNxM = Float[Array, "N M"]
Int3 = Int[Array, "3"]
IntN = Int[Array, "N"]

MeshAxes = Tuple[FloatN, FloatN, FloatN]
