# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os.path

from tessellate_ipu.utils import env_cpath_append

# Local include PATH update.
env_cpath_append(os.path.join(os.path.dirname(__file__), "vertex"))
