# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import json
import subprocess

j = subprocess.check_output(["gc-monitor", "-j"])
data = json.loads(j)
num_ipuMs = len(data["cards"])
num_ipus = 4 * num_ipuMs

# to be captured as a variable in the bash script that calls this python script
print(num_ipus)
