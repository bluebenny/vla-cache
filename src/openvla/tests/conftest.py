import os
import sys

OPENVLA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if OPENVLA_ROOT not in sys.path:
    sys.path.insert(0, OPENVLA_ROOT)

from experiments.robot.tiny_openvla import install_light_import_stubs  # noqa: E402

# Bare `prismatic` package, tensorflow stub, CUDA-event stubs (see tiny_openvla.py)
install_light_import_stubs()
