"""This is a helper module to deal with the directory structure"""

import sys
import os

# This should be the directory we started in
base = sys.path[0]
def base_path(*path):
    return os.path.normpath(os.path.join(base, "..", *path))

# Help us find the c++ python libraries
if os.environ.has_key("DEBUG"):
    if os.environ.has_key("EMU"):
        sys.path.insert(0, base_path("src", "emudebug"))
    else:
        sys.path.insert(0, base_path("src", "debug"))
else:
    if os.environ.has_key("EMU"):
        sys.path.insert(0, base_path("src", "emurelease"))
    else:
        sys.path.insert(0, base_path("src", "release"))

# Help us our python code
sys.path.insert(0, base_path("python"))

