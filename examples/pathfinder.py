"""This is a helper module to deal with the directory structure"""

import sys
import os
import platform

# This should be the directory we started in
base = sys.path[0]
def base_path(*path):
    return os.path.normpath(os.path.join(base, "..", *path))

# Help us find the c++ python libraries
if platform.system() == "Windows":
    if os.environ.has_key("DEBUG"):
        print("Importing DEBUG binary.")
        sys.path.insert(0, base_path("bin", "x64", "Debug"))
    else:
        print("Importing RELEASE binary. Define DEBUG evironment variable to run in DEBUG mode.")
        sys.path.insert(0, base_path("bin", "x64", "Release"))
else:        
    if os.environ.has_key("DEBUG"):
        print("Importing DEBUG binary.")
        if os.environ.has_key("EMU"):
            sys.path.insert(0, base_path("src", "emudebug"))
        else:
            sys.path.insert(0, base_path("src", "debug"))
    else:
        print("Importing RELEASE binary. Define DEBUG evironment variable to run in DEBUG mode.")
        if os.environ.has_key("EMU"):
            sys.path.insert(0, base_path("src", "emurelease"))
        else:
            sys.path.insert(0, base_path("src", "release"))

# Help us our python code
sys.path.insert(0, base_path("python"))

