#!/usr/bin/env python
"""Run local ES training simulation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cpu_distributed.cli import main
if __name__ == "__main__": sys.exit(main(["local"] + sys.argv[1:]))
