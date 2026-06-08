#!/usr/bin/env python
"""Start the coordinator server."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
print("Coordinator: use Coordinator class from cpu_distributed.coordinator")
print("Example: python -c 'from cpu_distributed.coordinator import Coordinator; ...'")
