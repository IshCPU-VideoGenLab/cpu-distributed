#!/usr/bin/env python
"""Start a worker node."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
print("Worker: use Worker class from cpu_distributed.worker")
print("Example: python -c 'from cpu_distributed.worker import Worker; ...'")
