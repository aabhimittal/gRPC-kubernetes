"""Make the service modules importable as `model`, `batcher`, ... like the
server does, without installing the package."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
