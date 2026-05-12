import os, sys
# Allow internal model code that uses 'from models.dinov2/dinov3...' to resolve
# against the src/ directory where models/ now lives.
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
