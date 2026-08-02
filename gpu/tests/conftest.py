"""Make the repository root importable so ``gpu.*`` resolves no matter
where pytest is invoked from."""

import os
import sys

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
