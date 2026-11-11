"""Certify the hyperedges of a small synthetic input."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hyrange.quick import certify_input


pair_counts = {(0, 1): 1, (0, 2): 1, (1, 2): 1, (2, 3): 1}
degrees = [1, 1, 2, 1]
result = certify_input(pair_counts, degrees, len(degrees))
print(dict(sorted(result['certified'].items())))
