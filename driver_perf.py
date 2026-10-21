import time
import sys

import pytest

t0 = time.monotonic()
rc = pytest.main(["tests/test_longrun_bench.py", "-q", "-p", "no:cacheprovider"])
print(f"DRIVER wall={time.monotonic() - t0:.1f}s rc={rc}")
sys.exit(int(rc))
