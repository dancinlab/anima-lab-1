"""Put the repo root on the import path for every test in this directory.

Modules under test live at the repo root (bench_v2.py, mitosis.py, trinity.py …),
so tests either do a sys.path dance at the top of each file — which forces
imports below code and provokes an E402 suppression — or it happens once here.
Once here.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
