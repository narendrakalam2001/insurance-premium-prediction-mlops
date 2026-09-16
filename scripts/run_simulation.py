# ============================================================
# RUN SIMULATION — runner script
# ============================================================

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from simulation.policyholder_simulator import simulate_applications

if __name__ == "__main__":
    simulate_applications(n=20, scenario="random")
