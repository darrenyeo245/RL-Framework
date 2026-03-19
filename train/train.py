import sys
from pathlib import Path

# Ensure local packages (`env`, `osc`) are resolved before site-packages modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from env import MediaEnv
from osc.osc_interface import OSCInterface

osc = OSCInterface()
env = MediaEnv(osc)

state, info = env.reset()
step_idx = 0

print("Training gestartet. Jeder Schritt wartet auf ein Reward-Signal (/reward).")

while True:
    action = np.random.uniform(-1, 1, 3)

    print(f"\n--- Step {step_idx} ---")
    print("Input action:", np.array2string(action, precision=3))
    print("Warte auf Reward (+ oder - in reward_input.py)...")

    state, reward, terminated, truncated, info = env.step(action)

    print("Output state:", np.array2string(np.asarray(state), precision=3))
    print("Output reward:", reward)

    step_idx += 1

    if terminated or truncated:
        state, info = env.reset()
