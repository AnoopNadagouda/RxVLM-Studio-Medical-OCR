import os
import sys
import time
import subprocess
from pathlib import Path

CHECKPOINT_DIR = Path("./checkpoints/stage2_multiline_best")
EVAL_SCRIPT = Path("scripts/evaluate_stage2.py")

def main():
    print("=" * 70)
    print("Automated Checkpoint Monitor & Evaluation Trigger (Stage 2)")
    print(f"Target Checkpoint Dir: {CHECKPOINT_DIR.resolve()}")
    print("=" * 70)

    adapter_config = CHECKPOINT_DIR / "adapter_config.json"
    adapter_weights = CHECKPOINT_DIR / "adapter_model.safetensors"
    bin_weights = CHECKPOINT_DIR / "adapter_model.bin"

    print("Monitoring directory for initial checkpoint...", flush=True)

    max_wait_seconds = 3600  # 1 hour timeout limit
    start_time = time.time()

    while True:
        checkpoint_found = adapter_config.exists() and (adapter_weights.exists() or bin_weights.exists())

        if checkpoint_found:
            print(f"\n[ALERT] First checkpoint detected in {CHECKPOINT_DIR}!", flush=True)
            print("Launching automated evaluation pipeline...", flush=True)
            
            cmd = [sys.executable, str(EVAL_SCRIPT), "50"]
            res = subprocess.run(cmd, capture_output=False)
            
            print(f"\n[Auto-Eval] Evaluation finished with exit code {res.returncode}", flush=True)
            break

        if time.time() - start_time > max_wait_seconds:
            print("\n[Timeout] Reached 1 hour monitoring limit without detecting checkpoint.", flush=True)
            break

        time.sleep(5)

if __name__ == "__main__":
    main()
