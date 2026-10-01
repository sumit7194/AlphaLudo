"""Background monitor for Rust Teacher Generator progress.

Writes td_ludo/generator_stats.json so the web dashboard can render real-time
progress of the 1,000,000-game dataset generation.
"""
from pathlib import Path
import re
import time
import json

ROOT = Path(__file__).resolve().parent.parent.parent
SHARD_DIR = ROOT / "data/sl_teacher_v153"
OUT_FILE = ROOT / "td_ludo/generator_stats.json"
LOG_FILE = ROOT / "sl_generator.log"

def main():
    while True:
        try:
            shards = list(SHARD_DIR.glob("shard_*.bin"))
            num_shards = len(shards)
            games = num_shards * 5000
            total_states = sum(s.stat().st_size for s in shards) // 20

            gpm = 1000.0
            sps = 2500.0
            if LOG_FILE.exists():
                with open(LOG_FILE, "r") as f:
                    lines = f.readlines()
                for line in reversed(lines):
                    m = re.search(r"Shard Rate:\s+([0-9.]+)\s+GPM\s+\(([0-9.]+)\s+SPS\)", line)
                    if m:
                        gpm = float(m.group(1))
                        sps = float(m.group(2))
                        break

            data = {
                "shards_completed": num_shards,
                "target_shards": 200,
                "games_completed": games,
                "target_games": 1000000,
                "progress_pct": round(games / 10000, 1),
                "total_states": total_states,
                "gpm": gpm,
                "sps": sps,
                "timestamp": time.time(),
            }
            with open(OUT_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
        time.sleep(3)

if __name__ == "__main__":
    main()
