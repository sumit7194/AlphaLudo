"""Background monitor for Rust Teacher Generator progress.

Writes td_ludo/generator_stats.json so the web dashboard can render real-time
progress of dataset generation. Reads both shard formats: v1 (20-byte records,
no header) and v2 (b"ALSL" header with the record count). Game counts come
from the data (v2 game ids / v1 all-base game starts would need a full read,
so the shard count x shard size is used, as the generator writes full shards).
"""
import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def shard_records(path: Path) -> int:
    with open(path, "rb") as f:
        head = f.read(16)
    if head[:4] == b"ALSL":
        return int.from_bytes(head[8:12], "little")
    return path.stat().st_size // 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default=str(ROOT / "data/sl_teacher_v2"))
    ap.add_argument("--log", default=str(ROOT / "sl_generator.log"))
    ap.add_argument("--target-games", type=int, default=1_000_000)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--out", default=str(ROOT / "td_ludo/generator_stats.json"))
    a = ap.parse_args()
    shard_dir, log_file, out_file = Path(a.shard_dir), Path(a.log), Path(a.out)
    target_shards = (a.target_games + a.shard_size - 1) // a.shard_size

    while True:
        try:
            shards = sorted(shard_dir.glob("shard_*.bin"))
            games = min(a.target_games, len(shards) * a.shard_size)
            total_states = sum(shard_records(s) for s in shards)
            gpm, sps = 0.0, 0.0
            if log_file.exists():
                for line in reversed(log_file.read_text(errors="ignore").splitlines()):
                    # v2 log: "| 1822.1 GPM (4568 SPS) |"; v1 log: "Shard Rate: X GPM (Y SPS)"
                    m = re.search(r"([0-9.]+)\s+GPM\s+\(([0-9.]+)\s+SPS\)", line)
                    if m:
                        gpm, sps = float(m.group(1)), float(m.group(2))
                        break
            data = {
                "shards_completed": len(shards),
                "target_shards": target_shards,
                "games_completed": games,
                "target_games": a.target_games,
                "progress_pct": round(100.0 * games / a.target_games, 1),
                "total_states": total_states,
                "gpm": gpm,
                "sps": sps,
                "timestamp": time.time(),
            }
            tmp = out_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(out_file)
        except Exception:
            pass
        time.sleep(3)


if __name__ == "__main__":
    main()
