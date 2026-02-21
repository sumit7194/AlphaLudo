import os
import subprocess
import re
import time

MODELS = [
    ("Main Kickstart", "experiments/kickstart/model_kickstart.pt"),
    ("Ghost 120", "experiments/kickstart/ghosts/ghost_cycle_120.pt"),
    ("Ghost 140", "experiments/kickstart/ghosts/ghost_cycle_140.pt"),
    ("Ghost 160", "experiments/kickstart/ghosts/ghost_cycle_160.pt"),
]

GAMES = 400

print(f"Starting Comparative Evaluation of {len(MODELS)} models.")
print(f"Games per model: {GAMES}")
print("-" * 60)

results = []

for name, path in MODELS:
    if not os.path.exists(path):
        print(f"❌ Missing: {path}")
        results.append((name, "N/A", "File not found"))
        continue

    print(f"Testing {name}...")
    start_time = time.time()
    
    # Run test_pure_model.py
    # We use subprocess to isolate runs
    cmd = [
        "python3", "-m", "src.test_pure_model",
        "--games", str(GAMES),
        "--checkpoint", path
    ]
    
    try:
        # Capture stdout
        ENV = os.environ.copy()
        ENV["ALPHALUDO_MODE"] = "TEST" # Ensure test config
        
        process = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
        output = process.stdout
        
        # Parse output for Win Rate
        # "Win Rate:           25.3%"
        match = re.search(r"Win Rate:\s+([\d\.]+)%", output)
        if match:
            win_rate = float(match.group(1))
            status = "✅ Done"
        else:
            win_rate = 0.0
            status = "❌ Parse Error"
            print(output[-500:]) # Print tail if error

        elapsed = time.time() - start_time
        print(f"   -> Result: {win_rate}% ({elapsed:.1f}s)")
        results.append((name, win_rate, status))

    except Exception as e:
        print(f"   -> Error: {e}")
        results.append((name, 0.0, "Error"))

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"{'Model Name':<20} | {'Win Rate':<10} | {'Status':<10}")
print("-" * 60)
for name, rate, status in results:
    rate_str = f"{rate}%" if isinstance(rate, float) else str(rate)
    print(f"{name:<20} | {rate_str:<10} | {status:<10}")
print("="*60)
