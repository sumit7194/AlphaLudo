import json
import os
import glob

RUN_NAME = "mastery_v2"
BASE_DIR = f"checkpoints_mastery/{RUN_NAME}"
ELO_PATH = f"{BASE_DIR}/elo_ratings.json"
GHOST_DIR = f"{BASE_DIR}/ghosts"

def cleanup_ghosts():
    # 1. Load Ratings
    if not os.path.exists(ELO_PATH):
        print(f"Elo file not found: {ELO_PATH}")
        return

    with open(ELO_PATH, 'r') as f:
        data = json.load(f)
    
    ratings = data.get('ratings', {})
    
    # 2. Identify Ghosts
    ghost_ratings = {k: v for k, v in ratings.items() if k.startswith('ghost_')}
    
    if not ghost_ratings:
        print("No ghosts found in Elo ratings.")
        return

    print(f"Found {len(ghost_ratings)} ghosts in Elo registry.")

    # 3. Sort and Keep Top 10
    sorted_ghosts = sorted(ghost_ratings.items(), key=lambda x: x[1], reverse=True)
    top_10 = sorted_ghosts[:10]
    top_10_keys = set(k for k, v in top_10)
    
    print("\nTop 10 Ghosts Keeping:")
    for k, v in top_10:
        print(f"  {k}: {v:.1f}")

    # 4. Update JSON structure
    # Remove non-top ghosts from 'ratings'
    # keys_to_remove = set(ghost_ratings.keys()) - top_10_keys
    # for k in keys_to_remove:
    #     del ratings[k]
    
    # Actually, user said "only keep top 10 ... deleting other old ghosts".
    # I should remove them from JSON too to keep it clean.
    new_ratings = {k: v for k, v in ratings.items() if not k.startswith('ghost_') or k in top_10_keys}
    data['ratings'] = new_ratings
    
    # Save JSON
    with open(ELO_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nUpdated {ELO_PATH} (Removed old ghosts).")

    # 5. Delete Files
    files = glob.glob(os.path.join(GHOST_DIR, "*.pt"))
    deleted_count = 0
    kept_count = 0
    
    for kv_path in files:
        filename = os.path.basename(kv_path) # ghost_123.pt
        ghost_key = filename.replace('.pt', '') # ghost_123
        
        if ghost_key in top_10_keys:
            kept_count += 1
        else:
            try:
                os.remove(kv_path)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {kv_path}: {e}")
                
    print(f"\nFile Cleanup Complete:")
    print(f"  Deleted: {deleted_count} files")
    print(f"  Kept:    {kept_count} files (Top 10)")

if __name__ == "__main__":
    cleanup_ghosts()
