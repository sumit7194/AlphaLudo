import sqlite3
import os
import subprocess

db_path = "checkpoints/td_v3_small/games.db"
backup_path = "checkpoints/td_v3_small/games.db.corrupted"
recovered_sql = "checkpoints/td_v3_small/recovered.sql"

if not os.path.exists(db_path):
    print(f"No database found at {db_path}")
    exit(1)

print("1. Renaming corrupted database...")
os.rename(db_path, backup_path)

print("2. Attempting SQLite recovery dump...")
try:
    # Use sqlite3 command line tool to dump the valid data from the corrupted DB
    with open(recovered_sql, "w") as f:
        subprocess.run(["sqlite3", backup_path, ".recover"], stdout=f, check=True)
    print("   Dump successful.")
except subprocess.CalledProcessError as e:
    print(f"   Note: sqlite3 .recover failed or is unavailable. Falling back to standard dump.")
    try:
        with open(recovered_sql, "w") as f:
            subprocess.run(["sqlite3", backup_path, ".dump"], stdout=f, check=True)
    except Exception as e2:
        print(f"   Standard dump also failed: {e2}")
        print("   Proceeding with a completely fresh database.")

print("3. Rebuilding database from valid recovered data...")
if os.path.exists(recovered_sql) and os.path.getsize(recovered_sql) > 0:
    try:
        with open(recovered_sql, "r") as f:
            sql_script = f.read()
            
        conn = sqlite3.connect(db_path)
        conn.executescript(sql_script)
        conn.commit()
        conn.close()
        print("   Rebuild successful!")
    except Exception as e:
        print(f"   Failed to rebuild from SQL: {e}")
        if os.path.exists(db_path):
            os.remove(db_path) # Clean up the broken attempt
else:
    print("   No valid data recovered. A fresh database will be created on next run.")

# Clean up temporary SQL file
if os.path.exists(recovered_sql):
    os.remove(recovered_sql)
    
print("\nRecovery process complete. You can now resume training.")
