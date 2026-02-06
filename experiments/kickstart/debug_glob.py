
import glob
import os
import time

BUFFER_PATTERN = "data/kickstart_buffer.pkl.part_*"

def check():
    print(f"CWD: {os.getcwd()}")
    files = glob.glob(BUFFER_PATTERN)
    files.sort(key=os.path.getmtime)
    
    print(f"Found {len(files)} files.")
    total_bytes = 0
    for f in files:
        sz = os.path.getsize(f)
        total_bytes += sz
        print(f" - {f} : {sz/1024/1024:.2f} MB (Time: {time.ctime(os.path.getmtime(f))})")
        
    gb = total_bytes / (1024**3)
    print(f"Total Size: {gb:.2f} GB")

if __name__ == "__main__":
    check()
