import os
import shutil

def migrate():
    src = "checkpoints/td_v2_11ch/experience_buffer.npz"
    dest_dir = "checkpoints/td_v3_small"
    dest = os.path.join(dest_dir, "experience_buffer.npz")
    
    if not os.path.exists(src):
        print(f"Error: Source buffer not found at {src}")
        return
        
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Copying {src} to {dest}...")
    shutil.copy2(src, dest)
    
    src_size = os.path.getsize(src) / (1024 * 1024)
    dest_size = os.path.getsize(dest) / (1024 * 1024)
    
    print(f"Migration complete!")
    print(f"Source size: {src_size:.2f} MB")
    print(f"Dest size:   {dest_size:.2f} MB")

if __name__ == "__main__":
    migrate()
