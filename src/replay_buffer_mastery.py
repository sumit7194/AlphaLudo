"""
Replay Buffer for AlphaLudo Mastery Architecture - 18 Channel Async Plan.
Stores: (state, policy, value)
"""

import random
from collections import deque
import torch

class ReplayBufferMastery:
    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, examples):
        """
        Args:
            examples: List of (state, policy, value)
                state: Tensor [18, 15, 15]
                policy: Tensor [225]
                value: Tensor [1]
        """
        for example in examples:
            self.buffer.append(example)
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(list(self.buffer), batch_size)
        
        states = torch.stack([ex[0] for ex in batch])
        policies = torch.stack([ex[1] for ex in batch])
        values = torch.stack([ex[2] for ex in batch])
        
        return states, policies, values
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()

    def save(self, path):
        """
        Save buffer with corruption protection:
        1. Write to .tmp file first
        2. Include checksum for validation
        3. Backup previous version to .bak
        4. Atomic replace
        """
        import pickle
        import os
        import hashlib
        
        try:
            temp_path = path + ".tmp"
            backup_path = path + ".bak"
            
            # Serialize data with checksum
            data = list(self.buffer)
            serialized = pickle.dumps(data)
            checksum = hashlib.md5(serialized).hexdigest()
            
            # Write to temp file with checksum header
            with open(temp_path, 'wb') as f:
                # Write checksum as first line (32 bytes + newline)
                f.write(f"{checksum}\n".encode('utf-8'))
                f.write(serialized)
            
            # Backup existing file (if exists)
            if os.path.exists(path):
                try:
                    os.replace(path, backup_path)
                except:
                    pass  # Best effort backup
            
            # Atomic replace temp -> main
            os.replace(temp_path, path)
            print(f"[Buffer] Saved {len(self.buffer)} samples to {path}")
            
        except Exception as e:
            print(f"[Buffer] Failed to save: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    def load(self, path):
        """
        Load buffer with corruption recovery:
        1. Try main file first
        2. Validate checksum
        3. If corrupt, try .bak backup
        4. If both fail, start fresh
        """
        import pickle
        import os
        import hashlib
        
        def try_load_file(filepath):
            """Try to load and validate a buffer file. Returns data or None."""
            if not os.path.exists(filepath):
                return None
            
            try:
                with open(filepath, 'rb') as f:
                    # Read checksum (first line)
                    first_line = f.readline()
                    
                    # Check if this is new format (with checksum) or old format
                    try:
                        expected_checksum = first_line.decode('utf-8').strip()
                        if len(expected_checksum) == 32:  # MD5 hex length
                            # New format: validate checksum
                            serialized = f.read()
                            actual_checksum = hashlib.md5(serialized).hexdigest()
                            
                            if actual_checksum != expected_checksum:
                                print(f"[Buffer] Checksum mismatch in {filepath}")
                                return None
                            
                            return pickle.loads(serialized)
                    except:
                        pass
                    
                    # Old format: no checksum, just pickle data
                    f.seek(0)
                    return pickle.load(f)
                    
            except Exception as e:
                print(f"[Buffer] Failed to load {filepath}: {e}")
                return None
        
        # Try main file first
        data = try_load_file(path)
        if data is not None:
            self.buffer.extend(data)
            print(f"[Buffer] Loaded {len(data)} samples from {path}")
            return True
        
        # Try backup file
        backup_path = path + ".bak"
        data = try_load_file(backup_path)
        if data is not None:
            self.buffer.extend(data)
            print(f"[Buffer] Recovered {len(data)} samples from backup!")
            # Restore backup as main file
            try:
                import shutil
                shutil.copy(backup_path, path)
            except:
                pass
            return True
        
        # Try .tmp file (interrupted save)
        temp_path = path + ".tmp"
        data = try_load_file(temp_path)
        if data is not None:
            self.buffer.extend(data)
            print(f"[Buffer] Recovered {len(data)} samples from temp file!")
            return True
        
        print(f"[Buffer] No valid buffer found, starting fresh")
        return False

