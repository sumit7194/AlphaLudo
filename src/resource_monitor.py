"""
Resource Monitor for AlphaLudo Training Dashboard

Uses psutil to collect system resource statistics for monitoring.
"""

import psutil
import os
from typing import Dict, List, Optional

def get_system_stats(pid_list: Optional[List[int]] = None) -> Dict:
    """
    Get system resource statistics.
    
    Args:
        pid_list: Optional list of PIDs to monitor. If None, returns overall stats.
        
    Returns:
        Dict with system and per-process stats.
    """
    stats = {
        'overall': {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'ram_used_gb': psutil.virtual_memory().used / (1024**3),
            'ram_total_gb': psutil.virtual_memory().total / (1024**3),
            'ram_percent': psutil.virtual_memory().percent,
        },
        'processes': {}
    }
    
    if pid_list:
        for pid in pid_list:
            try:
                proc = psutil.Process(pid)
                stats['processes'][pid] = {
                    'name': proc.name(),
                    'cpu_percent': proc.cpu_percent(interval=0.05),
                    'ram_mb': proc.memory_info().rss / (1024**2),
                    'status': proc.status(),
                    'threads': proc.num_threads(),
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                stats['processes'][pid] = {'status': 'terminated'}
    
    return stats


def get_gpu_stats() -> Optional[Dict]:
    """
    Get GPU stats (Apple Silicon MPS or NVIDIA).
    
    Returns:
        Dict with GPU stats or None if not available.
    """
    # For Apple Silicon, we can't directly get GPU memory usage easily
    # This would require Metal Performance Shaders introspection
    # For now, return a placeholder
    try:
        import torch
        if torch.backends.mps.is_available():
            return {
                'type': 'Apple MPS',
                'available': True,
                'note': 'MPS uses unified memory - see RAM usage'
            }
    except:
        pass
    
    # Try NVIDIA
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            'type': 'NVIDIA',
            'available': True,
            'used_mb': info.used / (1024**2),
            'total_mb': info.total / (1024**2),
            'percent': (info.used / info.total) * 100
        }
    except:
        pass
    
    return None


if __name__ == "__main__":
    # Test
    stats = get_system_stats()
    print(f"Overall CPU: {stats['overall']['cpu_percent']}%")
    print(f"RAM: {stats['overall']['ram_used_gb']:.1f}GB / {stats['overall']['ram_total_gb']:.1f}GB")
    
    gpu = get_gpu_stats()
    if gpu:
        print(f"GPU: {gpu}")
