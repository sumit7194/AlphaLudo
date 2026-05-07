"""Tiny side-car server that reports CPU/RAM/threads for a target PID.

Runs on port 8792 (separate from trainer's 8791) so we can attach it without
restarting the trainer. CORS-enabled so the dashboard's JS can fetch it.

Usage:
    python scripts/resource_monitor.py <pid>
    # default port 8792, override with --port

Returns JSON like:
    {"pid": 17115, "alive": true, "cpu_pct": 45.2, "rss_mb": 812.5,
     "vms_mb": 2103.1, "num_threads": 11, "uptime_sec": 1245}
"""
import argparse
import json
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    import psutil
except ImportError:
    raise SystemExit("psutil required. Install: pip install psutil")


def _parse_size_to_mb(s):
    """Convert '20G', '500M', '2048K' → MB float. Returns None if unparseable."""
    if not s:
        return None
    m = re.match(r'^([0-9.]+)([BKMGT]?)\+?$', s.strip())
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2) or 'B'
    return val * {'B': 1/(1024*1024), 'K': 1/1024, 'M': 1.0,
                  'G': 1024, 'T': 1024*1024}[unit]


def get_phys_footprint_mb(pid):
    """Use macOS `top` to get phys_footprint (matches Activity Monitor's
    'Memory' column, includes unified-memory/MPS allocations)."""
    try:
        # -F = no flags-folding (don't truncate columns).
        # Returns one row per pid; MEM is the 8th whitespace-delimited token
        # (PID COMMAND %CPU TIME #TH #WQ #PORTS MEM ...)
        out = subprocess.run(
            ['/usr/bin/top', '-l', '1', '-pid', str(pid), '-F'],
            capture_output=True, text=True, timeout=2,
        ).stdout
        for line in out.splitlines():
            parts = line.split()
            if parts and parts[0] == str(pid):
                return _parse_size_to_mb(parts[7])
        return None
    except Exception:
        return None


class ResourceHandler(BaseHTTPRequestHandler):
    target_pid = None

    def do_GET(self):
        if self.path.startswith('/resources'):
            payload = self._snapshot()
            body = json.dumps(payload).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'use /resources')

    def _snapshot(self):
        try:
            p = psutil.Process(self.target_pid)
            with p.oneshot():
                cpu = p.cpu_percent(interval=0.1)   # blocking 100ms sample
                mem = p.memory_info()
                threads = p.num_threads()
                uptime = time.time() - p.create_time()
            # macOS unified memory: psutil RSS only counts CPU-resident pages.
            # Use top's MEM (= phys_footprint via proc_pid_rusage) to capture
            # IOGPU/MPS allocations that match Activity Monitor's 'Memory' col.
            footprint_mb = get_phys_footprint_mb(self.target_pid)
            return {
                'pid': self.target_pid,
                'alive': True,
                'cpu_pct': round(cpu, 1),
                'rss_mb': round(mem.rss / 1024 / 1024, 1),          # CPU-only resident
                'footprint_mb': round(footprint_mb, 1) if footprint_mb else None,  # incl. MPS/IOGPU
                'vms_mb': round(mem.vms / 1024 / 1024, 1),
                'num_threads': threads,
                'uptime_sec': int(uptime),
            }
        except psutil.NoSuchProcess:
            return {'pid': self.target_pid, 'alive': False}
        except Exception as e:
            return {'pid': self.target_pid, 'alive': False, 'error': str(e)}

    def log_message(self, *args, **kwargs):
        pass  # silence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pid', type=int, help='PID of process to monitor')
    ap.add_argument('--port', type=int, default=8792)
    args = ap.parse_args()
    ResourceHandler.target_pid = args.pid
    server = HTTPServer(('0.0.0.0', args.port), ResourceHandler)
    print(f'[resource_monitor] watching PID {args.pid} on port {args.port}')
    print(f'  http://localhost:{args.port}/resources')
    server.serve_forever()


if __name__ == '__main__':
    main()
