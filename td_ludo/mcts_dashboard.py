import os
import time
import re
from flask import Flask, jsonify, send_from_directory
from collections import deque
import threading

app = Flask(__name__)

LOG_FILE = "nohup.out"

# Global state to store parser data
metrics = {
    'iteration': 0,
    'transitions_gathered': 0,
    'buffer_size': 0,
    'last_step_time': 0.0,
    'current_v_loss': 0.0,
    'current_p_loss': 0.0,
    'v_loss_history': deque(maxlen=200),
    'p_loss_history': deque(maxlen=200),
    'last_logs': deque(maxlen=30),
    'start_time': time.time(),
    'last_file_size': 0
}

def parse_log_line(line):
    """Extract metrics from a log line."""
    stripped_line = line.rstrip()
    if stripped_line:
        metrics['last_logs'].append(stripped_line)
        
    if "Gathered" in line and "transitions" in line:
        match = re.search(r"Gathered (\d+) transitions in ([\d.]+)s", line)
        if match:
            metrics['transitions_gathered'] += int(match.group(1))
            metrics['last_step_time'] = float(match.group(2))
            
    if "Buffer size:" in line:
        match = re.search(r"Buffer size: (\d+)", line)
        if match:
            metrics['buffer_size'] = int(match.group(1))
            
    if "Avg V-Loss:" in line:
        match = re.search(r"Avg V-Loss: ([\d.]+)", line)
        if match:
            loss = float(match.group(1))
            metrics['current_v_loss'] = loss
            metrics['v_loss_history'].append(loss)
            
    if "Avg P-Loss:" in line:
        match = re.search(r"Avg P-Loss: ([\d.]+)", line)
        if match:
            loss = float(match.group(1))
            metrics['current_p_loss'] = loss
            metrics['p_loss_history'].append(loss)
            
    if "--- Iteration" in line:
        match = re.search(r"--- Iteration (\d+) ---", line)
        if match:
            metrics['iteration'] = int(match.group(1))

def background_log_parser():
    """Continuously parse nohup.out without blocking flask."""
    
    # Do initial catch-up reading
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            for line in f:
                parse_log_line(line)
            metrics['last_file_size'] = f.tell()
            
    while True:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                f.seek(metrics['last_file_size'])
                lines = f.readlines()
                for line in lines:
                    parse_log_line(line)
                metrics['last_file_size'] = f.tell()
        time.sleep(1)

@app.route('/')
def serve_dashboard():
    return send_from_directory('static', 'mcts_dashboard.html')

@app.route('/api/metrics')
def get_metrics():
    uptime_sec = time.time() - metrics['start_time']
    throughput = 0
    if uptime_sec > 0:
        throughput = metrics['transitions_gathered'] / (uptime_sec / 60.0)

    # Return deep copies of deques as lists
    return jsonify({
        'iteration': metrics['iteration'],
        'buffer_size': metrics['buffer_size'],
        'last_step_time': metrics['last_step_time'],
        'throughput': throughput,
        'uptime_sec': uptime_sec,
        'current_v_loss': metrics['current_v_loss'],
        'current_p_loss': metrics['current_p_loss'],
        'logs': list(metrics['last_logs']),
        'history': {
            'v_loss': list(metrics['v_loss_history']),
            'p_loss': list(metrics['p_loss_history'])
        }
    })

if __name__ == '__main__':
    # Start log parser thread
    parser_thread = threading.Thread(target=background_log_parser, daemon=True)
    parser_thread.start()
    
    os.makedirs('static', exist_ok=True)
    print("Starting MCTS Dashboard server on http://localhost:5051")
    app.run(host='0.0.0.0', port=5051, debug=False)
