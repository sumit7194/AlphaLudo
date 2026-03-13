import os
import time
import re
import curses
from collections import deque

LOG_FILE = "nohup.out"
CHECKPOINT_DIR = "checkpoints/mcts_v1"

def parse_log_line(line, metrics):
    """Extract metrics from a log line."""
    if "Gathered" in line and "transitions" in line:
        # Expected: Gathered 64 transitions in 2.5s. Buffer size: 1000
        match = re.search(r"Gathered (\d+) transitions in ([\d.]+)s", line)
        if match:
            metrics['transitions_gathered'] += int(match.group(1))
            metrics['last_step_time'] = float(match.group(2))
            
    elif "Buffer size:" in line:
        match = re.search(r"Buffer size: (\d+)", line)
        if match:
            metrics['buffer_size'] = int(match.group(1))
            
    elif "Avg V-Loss:" in line:
        match = re.search(r"Avg V-Loss: ([\d.]+)", line)
        if match:
            loss = float(match.group(1))
            metrics['current_v_loss'] = loss
            metrics['v_loss_history'].append(loss)
            
    elif "Avg P-Loss:" in line:
        match = re.search(r"Avg P-Loss: ([\d.]+)", line)
        if match:
            loss = float(match.group(1))
            metrics['current_p_loss'] = loss
            metrics['p_loss_history'].append(loss)
            
    elif "--- Iteration" in line:
        match = re.search(r"--- Iteration (\d+) ---", line)
        if match:
            metrics['iteration'] = int(match.group(1))

def draw_dashboard(stdscr):
    # Setup curses
    curses.curs_set(0)
    stdscr.nodelay(1)
    
    # Init colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    
    metrics = {
        'iteration': 0,
        'transitions_gathered': 0,
        'buffer_size': 0,
        'last_step_time': 0.0,
        'current_v_loss': 0.0,
        'current_p_loss': 0.0,
        'v_loss_history': deque(maxlen=50),
        'p_loss_history': deque(maxlen=50),
        'last_file_size': 0,
        'start_time': time.time()
    }
    
    if not os.path.exists(LOG_FILE):
        stdscr.addstr(0, 0, f"Waiting for {LOG_FILE} to be created... (Did you start training with nohup?)", curses.color_pair(3))
        stdscr.refresh()
        while not os.path.exists(LOG_FILE):
            time.sleep(1)
            if stdscr.getch() == ord('q'): return

    # Seek to end of log initially to only show live updates, 
    # OR read from beginning to get history. Let's read from beginning for history.
    with open(LOG_FILE, 'r') as f:
        for line in f:
            parse_log_line(line, metrics)
        metrics['last_file_size'] = f.tell()

    while True:
        # Check for quit
        if stdscr.getch() == ord('q'):
            break
            
        # Read new lines from log
        with open(LOG_FILE, 'r') as f:
            f.seek(metrics['last_file_size'])
            lines = f.readlines()
            for line in lines:
                parse_log_line(line, metrics)
            metrics['last_file_size'] = f.tell()
            
        # Calculate uptime
        uptime = int(time.time() - metrics['start_time'])
        hrs, remainder = divmod(uptime, 3600)
        mins, secs = divmod(remainder, 60)
        
        # Calculate speed (transitions per min)
        if mins > 0 or hrs > 0:
            speed = metrics['transitions_gathered'] / (uptime / 60)
        else:
            speed = 0

        # Draw UI
        stdscr.clear()
        
        max_y, max_x = stdscr.getmaxyx()
        
        # Header
        header = f"🚀 ALPHA-LUDO MCTS DASHBOARD 🚀"
        stdscr.addstr(0, (max_x - len(header)) // 2, header, curses.color_pair(2) | curses.A_BOLD)
        stdscr.addstr(1, 0, "-" * max_x)
        
        # Status
        stdscr.addstr(3, 2, f"Uptime         : {hrs:02d}h {mins:02d}m {secs:02d}s", curses.color_pair(3))
        stdscr.addstr(4, 2, f"Iteration      : {metrics['iteration']}", curses.color_pair(1) | curses.A_BOLD)
        stdscr.addstr(5, 2, f"Buffer Size    : {metrics['buffer_size']} / 100000")
        stdscr.addstr(6, 2, f"Sim Speed      : {metrics['last_step_time']:.1f}s / step")
        stdscr.addstr(7, 2, f"Throughput     : {speed:.0f} transitions / min")
        
        # Loss Metrics
        stdscr.addstr(9,  2, f"Value Loss (MSE): {metrics['current_v_loss']:.4f}", curses.color_pair(3))
        stdscr.addstr(10, 2, f"Policy Loss (CE): {metrics['current_p_loss']:.4f}", curses.color_pair(3))
        
        # Draw mini sparklines if we have history
        if len(metrics['v_loss_history']) > 1:
            draw_sparkline(stdscr, 12, 2, "V-Loss Trend", list(metrics['v_loss_history']))
            
        if len(metrics['p_loss_history']) > 1:
            draw_sparkline(stdscr, 15, 2, "P-Loss Trend", list(metrics['p_loss_history']))
            
        # Footer
        stdscr.addstr(max_y-2, 0, "-" * max_x)
        stdscr.addstr(max_y-1, 2, "Press 'Q' to quit monitor. (Training will continue in background)", curses.color_pair(2))
        
        stdscr.refresh()
        time.sleep(1)

def draw_sparkline(stdscr, y, x, title, data):
    bars = " ▂▃▄▅▆▇█"
    stdscr.addstr(y, x, f"{title}: ")
    
    if max(data) == min(data):
        sparkline = bars[0] * len(data)
    else:
        # Scale data between 0 and 7
        spread = max(data) - min(data)
        sparkline = "".join([bars[int((val - min(data)) / spread * 7)] for val in data])
        
    stdscr.addstr(y, x + len(title) + 2, sparkline, curses.color_pair(1))

if __name__ == "__main__":
    try:
        curses.wrapper(draw_dashboard)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Dashboard error: {e}")
