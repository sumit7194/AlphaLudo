"""
Benchmark script for AlphaLudo 11-channel model throughput.
Tests different batch sizes for simulation and replay to find the sweetspot.

Run: td_env/bin/python benchmark_throughput.py
"""
import time
import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import td_ludo_cpp as ludo_cpp
from model import AlphaLudoV3
from trainer import TDTrainer

def benchmark_simulation(batch_size):
    print(f"\nBenchmarking Simulation Batch Size: {batch_size}")
    env = ludo_cpp.VectorGameState(batch_size, True)
    model = AlphaLudoV3().to("cpu") # Test on CPU first as it's common on Mac
    model.eval()
    
    start = time.time()
    total_moves = 0
    num_steps = 100
    
    for _ in range(num_steps):
        # 1. Get state tensor
        states = torch.from_numpy(env.get_state_tensor())
        
        # 2. Forward pass (inference)
        with torch.no_grad():
            policy, value, _ = model(states)
        
        # 3. Get legal moves
        legal_batch = env.get_legal_moves()
        
        # 4. Step
        actions = []
        for i in range(batch_size):
            if legal_batch[i]:
                actions.append(legal_batch[i][0]) # Dummy move
            else:
                actions.append(-1)
        
        env.step(actions)
        total_moves += batch_size
    
    end = time.time()
    duration = end - start
    m_per_sec = total_moves / duration
    print(f"  Moves/sec: {m_per_sec:.2f} | Duration: {duration:.2f}s")
    return m_per_sec

def benchmark_replay(batch_size):
    print(f"\nBenchmarking Replay Batch Size: {batch_size}")
    model = AlphaLudoV3()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Mock data
    states = torch.randn(batch_size, 11, 15, 15)
    next_states = torch.randn(batch_size, 11, 15, 15)
    rewards = torch.randn(batch_size)
    dones = torch.zeros(batch_size)
    weights = torch.ones(batch_size)
    
    start = time.time()
    num_steps = 20
    
    for _ in range(num_steps):
        optimizer.zero_grad()
        
        # Forward V(s)
        _, v_s, _ = model(states)
        
        # Target V(s')
        with torch.no_grad():
            _, v_next, _ = model(next_states)
        
        targets = rewards + 0.99 * v_next.squeeze() * (1 - dones)
        
        # Loss
        loss = (weights * (v_s.squeeze() - targets)**2).mean()
        loss.backward()
        optimizer.step()
    
    end = time.time()
    duration = end - start
    steps_per_sec = num_steps / duration
    samples_per_sec = (num_steps * batch_size) / duration
    print(f"  Steps/sec: {steps_per_sec:.2f} | Samples/sec: {samples_per_sec:.2f}")
    return samples_per_sec

if __name__ == "__main__":
    print("=" * 60)
    print("AlphaLudo 11-Channel Throughput Benchmark")
    print("=" * 60)
    
    # sim_results = {}
    # for bs in [128, 256, 512, 1024]:
    #     sim_results[bs] = benchmark_simulation(bs)
    
    rep_results = {}
    for bs in [512, 1024, 2048, 4096]: # Testing even higher batch sizes
        rep_results[bs] = benchmark_replay(bs)
        
    print("\n" + "=" * 60)
    print("Optimization Summary")
    print("=" * 60)
    
    # best_sim = max(sim_results, key=sim_results.get)
    # print(f"Best Simulation Batch Size: {best_sim} ({sim_results[best_sim]:.2f} moves/s)")
    
    best_rep = max(rep_results, key=rep_results.get)
    print(f"Best Replay Batch Size: {best_rep} ({rep_results[best_rep]:.2f} samples/s)")
