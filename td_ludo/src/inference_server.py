"""
Centralized GPU Inference Server for V9 Multi-Process Training

Runs as a separate process, owns the model on MPS GPU.
Actors send inference requests via queue, server batches them
and returns results via per-actor response queues.

This moves all neural network computation off CPU actors onto the GPU,
achieving ~5-6x speedup for CNN backbone inference.
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F


def inference_server_worker(
    context_length,
    device_str,
    request_queue,
    response_queues,
    weight_update_queue,
    stop_event,
    initial_weight_path,
):
    """
    Inference server process entry point.

    Batches inference requests from multiple actors and runs them on GPU.
    Handles both CNN feature extraction and full policy inference in one round-trip.
    """
    torch.set_num_threads(1)

    from src.model_v9 import AlphaLudoV9

    device = torch.device(device_str)

    # Create model on GPU
    model = AlphaLudoV9(context_length=context_length)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # Load initial weights
    if initial_weight_path and os.path.exists(initial_weight_path):
        try:
            state_dict = torch.load(initial_weight_path, map_location='cpu', weights_only=True)
            state_dict = _adapt_state_dict_context(state_dict, context_length)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"[InferenceServer] Warning: could not load weights: {e}")

    model = model.to(device)
    print(f"[InferenceServer] Started on {device_str}")

    # Stats
    total_requests = 0
    total_batches = 0
    last_stats_time = time.time()

    # Batch collection settings
    max_wait_ms = 15  # max time to wait for more requests before processing
    max_batch = 1024  # max samples per batch

    while not stop_event.is_set():
        # Check for weight updates (non-blocking)
        _check_weight_updates(model, weight_update_queue, context_length, device)

        # Collect a batch of requests
        batch = _collect_batch(request_queue, max_wait_ms, max_batch, stop_event)
        if not batch:
            continue

        # Process the batch on GPU
        results = _process_batch(model, batch, device, context_length)

        # Route results back to actors
        for req, result in zip(batch, results):
            actor_id = req['actor_id']
            try:
                response_queues[actor_id].put(result, timeout=1.0)
            except Exception:
                pass

        total_requests += len(batch)
        total_batches += 1

        # Log stats periodically
        now = time.time()
        if now - last_stats_time > 30.0:
            elapsed = now - last_stats_time
            rps = total_requests / elapsed if elapsed > 0 else 0
            print(f"[InferenceServer] {rps:.0f} requests/sec, "
                  f"{total_batches} batches, avg batch={total_requests/max(1,total_batches):.1f}")
            total_requests = 0
            total_batches = 0
            last_stats_time = now

    print("[InferenceServer] Shutting down")


def _adapt_state_dict_context(state_dict, model_K):
    """Slice positional embeddings and causal mask if checkpoint has different K."""
    if 'temporal_pos_embed.weight' in state_dict:
        ckpt_K = state_dict['temporal_pos_embed.weight'].shape[0]
        if ckpt_K != model_K:
            state_dict['temporal_pos_embed.weight'] = state_dict['temporal_pos_embed.weight'][:model_K]
    if 'causal_mask' in state_dict:
        ckpt_K = state_dict['causal_mask'].shape[0]
        if ckpt_K != model_K:
            state_dict['causal_mask'] = state_dict['causal_mask'][:model_K, :model_K]
    return state_dict


def _check_weight_updates(model, weight_update_queue, context_length, device):
    """Check for and apply weight updates from the learner."""
    try:
        while not weight_update_queue.empty():
            weight_path = weight_update_queue.get_nowait()
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
                state_dict = _adapt_state_dict_context(state_dict, context_length)
                model.load_state_dict(state_dict)
                model.to(device)
    except Exception:
        pass


def _collect_batch(request_queue, max_wait_ms, max_batch, stop_event):
    """Collect requests from multiple actors into a batch."""
    batch = []
    deadline = time.perf_counter() + max_wait_ms / 1000.0

    # Block on first request (up to 100ms to allow shutdown checks)
    try:
        req = request_queue.get(timeout=0.1)
        batch.append(req)
    except Exception:
        return batch

    # Collect more requests until deadline or max batch
    while len(batch) < max_batch and time.perf_counter() < deadline:
        try:
            req = request_queue.get_nowait()
            batch.append(req)
        except Exception:
            break

    return batch


def _process_batch(model, batch, device, context_length):
    """
    Process a batch of inference requests on GPU.

    Each request contains:
        - new_grids: (N_i, 14, 15, 15) new board states needing CNN
        - cached_features: (N_i, K-1, 80) previously cached CNN features
        - prev_actions: (N_i, K) action history
        - seq_mask: (N_i, K) padding mask
        - legal_mask: (N_i, 4) legal move mask
        - temperature: float
    """
    # Concatenate all requests into one big batch
    all_new_grids = []
    all_cached = []
    all_acts = []
    all_mask = []
    all_legal = []
    all_temps = []
    sizes = []

    for req in batch:
        n = req['new_grids'].shape[0]
        sizes.append(n)
        all_new_grids.append(req['new_grids'])
        all_cached.append(req['cached_features'])
        all_acts.append(req['prev_actions'])
        all_mask.append(req['seq_mask'])
        all_legal.append(req['legal_mask'])
        all_temps.append(req['temperature'])

    # Stack into single tensors
    grids_np = np.concatenate(all_new_grids, axis=0)
    cached_np = np.concatenate(all_cached, axis=0)
    acts_np = np.concatenate(all_acts, axis=0)
    mask_np = np.concatenate(all_mask, axis=0)
    legal_np = np.concatenate(all_legal, axis=0)

    total_n = grids_np.shape[0]
    K = context_length

    with torch.no_grad():
        # Move to GPU
        grids_t = torch.from_numpy(grids_np).float().to(device)
        cached_t = torch.from_numpy(cached_np).float().to(device)
        acts_t = torch.from_numpy(acts_np).to(device)
        mask_t = torch.from_numpy(mask_np).to(device)
        legal_t = torch.from_numpy(legal_np).float().to(device)

        # CNN on new grids
        new_cnn = model.compute_single_cnn_features(grids_t)  # (total_n, 80)

        # Insert new CNN features as last column of cached sequence
        # cached_t is (total_n, K-1, 80), new_cnn is (total_n, 80)
        full_cached = torch.cat([cached_t, new_cnn.unsqueeze(1)], dim=1)  # (total_n, K, 80)

        # Transformer + policy forward
        policy_logits = model.forward_policy_only_cached(
            full_cached, acts_t, mask_t, legal_t
        )

        # Per-request temperature and sampling
        results = []
        offset = 0
        for i, n in enumerate(sizes):
            logits_i = policy_logits[offset:offset + n]
            legal_i = legal_t[offset:offset + n]
            temp = all_temps[i]

            if temp != 1.0:
                probs = F.softmax(logits_i / temp, dim=1)
            else:
                probs = F.softmax(logits_i, dim=1)

            # Fix NaN/inf
            bad = torch.isnan(probs).any(dim=1) | (probs.sum(dim=1) < 0.99)
            if bad.any():
                probs[bad] = legal_i[bad] / legal_i[bad].sum(dim=1, keepdim=True).clamp_min(1e-8)
            probs = probs.clamp_min(1e-8)
            probs = probs / probs.sum(dim=1, keepdim=True)

            sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
            old_lps = torch.log(
                probs.gather(1, sampled.unsqueeze(1)).squeeze(1) + 1e-8
            )

            cnn_feats_i = new_cnn[offset:offset + n]

            results.append({
                'sampled_actions': sampled.cpu().numpy(),
                'old_log_probs': old_lps.cpu().numpy(),
                'new_cnn_features': cnn_feats_i.cpu().numpy(),
            })
            offset += n

    return results
