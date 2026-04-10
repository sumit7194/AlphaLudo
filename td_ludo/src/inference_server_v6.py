"""
GPU Inference Server for V6.1 Training

Actors send (states, masks) → server batches on GPU → returns policy probs.
No transformer context, no CNN caching — simple batch forward pass.
"""
import os
import time
import queue
import numpy as np
import torch
import torch.nn.functional as F


def inference_server_v6_worker(
    device_str, request_queue, response_queues,
    weight_path, weight_version, stop_event,
    in_channels=24, batch_wait_ms=5, max_batch=512,
):
    torch.set_num_threads(1)
    from src.model import AlphaLudoV5

    device = torch.device(device_str)
    model = AlphaLudoV5(num_res_blocks=10, num_channels=128, in_channels=in_channels)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if os.path.exists(weight_path):
        try:
            sd = torch.load(weight_path, map_location='cpu', weights_only=True)
            model.load_state_dict(sd)
        except Exception as e:
            print(f"[InferServer] Warning loading weights: {e}")

    model = model.to(device)
    print(f"[InferServer] Started on {device_str} (in_ch={in_channels})", flush=True)

    local_wv = weight_version.value
    last_weight_check = time.time()
    total_inferences = 0

    while not stop_event.is_set():
        # Collect batch of requests
        batch = []
        try:
            batch.append(request_queue.get(timeout=0.5))
        except queue.Empty:
            continue

        # Drain more requests (up to max_batch, wait batch_wait_ms)
        deadline = time.time() + batch_wait_ms / 1000.0
        while len(batch) < max_batch and time.time() < deadline:
            try:
                batch.append(request_queue.get_nowait())
            except queue.Empty:
                break

        # Unpack: each request is (actor_id, states_np, masks_np, temperature)
        actor_ids = []
        all_states = []
        all_masks = []
        all_temps = []
        sizes = []  # how many samples per actor

        for req in batch:
            aid, states, masks, temp = req
            n = len(states)
            actor_ids.append(aid)
            all_states.append(states)
            all_masks.append(masks)
            all_temps.append(temp)
            sizes.append(n)

        # Stack into single batch
        states_cat = torch.from_numpy(np.concatenate(all_states)).to(device, dtype=torch.float32)
        masks_cat = torch.from_numpy(np.concatenate(all_masks)).to(device, dtype=torch.float32)

        # GPU forward pass
        with torch.no_grad():
            logits = model.forward_policy_only(states_cat, masks_cat)

        logits_cpu = logits.cpu()
        total_inferences += len(states_cat)

        # Split results back to actors
        offset = 0
        for i, (aid, n, temp) in enumerate(zip(actor_ids, sizes, all_temps)):
            actor_logits = logits_cpu[offset:offset+n]

            if temp != 1.0:
                probs = F.softmax(actor_logits / temp, dim=1)
            else:
                probs = F.softmax(actor_logits, dim=1)

            probs = probs.clamp_min(1e-8)
            probs = probs / probs.sum(dim=1, keepdim=True)
            sampled = torch.multinomial(probs, 1).squeeze(1)
            old_lps = torch.log(probs.gather(1, sampled.unsqueeze(1)).squeeze(1) + 1e-8)

            try:
                response_queues[aid].put((sampled.numpy(), old_lps.numpy(), probs.numpy()), timeout=1.0)
            except:
                pass
            offset += n

        # Weight sync
        now = time.time()
        if now - last_weight_check > 3.0:
            cv = weight_version.value
            if cv > local_wv:
                try:
                    sd = torch.load(weight_path, map_location='cpu', weights_only=True)
                    model.load_state_dict(sd)
                    model.to(device)
                    local_wv = cv
                except:
                    pass
            last_weight_check = now

    print(f"[InferServer] Done. Total inferences: {total_inferences:,}", flush=True)
