import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from td_ludo.src.model import AlphaLudoV5
from torch.utils.mobile_optimizer import optimize_for_mobile

def export_for_android(weights_path, output_path):
    print(f"Loading weights from {weights_path}...")
    
    # 1. Initialize model
    model = AlphaLudoV5()
    
    # Load weights (handle possible checkpoint dict wrapper)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print("Model loaded successfully.")

    # 2. Trace the model
    print("Tracing model...")
    example_input = torch.zeros(1, 11, 15, 15)
    
    # Trace with just the state. The model will output raw probabilities for all 4 tokens.
    # The Android Kotlin code already manually screens out probabilities for illegal moves.
    # This prevents PyTorch Mobile from crashing on unsupported bitwise boolean ops.
    traced_model = torch.jit.trace(model, (example_input,), strict=False)
    
    # 3. Optimize for mobile
    print("Optimizing for mobile...")
    optimized_model = optimize_for_mobile(traced_model)
    
    # 4. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    optimized_model._save_for_lite_interpreter(output_path)
    print(f"Success! Android-ready model saved to: {output_path}")
    
    # Also save standard torchscript as backup
    ts_path = output_path.replace('.ptl', '_standard.pt')
    traced_model.save(ts_path)
    print(f"Standard TorchScript backup saved to: {ts_path}")

if __name__ == "__main__":
    weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), "td_ludo/checkpoints/ac_v5/model_latest.pt")
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apps/android/model.ptl")
    export_for_android(weights, output)
