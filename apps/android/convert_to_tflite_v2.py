
import os
import sys
import torch
import ai_edge_torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model_v3 import AlphaLudoV3

class MobileAlphaLudo(torch.nn.Module):
    """
    Stripped down and wrapped version of AlphaLudoV3 for mobile inference.
    """
    def __init__(self, original_model):
        super(MobileAlphaLudo, self).__init__()
        self.backbone = original_model
        
    def forward(self, x, legal_mask):
        # Use the inference method from original model
        return self.backbone.forward_policy_value(x, legal_mask)

def convert():
    print("🚀 Starting AI-Edge-Torch Conversion (PyTorch -> TFLite)...")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(script_dir, '../../experiments/kickstart/model_kickstart.pt')
    tflite_path = os.path.join(script_dir, 'model.tflite')
    
    # 1. Load PyTorch Model
    print(f"📥 Loading checkpoint: {ckpt_path}")
    device = torch.device('cpu')
    model = AlphaLudoV3(num_res_blocks=10, num_channels=128)
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    # Wrap
    mobile_model = MobileAlphaLudo(model)
    mobile_model.eval()
    
    # 2. Prepare Sample Inputs (Tuple of tensors as expected by forward)
    sample_input = (
        torch.randn(1, 21, 15, 15), # x
        torch.ones(1, 4)            # legal_mask
    )
    
    # 3. Convert using ai-edge-torch (Direct conversion)
    print("🔄 Converting directly to TFLite...")
    edge_model = ai_edge_torch.convert(mobile_model, sample_input)
    
    # 4. Save
    edge_model.export(tflite_path)
        
    print(f"🎉 Success! TFLite model saved to: {tflite_path}")
    print(f"   Size: {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    convert()
