
import os
import sys
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model_v3 import AlphaLudoV3

class MobileAlphaLudo(torch.nn.Module):
    """
    Stripped down wrapper for mobile.
    """
    def __init__(self, original_model):
        super(MobileAlphaLudo, self).__init__()
        self.backbone = original_model
        
    def forward(self, x, legal_mask):
        # Use the inference method from original model
        return self.backbone.forward_policy_value(x, legal_mask)

def convert():
    print("🚀 Starting PyTorch Mobile Conversion (TorchScript)...")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(script_dir, '../../experiments/kickstart/model_kickstart.pt')
    ptl_path = os.path.join(script_dir, 'model.ptl')
    
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
    
    # 2. Input Example
    example_input = (
        torch.randn(1, 21, 15, 15),
        torch.ones(1, 4)
    )
    
    # 3. Trace (JIT)
    print("🔄 Tracing model...")
    traced_script_module = torch.jit.trace(mobile_model, example_input)
    
    # 4. Optimize for Mobile
    print("⚡ Optimizing for mobile...")
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    
    # 5. Save
    traced_script_module_optimized._save_for_lite_interpreter(ptl_path)
        
    print(f"🎉 Success! PyTorch Lite model saved to: {ptl_path}")
    print(f"   Size: {os.path.getsize(ptl_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    convert()
