
import os
import sys
import torch
import glob
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

def convert_model(ckpt_path, output_path):
    print(f"\n🔄 Processing: {os.path.basename(ckpt_path)}...")
    
    # 1. Load PyTorch Model
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
    traced_script_module = torch.jit.trace(mobile_model, example_input)
    
    # 4. Optimize for Mobile
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    
    # 5. Save
    traced_script_module_optimized._save_for_lite_interpreter(output_path)
    
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"✅ Saved: {os.path.basename(output_path)} ({size_mb:.2f} MB)")

def main():
    print("🚀 Starting Batch PyTorch Mobile Conversion...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, '../../experiments/kickstart/models_to_test'))
    output_dir = os.path.join(script_dir, 'models_mobile')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .pt files
    pt_files = glob.glob(os.path.join(models_dir, "*.pt"))
    if not pt_files:
        print(f"❌ No models found in {models_dir}")
        return

    for pt_file in pt_files:
        name = os.path.basename(pt_file).replace(".pt", ".ptl")
        out_path = os.path.join(output_dir, name)
        convert_model(pt_file, out_path)
        
    print(f"\n🎉 All Done! Mobile models saved in: {output_dir}")

if __name__ == "__main__":
    main()
