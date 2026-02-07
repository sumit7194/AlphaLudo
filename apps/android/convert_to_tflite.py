
import os
import sys
import torch
import torch.nn as nn
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model_v3 import AlphaLudoV3

class MobileAlphaLudo(nn.Module):
    """
    Stripped down version of AlphaLudoV3 for mobile inference.
    - Removes auxiliary heads
    - Removes batch norm layers in inference mode (fused usually, but good to be explicit if needed)
    - Outputs only Policy and Value
    """
    def __init__(self, original_model):
        super(MobileAlphaLudo, self).__init__()
        self.backbone = original_model
        
    def forward(self, x, legal_mask):
        # Use the inference method from original model
        # It skips aux head
        return self.backbone.forward_policy_value(x, legal_mask)

def convert():
    print("🚀 Starting TFLite Conversion...")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(script_dir, '../../experiments/kickstart/model_kickstart.pt')
    onnx_path = os.path.join(script_dir, 'model.onnx')
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
    
    # Wrap for export (stripping aux head logic if needed, though forward_policy_value handles it)
    mobile_model = MobileAlphaLudo(model)
    mobile_model.eval()
    
    # 2. Trace/Export to ONNX
    print("🔄 Converting to ONNX...")
    dummy_input = torch.randn(1, 21, 15, 15)
    dummy_mask = torch.ones(1, 4) # All moves legal
    
    torch.onnx.export(
        mobile_model,
        (dummy_input, dummy_mask),
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input', 'legal_mask'],
        output_names=['policy', 'value'],
        dynamic_axes={'input': {0: 'batch_size'}, 'legal_mask': {0: 'batch_size'}, 'policy': {0: 'batch_size'}, 'value': {0: 'batch_size'}}
    )
    print(f"✅ ONNX exported to {onnx_path}")
    
    # 3. Convert ONNX to TensorFlow (Intermediate)
    print("🔄 Converting ONNX to TensorFlow Graph...")
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_path = os.path.join(script_dir, 'model_tf')
    tf_rep.export_graph(tf_path)
    print(f"✅ TensorFlow Graph exported to {tf_path}")
    
    # 4. Convert TF to TFLite
    print("🔄 Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    
    # Optimize for latency/size
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # Enable TensorFlow ops.
    ]
    
    tflite_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"🎉 Success! TFLite model saved to: {tflite_path}")
    print(f"   Size: {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    convert()
