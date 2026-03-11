import json
import torch
import torch.nn.functional as F
import sys
import os

# Add td_ludo to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'td_ludo'))

import src.tensor_utils as tu
from src.model import AlphaLudoV5
import td_ludo_cpp as ludo_cpp

# Load snapshot
with open('snapshot.json', 'r') as f:
    data = json.load(f)

# Reconstruct a mock Python game state
state = ludo_cpp.GameState()
state.current_player = 2
state.active_players = [True, False, True, False]

# Set P0 tokens
for t_data in data['players'][0]['tokens']:
    t_idx = t_data['id']
    if t_data.get('state') == 'BOARD':
        state.player_positions[0][t_idx] = t_data['position']
    elif t_data.get('state') == 'BASE':
        state.player_positions[0][t_idx] = tu.BASE_POS

# Set P2 tokens
for t in range(4):
    state.player_positions[2][t] = tu.BASE_POS

tensor = tu.state_to_tensor_mastery(state)

model = AlphaLudoV5()
model.load_state_dict(torch.load('td_ludo/checkpoints/ac_v5/model_sl.pt', map_location='cpu'))
model.eval()

with torch.no_grad():
    x = torch.from_numpy(tensor).unsqueeze(0)
    policy_logits, _ = model(x)
    probs = F.softmax(policy_logits, dim=1)
    
print("Token Logits:", policy_logits.numpy()[0])
print("Token Probabilities:", probs.numpy()[0])
