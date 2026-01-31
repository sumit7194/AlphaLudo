from src.game_db import GameDB
import json

db = GameDB("training_history.db")
stats = db.get_all_stats()
print("DB Stats:")
print(json.dumps(stats, indent=2))
