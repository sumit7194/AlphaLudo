import sqlite3
import time
import os
from collections import defaultdict

class GameDB:
    """
    SQLite wrapper for storing AlphaLudo game results.
    Schema:
        games (
            id INTEGER PRIMARY KEY,
            timestamp REAL,
            p0 TEXT,
            p1 TEXT,
            p2 TEXT,
            p3 TEXT,
            winner INTEGER
        )
    """
    
    def __init__(self, db_path="game_history.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                p0 TEXT,
                p1 TEXT,
                p2 TEXT,
                p3 TEXT,
                winner INTEGER
            )
        ''')
        
        # Indices for faster lookup by player name
        c.execute('CREATE INDEX IF NOT EXISTS idx_p0 ON games(p0)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_p1 ON games(p1)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_p2 ON games(p2)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_p3 ON games(p3)')
        
        conn.commit()
        conn.close()
        
    def add_game(self, identities, winner):
        """
        Record a completed game.
        
        Args:
            identities: List of 4 strings (names of models at p0-p3)
            winner: Index of winning player (0-3), or -1 if invalid
        """
        if len(identities) != 4:
            print(f"DB Error: Expected 4 identities, got {len(identities)}")
            return
            
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            c.execute('''
                INSERT INTO games (timestamp, p0, p1, p2, p3, winner)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (time.time(), identities[0], identities[1], identities[2], identities[3], winner))
            conn.commit()
        except Exception as e:
            print(f"DB Error adding game: {e}")
        finally:
            conn.close()
            
    def get_all_stats(self):
        """
        Get aggregate win/game counts for ALL models found in the DB.
        
        Returns:
            dict: { 'ModelName': {'wins': int, 'games': int}, ... }
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        stats = defaultdict(lambda: {'wins': 0, 'games': 0})
        
        try:
            # We need to scan games. 
            # Ideally we'd use a smarter query, but given Ludo has 4 players, 
            # simpler to fetch relevant columns and aggregate in Python for flexibility 
            # unless data is huge. For <1M games, this is fine.
            # OPTIMIZATION: Use SQL grouping if performance becomes an issue.
            
            c.execute('SELECT p0, p1, p2, p3, winner FROM games')
            rows = c.fetchall()
            
            for r in rows:
                p0, p1, p2, p3, winner = r
                players = [p0, p1, p2, p3]
                
                # Update games count (unique players in this game)
                # If a model plays against itself (Main vs Main), does it count as 1 game or 2?
                # Usually standard winrate is per-match participation.
                # If 'Main' is at P0 and P1, it played 1 game (as a team? no, distinct agents).
                # But for "Model Winrate", we usually mean "Instance Winrate".
                # If Main wins against Main, winrate is 50%?
                # Let's count per-slot appearance for now, as that's how we did it in session stats.
                # Actually, session stats logic was: "unique_models = set(identities) ... session_stats[name]['games'] += 1".
                # So if Main plays 3 slots, it counts as 1 game played.
                
                unique_models = set(players)
                for model in unique_models:
                    if model == 'Main': 
                        # We typically track global Main stats elsewhere, 
                        # but let's track it here too for consistency if needed.
                        pass
                        
                    stats[model]['games'] += 1
                    
                    if winner != -1 and players[winner] == model:
                        stats[model]['wins'] += 1
                        
        except Exception as e:
            print(f"DB Error fetching stats: {e}")
        finally:
            conn.close()
            
        return dict(stats)

    def get_model_stats(self, model_name):
        """Get stats for a specific model."""
        all_stats = self.get_all_stats() # Naive re-use
        return all_stats.get(model_name, {'wins': 0, 'games': 0})
