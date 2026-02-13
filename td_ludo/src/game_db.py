"""
Game History Database for TD-Ludo

SQLite database storing every game result for:
- Historical win rate analysis
- Per-opponent breakdowns
- Dashboard game history tab
- Long-term training analytics
"""

import sqlite3
import time
import os
from collections import defaultdict


class GameDB:
    """
    SQLite wrapper for storing game results.
    
    Schema:
        games (
            id INTEGER PRIMARY KEY,
            timestamp REAL,
            game_num INTEGER,
            p0 TEXT, p1 TEXT, p2 TEXT, p3 TEXT,
            winner INTEGER,
            game_length INTEGER,
            avg_td_error REAL,
            model_player_idx INTEGER
        )
    """
    
    def __init__(self, db_path):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create tables and indices."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                game_num INTEGER,
                p0 TEXT, p1 TEXT, p2 TEXT, p3 TEXT,
                winner INTEGER,
                game_length INTEGER DEFAULT 0,
                avg_td_error REAL DEFAULT 0.0,
                model_player_idx INTEGER DEFAULT 0
            )
        ''')
        
        c.execute('CREATE INDEX IF NOT EXISTS idx_game_num ON games(game_num)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON games(timestamp)')
        
        conn.commit()
        conn.close()
    
    def add_game(self, game_num, identities, winner, game_length=0, 
                 avg_td_error=0.0, model_player_idx=0):
        """
        Record a completed game.
        
        Args:
            game_num: Global game counter
            identities: List of 4 player names
            winner: Winner index (0-3) or -1
            game_length: Number of moves in the game
            avg_td_error: Average TD error during this game
            model_player_idx: Which seat the model was in
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute('''
                INSERT INTO games 
                (timestamp, game_num, p0, p1, p2, p3, winner, game_length, avg_td_error, model_player_idx)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(), game_num,
                identities[0], identities[1], identities[2], identities[3],
                winner, game_length, avg_td_error, model_player_idx
            ))
            conn.commit()
        except Exception as e:
            print(f"[GameDB] Error: {e}")
        finally:
            conn.close()
    
    def get_all_stats(self):
        """
        Get aggregate win/game counts for all models.
        
        Returns:
            dict: { 'Name': {'wins': int, 'games': int, 'win_rate': float}, ... }
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        stats = defaultdict(lambda: {'wins': 0, 'games': 0, 'win_rate': 0.0})
        
        try:
            c.execute('SELECT p0, p1, p2, p3, winner FROM games')
            for p0, p1, p2, p3, winner in c.fetchall():
                players = [p0, p1, p2, p3]
                unique = set(players)
                for name in unique:
                    stats[name]['games'] += 1
                    if winner >= 0 and players[winner] == name:
                        stats[name]['wins'] += 1
            
            # Calculate win rates
            for name in stats:
                g = stats[name]['games']
                if g > 0:
                    stats[name]['win_rate'] = round(stats[name]['wins'] / g * 100, 1)
        except Exception as e:
            print(f"[GameDB] Stats error: {e}")
        finally:
            conn.close()
        
        return dict(stats)
    
    def get_opponent_stats(self, model_name='Main'):
        """
        Get win rates broken down by opponent type.
        
        Returns:
            dict: { 'Heuristic': {'wins': int, 'games': int, 'win_rate': float}, ... }
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        opponent_stats = defaultdict(lambda: {'wins': 0, 'games': 0, 'win_rate': 0.0})
        
        try:
            c.execute('SELECT p0, p1, p2, p3, winner FROM games')
            for p0, p1, p2, p3, winner in c.fetchall():
                players = [p0, p1, p2, p3]
                
                # Find opponents in this game
                model_in_game = model_name in players
                if not model_in_game:
                    continue
                
                opponents = set(p for p in players if p != model_name)
                model_won = winner >= 0 and players[winner] == model_name
                
                for opp in opponents:
                    opponent_stats[opp]['games'] += 1
                    if model_won:
                        opponent_stats[opp]['wins'] += 1
            
            for opp in opponent_stats:
                g = opponent_stats[opp]['games']
                if g > 0:
                    opponent_stats[opp]['win_rate'] = round(
                        opponent_stats[opp]['wins'] / g * 100, 1)
        except Exception as e:
            print(f"[GameDB] Opponent stats error: {e}")
        finally:
            conn.close()
        
        return dict(opponent_stats)
    
    def get_recent_games(self, n=50):
        """Get most recent N games for dashboard history tab."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        games = []
        try:
            c.execute('''
                SELECT game_num, p0, p1, p2, p3, winner, game_length, avg_td_error, timestamp
                FROM games ORDER BY id DESC LIMIT ?
            ''', (n,))
            for row in c.fetchall():
                game_num, p0, p1, p2, p3, winner, length, td_err, ts = row
                players = [p0, p1, p2, p3]
                games.append({
                    'game': game_num,
                    'players': players,
                    'winner': winner,
                    'winner_name': players[winner] if 0 <= winner < 4 else 'None',
                    'length': length,
                    'td_error': round(td_err, 4),
                    'timestamp': ts,
                })
        except Exception as e:
            print(f"[GameDB] Recent games error: {e}")
        finally:
            conn.close()
        
        return games
    
    def get_total_games(self):
        """Get total number of games recorded."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        try:
            c.execute('SELECT COUNT(*) FROM games')
            return c.fetchone()[0]
        except Exception:
            return 0
        finally:
            conn.close()
    
    def to_dict(self):
        """Serialize for JSON API response."""
        return {
            'total_games': self.get_total_games(),
            'all_stats': self.get_all_stats(),
            'opponent_stats': self.get_opponent_stats(),
            'recent_games': self.get_recent_games(30),
        }
