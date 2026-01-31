import asyncio
import websockets
import json

async def fetch_history():
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri) as websocket:
            # Wait for the first message, which should be or contain metrics_history
            # Actually, the visualizer sends multiple messages on connect.
            # 1. current_state
            # 2. metrics_history
            # 3. dice_stats
            # We need to listen until we get 'metrics_history'.
            
            for _ in range(10): # Try 10 messages
                message = await websocket.recv()
                data = json.loads(message)
                if data.get("type") == "metrics_history":
                    print(json.dumps(data["history"], indent=2))
                    return
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(fetch_history())
