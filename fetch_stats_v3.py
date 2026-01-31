import asyncio
import websockets
import json

async def fetch_history():
    uri = "ws://localhost:8765"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected. Listening for metrics...")
            count = 0
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                count += 1
                if data.get("type") == "metrics_history":
                    print("FOUND METRICS!")
                    print(json.dumps(data["history"], indent=2))
                    return
                if count > 500: # Safety break
                    print("Listened to 500 messages, no metrics history found.")
                    return
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(fetch_history())
