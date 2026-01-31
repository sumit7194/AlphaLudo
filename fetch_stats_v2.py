import asyncio
import websockets
import json

async def fetch_history():
    uri = "ws://localhost:8765"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected.")
            for i in range(5):
                message = await websocket.recv()
                data = json.loads(message)
                print(f"Msg {i}: Type={data.get('type')}")
                if data.get("type") == "metrics_history":
                    print("FOUND METRICS!")
                    print(json.dumps(data["history"], indent=2))
                    return
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(fetch_history())
