import asyncio
import websockets
import http
import os
import mimetypes

# Configuration
PROXY_PORT = 8090
TARGET_WS_URL = "ws://localhost:8765"
ROOT_DIR = os.getcwd()

# Dynamic WebSocket URL (replaces hardcoded localhost)
DYNAMIC_WS_JS = "const socket = new WebSocket((window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host);"

async def ws_proxy_handler(client_ws):
    """Proxies messages between Client (Ngrok/Phone) and Target (Visualizer)."""
    print(f"[Proxy] New WebSocket connection from {client_ws.remote_address}")
    try:
        async with websockets.connect(TARGET_WS_URL) as target_ws:
            
            async def forward(source, dest):
                async for msg in source:
                    await dest.send(msg)
            
            # Run both directions
            await asyncio.gather(
                forward(client_ws, target_ws),
                forward(target_ws, client_ws)
            )
    except Exception as e:
        print(f"[Proxy] Connection closed: {e}")

from websockets.asyncio.server import Response
from websockets.http import Headers

async def http_handler(connection, request):
    """Handles HTTP requests (Static Files + Injection)."""
    # 1. Check for WebSocket Upgrade
    # If connection is upgrading, return None to let websockets library handle it.
    if "Upgrade" in request.headers and request.headers["Upgrade"].lower() == "websocket":
        return None

    # websockets 10+ passes (connection, request)
    path = request.path.split("?")[0]
    if path == "/":
        path = "/index.html"
    
    # Security: Prevent directory traversal
    clean_path = path.lstrip("/")
    full_path = os.path.join(ROOT_DIR, clean_path)
    if not os.path.abspath(full_path).startswith(ROOT_DIR):
        return Response(403, "Forbidden", Headers(), b"Forbidden")

    if not os.path.exists(full_path) or not os.path.isfile(full_path):
         return Response(404, "Not Found", Headers(), b"Not Found")

    # Read File
    try:
        mode = 'rb'
        content_type, _ = mimetypes.guess_type(full_path)
        if content_type is None:
            content_type = "application/octet-stream"

        # Injection for HTML files
        if path.endswith(".html"):
             with open(full_path, 'r', encoding='utf-8') as f:
                 content = f.read()
             
             # INJECT: Replace localhost:8765 with dynamic host if found
             target_str = "const socket = new WebSocket('ws://localhost:8765');"
             if target_str in content:
                 content = content.replace(target_str, DYNAMIC_WS_JS)
                 print(f"[Proxy] Injected Dynamic WS URL into {path}")
             
             body = content.encode('utf-8')
        else:
             with open(full_path, 'rb') as f:
                 body = f.read()
        
        headers = Headers([
            ("Content-Type", content_type),
            ("Content-Length", str(len(body)))
        ])
        return Response(200, "OK", headers, body)

    except Exception as e:
        print(f"[Proxy] Error serving {path}: {e}")
        return Response(500, "Internal Server Error", Headers(), b"Error")


async def main():
    print(f"[Proxy] Starting Remote Access Server on port {PROXY_PORT}...")
    print(f"[Proxy] - HTTP: Serving {ROOT_DIR}")
    print(f"[Proxy] - WS:   Proxying to {TARGET_WS_URL}")
    
    # Websockets serve() handles WS upgrades automatically.
    # process_request handles HTTP. If it returns response, WS handshake is skipped.
    # If it returns None, WS handshake proceeds -> ws_proxy_handler called.
    async with websockets.serve(ws_proxy_handler, "0.0.0.0", PROXY_PORT, process_request=http_handler):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Proxy] Stopped.")
