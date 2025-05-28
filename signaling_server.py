import asyncio
import websockets
import json

CONNECTIONS = {}

async def handler(websocket):
    peer_id = None
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if "type" in data:
                    if data["type"] == "register":
                        peer_id = data["peer_id"]
                        CONNECTIONS[peer_id] = websocket
                        print(f"Peer registered: {peer_id}")
                    elif data["type"] == "offer" and "peer_id" in data and data["peer_id"] in CONNECTIONS:
                        await CONNECTIONS[data["peer_id"]].send(message)
                    elif data["type"] == "answer" and "peer_id" in data and data["peer_id"] in CONNECTIONS:
                        await CONNECTIONS[data["peer_id"]].send(message)
                    elif data["type"] == "candidate" and "peer_id" in data and data["peer_id"] in CONNECTIONS:
                        await CONNECTIONS[data["peer_id"]].send(message)
            except json.JSONDecodeError:
                print(f"Received non-JSON message: {message}")
            except KeyError as e:
                print(f"Missing key in message: {e}")
    finally:
        if peer_id in CONNECTIONS:
            del CONNECTIONS[peer_id]
            print(f"Peer disconnected: {peer_id}")

async def main():
    async with websockets.serve(handler, "localhost", 8080):
        print("WebSocket signaling server started at ws://localhost:8080")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())