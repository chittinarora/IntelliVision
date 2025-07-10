# events.py (new file)
from collections import defaultdict
import asyncio

class EventManager:
    def __init__(self):
        self.connections = defaultdict(list)
        self.event_queue = asyncio.Queue()

    async def push_event(self, event_type, data):
        await self.event_queue.put({"type": event_type, "data": data})

    async def broadcast(self):
        while True:
            event = await self.event_queue.get()
            for ws in self.connections[event["type"]]:
                try:
                    await ws.send_json(event)
                except:
                    self.connections[event["type"]].remove(ws)

    def subscribe(self, websocket, event_types):
        for event_type in event_types:
            self.connections[event_type].append(websocket)

event_manager = EventManager()
