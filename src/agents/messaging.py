import os
import json
import redis
from typing import Dict, Any

from src import config  # noqa: F401

class MessageBroker:
    """
    Handles pub/sub messaging via Redis to allow asynchronous logging and 
    decoupled monitoring of agent state transitions.
    """
    def __init__(self):
        self.mock_mode = os.getenv("ENABLE_MOCK_DATA", "True").lower() == "true"
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        if not self.mock_mode:
            try:
                self.redis = redis.from_url(self.redis_url)
            except Exception as e:
                print(f"Failed to connect to Redis: {e}. Falling back to mock mode.")
                self.mock_mode = True

    def publish(self, channel: str, message: Dict[str, Any]):
        """
        Publishes a message to the specified Redis channel.
        """
        payload = json.dumps(message)
        if self.mock_mode:
            print(f"[MOCK REDIS - {channel}] -> {payload}")
        else:
            try:
                self.redis.publish(channel, payload)
            except Exception as e:
                print(f"Redis publish error: {e}")

# Singleton instance
broker = MessageBroker()
