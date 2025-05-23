# session_wrapper.py
from datetime import datetime
import requests

class SessionWrapper:
    def __init__(self, session: requests.Session, token_expiry: datetime):
        self.session = session
        self.token_expiry = token_expiry

    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.token_expiry
