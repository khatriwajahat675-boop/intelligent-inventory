"""Small in-process sliding-window limiter for login/expensive endpoints.
For multi-instance deployments enforce the same limit in Nginx (infra/nginx.conf) or Redis."""
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, limit: int, window_s: float = 60.0, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window_s, clock
        self.hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now, q = self.clock(), self.hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True
