import logging
import re

import uvicorn

from .main import app
from .settings import Settings


class HealthRequestFilter(logging.Filter):
    """Suppress noisy Uvicorn access logs for health probes only."""

    _health_request = re.compile(r'"GET /health(?:[?\s])')

    def filter(self, record: logging.LogRecord) -> bool:
        return self._health_request.search(record.getMessage()) is None


def configure_access_logging() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(log_filter, HealthRequestFilter) for log_filter in access_logger.filters):
        access_logger.addFilter(HealthRequestFilter())


if __name__ == "__main__":
    settings = Settings.from_environment()
    configure_access_logging()
    uvicorn.run(app, host=settings.bind_host, port=settings.port)
