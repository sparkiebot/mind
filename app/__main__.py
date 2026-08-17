import uvicorn

from .main import app
from .settings import Settings


if __name__ == "__main__":
    settings = Settings.from_environment()
    uvicorn.run(app, host=settings.bind_host, port=settings.port)
