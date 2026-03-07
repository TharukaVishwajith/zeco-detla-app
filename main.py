import os

import uvicorn

from app.main import app


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("UVICORN_HOST", "0.0.0.0"),
        port=int(os.getenv("UVICORN_PORT", "8000")),
        reload=os.getenv("UVICORN_RELOAD", "false").lower() == "true",
    )
