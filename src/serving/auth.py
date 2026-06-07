"""API key authentication for the OCR REST API.

The key is read from the OCR_API_KEY environment variable.
If the variable is not set, the server starts in open/dev mode and
logs a warning — useful for local development without breaking anything.

Mobile clients must send the key in the X-API-Key request header:
    X-API-Key: <your-key>
"""

import logging
import os

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

_API_KEY: str | None = os.environ.get("OCR_API_KEY")

if _API_KEY:
    logger.info("API key authentication enabled.")
else:
    logger.warning(
        "OCR_API_KEY is not set — server running in open/dev mode. "
        "Set OCR_API_KEY in production."
    )


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency — inject into any route that requires authentication."""
    if not _API_KEY:
        return  # dev mode: no key required

    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
        )

    if x_api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
