# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Security utilities for API serving.

This module provides security features:
- Request size limits
- Content-type validation
- Optional API key authentication
- Rate limiting helpers
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

# Maximum request payload size (10 MB)
MAX_PAYLOAD_SIZE = 10 * 1024 * 1024

# Allowed content types
ALLOWED_CONTENT_TYPES = {
    "application/json",
    "multipart/form-data",
}


class PayloadSizeMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce payload size limits."""

    def __init__(self, app: Any, max_size: int = MAX_PAYLOAD_SIZE) -> None:
        """Initialize middleware.

        Args:
            app: FastAPI application.
            max_size: Maximum payload size in bytes.
        """
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Process request and check payload size."""
        content_length = request.headers.get("content-length")

        if content_length:
            if int(content_length) > self.max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"Payload too large. Maximum size: {self.max_size} bytes",
                )

        return await call_next(request)


def validate_content_type(content_type: str | None) -> None:
    """Validate request content type.

    Args:
        content_type: Content-Type header value.

    Raises:
        HTTPException: If content type is not allowed.
    """
    if content_type is None:
        return

    # Extract base content type (ignore charset, boundary, etc.)
    base_type = content_type.split(";")[0].strip().lower()

    if base_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {content_type}",
        )


class APIKeyAuth:
    """Optional API key authentication.

    Usage:
        auth = APIKeyAuth(api_key="secret")

        @app.get("/protected")
        async def protected(key: str = Depends(auth)):
            return {"status": "authorized"}
    """

    def __init__(self, api_key: str | None = None, header_name: str = "X-API-Key") -> None:
        """Initialize API key auth.

        Args:
            api_key: Required API key. None disables auth.
            header_name: Header name for API key.
        """
        self.api_key = api_key
        self.header_name = header_name

    def __call__(self, request: Request) -> str | None:
        """Validate API key from request.

        Args:
            request: FastAPI request.

        Returns:
            API key if valid.

        Raises:
            HTTPException: If API key is invalid.
        """
        if self.api_key is None:
            return None

        provided_key = request.headers.get(self.header_name)

        if provided_key is None:
            raise HTTPException(
                status_code=401,
                detail="Missing API key",
            )

        if provided_key != self.api_key:
            raise HTTPException(
                status_code=403,
                detail="Invalid API key",
            )

        return provided_key


def get_client_ip(request: Request) -> str:
    """Get client IP address from request.

    Handles X-Forwarded-For header for proxied requests.

    Args:
        request: FastAPI request.

    Returns:
        Client IP address.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
