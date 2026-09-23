# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# =============================================================================
# Project     : Unbihexium
# Module      : src/unbihexium/serving/security.py
# Title       : Request limits, authentication and rate limiting
# Author      : Olaf Yunus Laitinen Imanov <yunus.z.imanov@helsinki.fi>
# Affiliation : University of Helsinki
# Copyright   : 2025-2026 Unbihexium OSS Foundation and contributors
# Licence     : Mozilla Public License 2.0, see LICENSE.txt
# Python      : CPython 3.10 to 3.14, requires FastAPI and Starlette
# =============================================================================
#
# Abstract
# --------
#   RequestSizeLimitMiddleware   ASGI middleware that answers 413 when the
#   (alias PayloadSizeMiddleware)  declared Content-Length or the bytes
#                                actually received exceed the limit; chunked
#                                uploads without a Content-Length are counted
#                                as they arrive, so the limit cannot be
#                                bypassed
#   validate_content_type        415 for unsupported media types
#   APIKeyAuth                   optional API key in a header, compared in
#                                constant time (hmac.compare_digest) to avoid
#                                timing side channels
#   RateLimiter                  token bucket per client: `rate` requests per
#                                minute with bursts up to `burst`; answers
#                                429 with a Retry-After header
#   get_client_ip                client address; X-Forwarded-For is trusted
#                                only when requested, because clients can
#                                set it freely
#
# References
# ----------
# Fielding, R., Nottingham, M. and Reschke, J. (2022). HTTP Semantics. IETF
# RFC 9110, sections 15.5.14 (413) and 15.5.16 (415).
# Nottingham, M. and Fielding, R. (2012). Additional HTTP Status Codes.
# IETF RFC 6585, section 4 (429 Too Many Requests).
# Turner, J. (1986). New directions in communications (or which way to the
# information age?). IEEE Communications Magazine 24(10), 8-15 (token and
# leaky bucket rate control).
# OWASP Foundation (2023). OWASP API Security Top 10, API4:2023
# Unrestricted Resource Consumption.
# =============================================================================

# Postpone the evaluation of annotations so that modern type syntax works on
# every supported Python version.
from __future__ import annotations

# Constant-time comparison.
import hmac

# JSON error bodies.
import json

# Serialise access to the buckets.
import threading

# Monotonic clock.
import time

# Types of loosely structured values and callables.
from typing import Any, Callable

# HTTP errors and requests of FastAPI.
from fastapi import HTTPException, Request

# Default maximum request body (10 MiB).
MAX_PAYLOAD_SIZE = 10 * 1024 * 1024

# Media types accepted by the service.
ALLOWED_CONTENT_TYPES = {
    "application/json",  # JSON bodies.
    "multipart/form-data",  # File uploads.
}  # End of the media types.


# Error raised while reading a body that grows too large. It is an
# HTTPException so that FastAPI, which wraps other errors of body reading in
# a 400 response, turns it into a 413 response.
class _TooLarge(HTTPException):
    # Error with the limit in the message.
    def __init__(self, limit: int) -> None:
        # 413 Content Too Large.
        super().__init__(status_code=413, detail=f"request body too large: limit is {limit} bytes")


# ASGI middleware that limits the size of request bodies.
class RequestSizeLimitMiddleware:
    # Wrap an ASGI application.
    def __init__(self, app: Any, max_size: int = MAX_PAYLOAD_SIZE) -> None:
        # Inner application.
        self.app = app
        # Limit in bytes.
        self.max_size = int(max_size)

    # Send a JSON 413 response.
    async def _reject(self, send: Callable[..., Any], size: int | None) -> None:
        # Message with the limit.
        detail = f"request body too large: limit is {self.max_size} bytes"
        # Declared size when known.
        if size is not None:
            # Append it.
            detail += f", got {size}"
        # Body bytes.
        body = json.dumps({"detail": detail}).encode("utf-8")
        # Status line and headers.
        headers = [
            (b"content-type", b"application/json"),  # JSON body.
            (b"content-length", str(len(body)).encode("ascii")),  # Body size.
            (b"connection", b"close"),  # Do not read the rest.
        ]  # End of the headers.
        # Response start with 413 Content Too Large.
        await send({"type": "http.response.start", "status": 413, "headers": headers})
        # Body.
        await send({"type": "http.response.body", "body": body})

    # Handle one ASGI connection.
    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        # Only HTTP requests have bodies to limit.
        if scope["type"] != "http":
            # Pass other connections through.
            await self.app(scope, receive, send)
            # Done.
            return
        # Declared body size.
        declared = dict(scope.get("headers") or []).get(b"content-length")
        # Reject early when the declared size is too large or malformed.
        if declared is not None:
            # Parse the header.
            try:
                # Declared bytes.
                size = int(declared)
            # Malformed headers.
            except ValueError:
                # Treat as too large rather than trusting it.
                size = self.max_size + 1
            # Compare with the limit.
            if size > self.max_size:
                # Answer 413 without reading the body.
                await self._reject(send, size)
                # Done.
                return
        # Bytes received so far.
        received = 0
        # Whether the response has started.
        started = False

        # Count the body bytes as they arrive.
        async def limited_receive() -> dict[str, Any]:
            # Running total.
            nonlocal received
            # Next message.
            message = await receive()
            # Body chunks.
            if message["type"] == "http.request":
                # Add the chunk size.
                received += len(message.get("body", b""))
                # Stop reading once the limit is exceeded.
                if received > self.max_size:
                    # Abort the request.
                    raise _TooLarge(self.max_size)
            # Return the message.
            return message

        # Remember whether the application started its response.
        async def tracking_send(message: dict[str, Any]) -> None:
            # Running flag.
            nonlocal started
            # Response starts.
            if message["type"] == "http.response.start":
                # Mark it.
                started = True
            # Forward the message.
            await send(message)

        # Run the application with the counting receive.
        try:
            # Inner application.
            await self.app(scope, limited_receive, tracking_send)
        # Bodies that grew too large.
        except _TooLarge:
            # A response can only be sent if none started.
            if not started:
                # Answer 413.
                await self._reject(send, None)


# Name of earlier releases.
PayloadSizeMiddleware = RequestSizeLimitMiddleware


# Reject unsupported media types with 415.
def validate_content_type(content_type: str | None) -> None:
    # Requests without a body have no type.
    if content_type is None:
        # Nothing to check.
        return
    # Media type without parameters such as charset or boundary.
    base = content_type.split(";")[0].strip().lower()
    # Unsupported types.
    if base not in ALLOWED_CONTENT_TYPES:
        # Explain the problem.
        raise HTTPException(status_code=415, detail=f"unsupported content type: {content_type}")


# Optional API key authentication as a FastAPI dependency.
class APIKeyAuth:
    # Configure the key; None disables authentication.
    def __init__(self, api_key: str | None = None, header_name: str = "X-API-Key") -> None:
        # Expected key.
        self.api_key = api_key
        # Header that carries the key.
        self.header_name = header_name

    # Check the key of a request.
    def __call__(self, request: Request) -> str | None:
        # Open service.
        if self.api_key is None:
            # Nothing to check.
            return None
        # Key sent by the client.
        provided = request.headers.get(self.header_name)
        # Missing keys.
        if provided is None:
            # Unauthenticated.
            raise HTTPException(status_code=401, detail="missing API key")
        # Constant-time comparison of the encoded keys.
        if not hmac.compare_digest(provided.encode("utf-8"), self.api_key.encode("utf-8")):
            # Wrong key.
            raise HTTPException(status_code=403, detail="invalid API key")
        # Return the key.
        return provided


# Client address of a request.
def get_client_ip(request: Request, trust_forwarded: bool = False) -> str:
    # Forwarded addresses only behind a trusted proxy.
    if trust_forwarded:
        # Header set by proxies.
        forwarded = request.headers.get("X-Forwarded-For")
        # The first entry is the original client.
        if forwarded:
            # Return it.
            return forwarded.split(",")[0].strip()
    # Address of the TCP peer.
    return request.client.host if request.client else "unknown"


# Token bucket rate limiter per client.
class RateLimiter:
    # Configure the limit.
    def __init__(
        self,  # The limiter.
        rate_per_minute: float,  # Sustained requests per minute.
        burst: int | None = None,  # Bucket size; default one minute of requests.
        clock: Callable[[], float] = time.monotonic,  # Clock in seconds.
        trust_forwarded: bool = False,  # Use X-Forwarded-For as the client key.
    ) -> None:  # The constructor returns nothing.
        # Positive rates only.
        if rate_per_minute <= 0:
            # Explain the problem.
            raise ValueError(f"rate_per_minute must be positive, got {rate_per_minute}")
        # Tokens added per second.
        self.rate = float(rate_per_minute) / 60.0
        # Bucket capacity.
        self.capacity = float(burst if burst is not None else max(1, int(rate_per_minute)))
        # Clock.
        self.clock = clock
        # Proxy header trust.
        self.trust_forwarded = trust_forwarded
        # Buckets: key -> (tokens, time of the last update).
        self._buckets: dict[str, tuple[float, float]] = {}
        # Lock for concurrent requests.
        self._lock = threading.Lock()

    # Take one token for a key; returns 0 when allowed, else seconds to wait.
    def acquire(self, key: str) -> float:
        # Serialise updates.
        with self._lock:
            # Current time.
            now = self.clock()
            # Stored state, a full bucket for new keys.
            tokens, last = self._buckets.get(key, (self.capacity, now))
            # Refill since the last update, up to the capacity.
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            # A token is available.
            if tokens >= 1.0:
                # Take it.
                self._buckets[key] = (tokens - 1.0, now)
                # Allowed.
                return 0.0
            # Keep the refilled state.
            self._buckets[key] = (tokens, now)
            # Time until one token is available.
            return (1.0 - tokens) / self.rate

    # FastAPI dependency: 429 when the client exceeds the limit.
    def __call__(self, request: Request) -> None:
        # Seconds to wait.
        wait = self.acquire(get_client_ip(request, self.trust_forwarded))
        # Over the limit.
        if wait > 0:
            # Too many requests, with the waiting time rounded up.
            raise HTTPException(
                status_code=429,  # Too Many Requests.
                detail="rate limit exceeded",  # Message.
                headers={"Retry-After": str(int(wait) + 1)},  # Seconds to wait.
            )  # End of the error.


# =============================================================================
# End of module src/unbihexium/serving/security.py
# Part of Unbihexium (https://github.com/unbihexium-oss/unbihexium).
# Cite the project as described in CITATION.cff.
# =============================================================================
