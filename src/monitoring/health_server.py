"""
Async health and readiness endpoints for Hunter.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Dict, Optional

from aiohttp import web

from src.monitoring.metrics import track_health_check


ReadinessChecker = Callable[[], Awaitable[Dict[str, object]]]


class HealthServer:
    """
    Lightweight aiohttp server exposing /live and /ready endpoints.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8081,
        readiness_check: Optional[ReadinessChecker] = None,
    ):
        self._host = host
        self._port = port
        self._readiness_check = readiness_check
        self._app = web.Application()
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._start_time = time.time()

        self._app.add_routes(
            [
                web.get("/live", self._handle_live),
                web.get("/ready", self._handle_ready),
            ]
        )

    async def _handle_live(self, _: web.Request) -> web.Response:
        uptime = time.time() - self._start_time
        track_health_check("live", ready=True)
        return web.json_response({"status": "ok", "uptime_seconds": uptime})

    async def _handle_ready(self, _: web.Request) -> web.Response:
        ready_payload = {
            "status": "ok",
            "ready": True,
        }

        if self._readiness_check:
            try:
                readiness_result = await self._readiness_check()
                ready_payload.update(readiness_result)
                ready = bool(readiness_result.get("ready", True))
            except Exception as exc:  # pragma: no cover - defensive
                ready_payload = {"status": "error", "ready": False, "error": str(exc)}
                ready = False
        else:
            ready = True

        status_code = 200 if ready else 503
        track_health_check("ready" if ready else "not_ready", ready=ready)
        return web.json_response(ready_payload, status=status_code)

    async def start(self) -> None:
        if self._runner:
            return

        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

    async def stop(self) -> None:
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None


async def start_health_server(
    host: str = "0.0.0.0",
    port: int = 8081,
    readiness_check: Optional[ReadinessChecker] = None,
) -> HealthServer:
    server = HealthServer(host=host, port=port, readiness_check=readiness_check)
    await server.start()
    return server

