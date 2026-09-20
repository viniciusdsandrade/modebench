"""The connector that opens a real websocket. It is the only module of the suite with a network.

The `websockets` exceptions become `ConnectionEnded`, so the replay depends
on one exception type and not on a library.
"""

import asyncio
from collections.abc import Mapping

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, WebSocketException

from modebench.stt.replay import ConnectionEnded, WsConnection


class _WebsocketsConnection:
    def __init__(self, connection: ClientConnection) -> None:
        self._connection = connection

    async def send(self, message: str | bytes) -> None:
        try:
            await self._connection.send(message)
        except ConnectionClosed as exc:
            raise ConnectionEnded(str(exc)) from exc

    async def recv(self) -> str | bytes:
        try:
            return await self._connection.recv()
        except ConnectionClosed as exc:
            raise ConnectionEnded(str(exc)) from exc

    async def close(self) -> None:
        await self._connection.close()


async def websockets_connector(
    url: str, headers: Mapping[str, str], timeout_s: float
) -> WsConnection:
    """Open a websocket with the credential headers and return it for the replay."""
    try:
        connection = await asyncio.wait_for(
            connect(url, additional_headers=dict(headers), max_size=None), timeout_s
        )
    except WebSocketException as exc:
        raise ConnectionEnded(f"{type(exc).__name__}: {exc}") from exc
    return _WebsocketsConnection(connection)
