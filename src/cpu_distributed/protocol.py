"""Simple TCP message protocol for coordinator-worker communication.

Messages are JSON objects prefixed with a 4-byte length header.
Communication: seeds + scalars only (not gradient tensors).
"""

import json
import logging
import socket
import struct
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

HEADER_SIZE = 4  # 4-byte big-endian length prefix


def send_message(sock: socket.socket, msg: Dict[str, Any]) -> None:
    """Send a JSON message with length prefix.

    Args:
        sock: TCP socket.
        msg: Dictionary to send as JSON.
    """
    data = json.dumps(msg).encode("utf-8")
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)


def recv_message(sock: socket.socket, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Receive a length-prefixed JSON message.

    Args:
        sock: TCP socket.
        timeout: Socket timeout in seconds.

    Returns:
        Parsed message dictionary, or None on failure.
    """
    if timeout is not None:
        sock.settimeout(timeout)

    try:
        header = _recv_exact(sock, HEADER_SIZE)
        if header is None:
            return None
        length = struct.unpack("!I", header)[0]

        if length > 10_000_000:  # 10 MB sanity limit
            logger.error("Message too large: %d bytes", length)
            return None

        data = _recv_exact(sock, length)
        if data is None:
            return None

        return json.loads(data.decode("utf-8"))

    except (socket.timeout, ConnectionError, json.JSONDecodeError) as e:
        logger.debug("recv_message failed: %s", e)
        return None


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Receive exactly n bytes from socket."""
    data = b""
    while len(data) < n:
        try:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        except (ConnectionError, OSError):
            return None
    return data


# Message types
MSG_REGISTER = "register"
MSG_TASK = "task"           # Coordinator → Worker: seeds to evaluate
MSG_RESULT = "result"       # Worker → Coordinator: fitness values
MSG_PARAMS = "params"       # Coordinator → Worker: updated parameters
MSG_SHUTDOWN = "shutdown"
MSG_HEARTBEAT = "heartbeat"
