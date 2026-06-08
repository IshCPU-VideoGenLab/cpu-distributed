"""Tests for cpu_distributed.protocol."""
import socket
import threading
import pytest
from cpu_distributed.protocol import send_message, recv_message


class TestProtocol:
    def test_roundtrip(self) -> None:
        """Send and receive a message over localhost."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen(1)

        received = {}

        def server_fn():
            conn, _ = server.accept()
            msg = recv_message(conn, timeout=5.0)
            received.update(msg or {})
            conn.close()
            server.close()

        t = threading.Thread(target=server_fn)
        t.start()

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))
        send_message(client, {"type": "test", "value": 42})
        client.close()
        t.join(timeout=5.0)

        assert received.get("type") == "test"
        assert received.get("value") == 42
