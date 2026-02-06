import contextlib
import json
import os
import socket
from enum import IntEnum, auto
from typing import Any, Final


WORKER_PORT: Final[int] = 7861
HEADER_SIZE: Final[int] = 4


@contextlib.contextmanager
def suppress_pyopenjtalk_stderr():
    """
    Redirect C-level stderr (fd 2) to devnull to suppress Open JTalk warnings
    such as "First mora should not be short pause" from jpcommon_label.c.
    """
    stderr_fd = 2
    try:
        save_fd = os.dup(stderr_fd)
    except OSError:
        yield
        return
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
        yield
    finally:
        try:
            os.dup2(save_fd, stderr_fd)
        except OSError:
            pass
        try:
            os.close(save_fd)
        except OSError:
            pass


class RequestType(IntEnum):
    STATUS = auto()
    QUIT_SERVER = auto()
    PYOPENJTALK = auto()


class ConnectionClosedException(Exception):
    pass


# socket communication


def send_data(sock: socket.socket, data: dict[str, Any]):
    json_data = json.dumps(data).encode()
    header = len(json_data).to_bytes(HEADER_SIZE, byteorder="big")
    sock.sendall(header + json_data)


def __receive_until(sock: socket.socket, size: int):
    data = b""
    while len(data) < size:
        part = sock.recv(size - len(data))
        if part == b"":
            raise ConnectionClosedException("接続が閉じられました")
        data += part

    return data


def receive_data(sock: socket.socket) -> dict[str, Any]:
    header = __receive_until(sock, HEADER_SIZE)
    data_length = int.from_bytes(header, byteorder="big")
    body = __receive_until(sock, data_length)
    return json.loads(body.decode())
