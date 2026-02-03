import socket
from typing import Any, cast

import style_bert_vits2.logging as sbv2_logging
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.worker_common import (
    ConnectionClosedException,
    RequestType,
    receive_data,
    send_data,
)

logger = sbv2_logging.logger


class WorkerClient:
    """pyopenjtalk worker client"""

    def __init__(self, port: int) -> None:
        self.port = port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # timeout: 60 seconds
        sock.settimeout(60)
        sock.connect((socket.gethostname(), port))
        self.sock = sock

    def __enter__(self) -> "WorkerClient":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def _reconnect(self) -> None:
        """Reconnect to the worker server"""
        self.close()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(60)
        sock.connect((socket.gethostname(), self.port))
        self.sock = sock

    def dispatch_pyopenjtalk(self, func: str, *args: Any, **kwargs: Any) -> Any:
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if self.sock is None:
                    self._reconnect()
                assert self.sock is not None
                data = {
                    "request-type": RequestType.PYOPENJTALK,
                    "func": func,
                    "args": args,
                    "kwargs": kwargs,
                }
                logger.trace(f"client sends request: {data}")
                send_data(self.sock, data)
                logger.trace("client sent request successfully")
                response = receive_data(self.sock)
                logger.trace(f"client received response: {response}")
                return response.get("return")
            except (
                ConnectionClosedException,
                BrokenPipeError,
                ConnectionResetError,
                OSError,
                socket.error,
            ) as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Connection error during dispatch (attempt {attempt + 1}/{max_retries + 1}): {e}. Reconnecting..."
                    )
                    try:
                        self._reconnect()
                    except Exception as reconnect_error:
                        logger.error(f"Failed to reconnect: {reconnect_error}")
                        raise
                else:
                    logger.error(f"Connection error after {max_retries + 1} attempts: {e}")
                    raise

    def status(self) -> int:
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if self.sock is None:
                    self._reconnect()
                assert self.sock is not None
                data = {"request-type": RequestType.STATUS}
                logger.trace(f"client sends request: {data}")
                send_data(self.sock, data)
                logger.trace("client sent request successfully")
                response = receive_data(self.sock)
                logger.trace(f"client received response: {response}")
                return cast(int, response.get("client-count"))
            except (
                ConnectionClosedException,
                BrokenPipeError,
                ConnectionResetError,
                OSError,
                socket.error,
            ) as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Connection error during status check (attempt {attempt + 1}/{max_retries + 1}): {e}. Reconnecting..."
                    )
                    try:
                        self._reconnect()
                    except Exception as reconnect_error:
                        logger.error(f"Failed to reconnect: {reconnect_error}")
                        raise
                else:
                    logger.error(f"Connection error after {max_retries + 1} attempts: {e}")
                    raise RuntimeError(f"Failed to get status after {max_retries + 1} attempts") from e
        raise RuntimeError("Failed to get status")

    def quit_server(self) -> None:
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                if self.sock is None:
                    return
                assert self.sock is not None
                data = {"request-type": RequestType.QUIT_SERVER}
                logger.trace(f"client sends request: {data}")
                send_data(self.sock, data)
                logger.trace("client sent request successfully")
                response = receive_data(self.sock)
                logger.trace(f"client received response: {response}")
                return
            except (
                ConnectionClosedException,
                BrokenPipeError,
                ConnectionResetError,
                OSError,
                socket.error,
            ) as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Connection error during quit (attempt {attempt + 1}/{max_retries + 1}): {e}. Reconnecting..."
                    )
                    try:
                        self._reconnect()
                    except Exception as reconnect_error:
                        logger.error(f"Failed to reconnect: {reconnect_error}")
                        raise
                else:
                    logger.error(f"Connection error after {max_retries + 1} attempts: {e}")
                    raise
