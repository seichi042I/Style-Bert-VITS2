"""
Run the pyopenjtalk worker in a separate process
to avoid user dictionary access error.

Multiple worker server processes can be started for parallel processing
(see ``initialize_worker(num_workers=...)``).  Each calling thread is
assigned its own worker via round-robin so that requests can be processed
concurrently without socket contention.
"""

import socket
import threading
from typing import Any, Optional

from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.worker_client import WorkerClient
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.worker_common import (
    ConnectionClosedException,
    WORKER_PORT,
    suppress_pyopenjtalk_stderr,
)


# ── Worker entry (server process + client connection + per-worker lock) ──────


class _WorkerEntry:
    """One worker server process paired with a client connection."""

    __slots__ = ("client", "port", "lock")

    def __init__(self, client: WorkerClient, port: int) -> None:
        self.client = client
        self.port = port
        # Per-worker lock: threads sharing the same entry are serialised,
        # but threads on *different* entries run in parallel.
        self.lock = threading.Lock()


# ── Module state ─────────────────────────────────────────────────────────────

# Legacy reference (first worker's client) – kept for backward compatibility.
WORKER_CLIENT: Optional[WorkerClient] = None

_WORKERS: list[_WorkerEntry] = []
_BASE_PORT: int = WORKER_PORT

# Thread-local worker assignment (round-robin).
_thread_local = threading.local()
_assign_counter = 0
_assign_lock = threading.Lock()

# Lock for direct pyopenjtalk calls (non-worker fallback mode).
_DIRECT_CALL_LOCK = threading.Lock()


# ── Internal helpers ─────────────────────────────────────────────────────────


def _get_worker() -> Optional[_WorkerEntry]:
    """Return the ``_WorkerEntry`` assigned to the current thread."""
    if not _WORKERS:
        return None
    # Assign once per thread.  Re-assign when _WORKERS has been rebuilt
    # (e.g. after terminate_worker + re-initialise).
    if not hasattr(_thread_local, "worker") or _thread_local.worker not in _WORKERS:
        global _assign_counter
        with _assign_lock:
            idx = _assign_counter % len(_WORKERS)
            _assign_counter += 1
        _thread_local.worker = _WORKERS[idx]
    return _thread_local.worker


def _start_or_connect_worker(port: int) -> WorkerClient:
    """Connect to an existing worker server on *port*, or start a new one."""
    import sys
    import time

    try:
        return WorkerClient(port)
    except (OSError, socket.timeout):
        logger.debug(f"Starting pyopenjtalk worker server on port {port}")
        import os
        import subprocess

        worker_pkg_path = os.path.relpath(
            os.path.dirname(__file__), os.getcwd()
        ).replace(os.sep, ".")
        args = [sys.executable, "-m", worker_pkg_path, "--port", str(port)]

        if sys.platform.startswith("win"):
            cf = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore
            si = subprocess.STARTUPINFO()  # type: ignore
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore
            si.wShowWindow = subprocess.SW_HIDE  # type: ignore
            subprocess.Popen(args, creationflags=cf, startupinfo=si)
        else:
            subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        # Wait until the server starts listening.
        for _ in range(20):
            try:
                return WorkerClient(port)
            except OSError:
                time.sleep(0.5)

        raise TimeoutError(f"ポート {port} のワーカーサーバーに接続できませんでした")


def _restart_worker_entry(worker: _WorkerEntry) -> None:
    """Restart a single worker server (called while holding ``worker.lock``)."""
    logger.warning(f"Restarting pyopenjtalk worker on port {worker.port}...")
    try:
        worker.client.close()
    except Exception:
        pass
    worker.client = _start_or_connect_worker(worker.port)


def _dispatch(func_name: str, *args: Any, **kwargs: Any) -> Any:
    """Dispatch a pyopenjtalk call to the current thread's worker."""
    worker = _get_worker()
    if worker is not None:
        with worker.lock:
            max_retries = 1
            for attempt in range(max_retries + 1):
                try:
                    return worker.client.dispatch_pyopenjtalk(func_name, *args, **kwargs)
                except (
                    ConnectionClosedException,
                    BrokenPipeError,
                    ConnectionResetError,
                    OSError,
                    socket.error,
                ) as e:
                    if attempt < max_retries:
                        logger.warning(
                            f"Worker connection error during {func_name} "
                            f"(attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            "Restarting worker..."
                        )
                        _restart_worker_entry(worker)
                    else:
                        logger.error(
                            f"Worker connection error after {max_retries + 1} attempts: {e}"
                        )
                        raise
    else:
        # Non-worker fallback: call pyopenjtalk directly (serialised).
        import pyopenjtalk

        with _DIRECT_CALL_LOCK:
            with suppress_pyopenjtalk_stderr():
                return getattr(pyopenjtalk, func_name)(*args, **kwargs)


def _broadcast(func_name: str, *args: Any, **kwargs: Any) -> Any:
    """Send a pyopenjtalk call to ALL worker servers.

    Used for dictionary operations that must be applied to every server
    process so that they all share the same dictionary state.
    """
    if _WORKERS:
        result = None
        for worker in _WORKERS:
            with worker.lock:
                max_retries = 1
                for attempt in range(max_retries + 1):
                    try:
                        result = worker.client.dispatch_pyopenjtalk(
                            func_name, *args, **kwargs
                        )
                        break
                    except (
                        ConnectionClosedException,
                        BrokenPipeError,
                        ConnectionResetError,
                        OSError,
                        socket.error,
                    ) as e:
                        if attempt < max_retries:
                            logger.warning(
                                f"Worker connection error during {func_name} "
                                f"on port {worker.port} "
                                f"(attempt {attempt + 1}/{max_retries + 1}): {e}. "
                                "Restarting worker..."
                            )
                            _restart_worker_entry(worker)
                        else:
                            logger.error(
                                f"Worker connection error after {max_retries + 1} attempts: {e}"
                            )
                            raise
        return result
    else:
        import pyopenjtalk

        with _DIRECT_CALL_LOCK:
            with suppress_pyopenjtalk_stderr():
                return getattr(pyopenjtalk, func_name)(*args, **kwargs)


# ── pyopenjtalk public interface ─────────────────────────────────────────────
# g2p(): not used


def run_frontend(text: str) -> list[dict[str, Any]]:
    ret = _dispatch("run_frontend", text)
    assert isinstance(ret, list)
    return ret


def make_label(njd_features: Any) -> list[str]:
    ret = _dispatch("make_label", njd_features)
    assert isinstance(ret, list)
    return ret


# Dictionary operations.
# mecab_dict_index: ディスク上の辞書ファイルを生成するだけなので 1 台で十分。
# unset_user_dict / update_global_jtalk_with_user_dict: 各サーバープロセスの
#   メモリ上の辞書を操作するため全ワーカーへのブロードキャストが必要。


def mecab_dict_index(path: str, out_path: str, dn_mecab: Optional[str] = None) -> None:
    _dispatch("mecab_dict_index", path, out_path, dn_mecab)


def update_global_jtalk_with_user_dict(path: str) -> None:
    _broadcast("update_global_jtalk_with_user_dict", path)


def unset_user_dict() -> None:
    _broadcast("unset_user_dict")


# ── Initialisation / teardown ────────────────────────────────────────────────


def initialize_worker(port: int = WORKER_PORT, num_workers: int = 1) -> None:
    """Start *num_workers* pyopenjtalk worker server processes.

    Safe to call multiple times.  Additional servers are started only when
    *num_workers* exceeds the current count.  Each server runs on a
    consecutive port starting from *port*.
    """
    import atexit
    import signal

    global WORKER_CLIENT, _BASE_PORT
    _BASE_PORT = port

    current = len(_WORKERS)
    if current >= num_workers:
        return

    for i in range(current, num_workers):
        worker_port = port + i
        client = _start_or_connect_worker(worker_port)
        _WORKERS.append(_WorkerEntry(client, worker_port))
        logger.debug(f"pyopenjtalk worker server started on port {worker_port}")

    # backward compat
    WORKER_CLIENT = _WORKERS[0].client

    if current == 0:
        # First call – register cleanup handlers.
        atexit.register(terminate_worker)

        def signal_handler(signum: int, frame: Any) -> None:
            terminate_worker()

        try:
            signal.signal(signal.SIGTERM, signal_handler)
        except ValueError:
            # signal only works in main thread
            pass


def terminate_worker() -> None:
    """Shut down all worker server processes."""
    global WORKER_CLIENT
    logger.debug("pyopenjtalk worker servers terminated")

    for worker in _WORKERS:
        with worker.lock:
            try:
                worker.client.quit_server()
            except Exception as e:
                logger.error(e)
            try:
                worker.client.close()
            except Exception:
                pass

    _WORKERS.clear()
    WORKER_CLIENT = None
