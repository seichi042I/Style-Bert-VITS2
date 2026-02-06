"""
Run the pyopenjtalk worker in a separate process
to avoid user dictionary access error
"""

import socket
from typing import Any, Optional

from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.worker_client import WorkerClient
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.worker_common import (
    ConnectionClosedException,
    WORKER_PORT,
    suppress_pyopenjtalk_stderr,
)


WORKER_CLIENT: Optional[WorkerClient] = None
_WORKER_PORT: int = WORKER_PORT


# pyopenjtalk interface
# g2p(): not used


def _restart_worker() -> None:
    """Restart the worker process"""
    global WORKER_CLIENT
    logger.warning("Restarting pyopenjtalk worker...")
    terminate_worker()
    initialize_worker(_WORKER_PORT)


def run_frontend(text: str) -> list[dict[str, Any]]:
    if WORKER_CLIENT is not None:
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                ret = WORKER_CLIENT.dispatch_pyopenjtalk("run_frontend", text)
                assert isinstance(ret, list)
                return ret
            except (
                ConnectionClosedException,
                BrokenPipeError,
                ConnectionResetError,
                OSError,
                socket.error,
            ) as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Worker connection error during run_frontend (attempt {attempt + 1}/{max_retries + 1}): {e}. Restarting worker..."
                    )
                    _restart_worker()
                else:
                    logger.error(f"Worker connection error after {max_retries + 1} attempts: {e}")
                    raise
    else:
        # without worker
        import pyopenjtalk

        with suppress_pyopenjtalk_stderr():
            return pyopenjtalk.run_frontend(text)


def make_label(njd_features: Any) -> list[str]:
    if WORKER_CLIENT is not None:
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                ret = WORKER_CLIENT.dispatch_pyopenjtalk("make_label", njd_features)
                assert isinstance(ret, list)
                return ret
            except (
                ConnectionClosedException,
                BrokenPipeError,
                ConnectionResetError,
                OSError,
                socket.error,
            ) as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Worker connection error during make_label (attempt {attempt + 1}/{max_retries + 1}): {e}. Restarting worker..."
                    )
                    _restart_worker()
                else:
                    logger.error(f"Worker connection error after {max_retries + 1} attempts: {e}")
                    raise
    else:
        # without worker
        import pyopenjtalk

        with suppress_pyopenjtalk_stderr():
            return pyopenjtalk.make_label(njd_features)


def mecab_dict_index(path: str, out_path: str, dn_mecab: Optional[str] = None) -> None:
    if WORKER_CLIENT is not None:
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                WORKER_CLIENT.dispatch_pyopenjtalk("mecab_dict_index", path, out_path, dn_mecab)
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
                        f"Worker connection error during mecab_dict_index (attempt {attempt + 1}/{max_retries + 1}): {e}. Restarting worker..."
                    )
                    _restart_worker()
                else:
                    logger.error(f"Worker connection error after {max_retries + 1} attempts: {e}")
                    raise
    else:
        # without worker
        import pyopenjtalk

        with suppress_pyopenjtalk_stderr():
            pyopenjtalk.mecab_dict_index(path, out_path, dn_mecab)


def update_global_jtalk_with_user_dict(path: str) -> None:
    if WORKER_CLIENT is not None:
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                WORKER_CLIENT.dispatch_pyopenjtalk("update_global_jtalk_with_user_dict", path)
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
                        f"Worker connection error during update_global_jtalk_with_user_dict (attempt {attempt + 1}/{max_retries + 1}): {e}. Restarting worker..."
                    )
                    _restart_worker()
                else:
                    logger.error(f"Worker connection error after {max_retries + 1} attempts: {e}")
                    raise
    else:
        # without worker
        import pyopenjtalk

        with suppress_pyopenjtalk_stderr():
            pyopenjtalk.update_global_jtalk_with_user_dict(path)


def unset_user_dict() -> None:
    if WORKER_CLIENT is not None:
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                WORKER_CLIENT.dispatch_pyopenjtalk("unset_user_dict")
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
                        f"Worker connection error during unset_user_dict (attempt {attempt + 1}/{max_retries + 1}): {e}. Restarting worker..."
                    )
                    _restart_worker()
                else:
                    logger.error(f"Worker connection error after {max_retries + 1} attempts: {e}")
                    raise
    else:
        # without worker
        import pyopenjtalk

        with suppress_pyopenjtalk_stderr():
            pyopenjtalk.unset_user_dict()


# initialize module when imported


def initialize_worker(port: int = WORKER_PORT) -> None:
    import atexit
    import signal
    import socket
    import sys
    import time

    global WORKER_CLIENT, _WORKER_PORT
    _WORKER_PORT = port
    if WORKER_CLIENT:
        return

    client = None
    try:
        client = WorkerClient(port)
    except (OSError, socket.timeout):
        logger.debug("try starting pyopenjtalk worker server")
        import os
        import subprocess

        worker_pkg_path = os.path.relpath(
            os.path.dirname(__file__), os.getcwd()
        ).replace(os.sep, ".")
        args = [sys.executable, "-m", worker_pkg_path, "--port", str(port)]
        # new session, new process group
        if sys.platform.startswith("win"):
            cf = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore
            si = subprocess.STARTUPINFO()  # type: ignore
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore
            si.wShowWindow = subprocess.SW_HIDE  # type: ignore
            subprocess.Popen(args, creationflags=cf, startupinfo=si)
        else:
            # align with Windows behavior
            # start_new_session is same as specifying setsid in preexec_fn
            subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        # wait until server listening
        count = 0
        while True:
            try:
                client = WorkerClient(port)
                break
            except OSError:
                time.sleep(0.5)
                count += 1
                # 20: max number of retries
                if count == 20:
                    raise TimeoutError("サーバーに接続できませんでした")

    logger.debug("pyopenjtalk worker server started")
    WORKER_CLIENT = client
    atexit.register(terminate_worker)

    # when the process is killed
    def signal_handler(signum: int, frame: Any):
        terminate_worker()

    try:
        signal.signal(signal.SIGTERM, signal_handler)
    except ValueError:
        # signal only works in main thread
        pass


# top-level declaration
def terminate_worker() -> None:
    logger.debug("pyopenjtalk worker server terminated")
    global WORKER_CLIENT
    if not WORKER_CLIENT:
        return

    # prepare for unexpected errors
    try:
        if WORKER_CLIENT.status() == 1:
            WORKER_CLIENT.quit_server()
    except Exception as e:
        logger.error(e)

    WORKER_CLIENT.close()
    WORKER_CLIENT = None
