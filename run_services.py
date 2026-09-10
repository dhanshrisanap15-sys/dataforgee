"""
Say That Sound — Persistent Service Supervisor & Process Manager.

Runs and monitors both the Token Server and the LiveKit Voice Agent Worker:
  1. Token Server (http://localhost:8080)
  2. LiveKit Agent Worker (agent.py dev)

Features:
  - Automatic crash recovery & process restarts with backoff
  - Combined stdout/stderr logging to console and logs/
  - Graceful shutdown on Ctrl+C (SIGINT)
  - Health status checks
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [SUPERVISOR] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("say-that-sound.supervisor")


class ManagedProcess:
    """Represents a supervised long-running child process."""

    def __init__(self, name: str, command: list, log_file: Path):
        self.name = name
        self.command = command
        self.log_file = log_file
        self.process: Optional[subprocess.Popen] = None
        self.restart_count = 0
        self.last_started: float = 0.0
        self._should_run = True

    def start(self):
        """Start or restart the managed process."""
        self.last_started = time.time()
        logger.info(f"Starting [{self.name}]: {' '.join(self.command)}")
        log_fp = open(self.log_file, "a", encoding="utf-8")
        
        # Set UTF-8 encoding in child environment
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        self.process = subprocess.Popen(
            self.command,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(Path(__file__).parent),
        )
        logger.info(f"[{self.name}] started with PID={self.process.pid} (logging to {self.log_file.name})")

    def is_alive(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None

    def stop(self, timeout: float = 5.0):
        """Gracefully terminate process."""
        self._should_run = False
        if self.process and self.is_alive():
            logger.info(f"Stopping [{self.name}] (PID={self.process.pid})...")
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing [{self.name}] (PID={self.process.pid})")
                self.process.kill()
                self.process.wait()
            logger.info(f"[{self.name}] stopped cleanly.")


async def supervise():
    """Supervisor loop monitoring child processes."""
    token_cmd = [sys.executable, "token_server.py"]
    agent_cmd = [sys.executable, "agent.py", "dev"]

    services = {
        "TokenServer": ManagedProcess("TokenServer", token_cmd, LOGS_DIR / "token_server.log"),
        "AgentWorker": ManagedProcess("AgentWorker", agent_cmd, LOGS_DIR / "agent.log"),
    }

    # Start all services
    for s in services.values():
        s.start()

    logger.info("All services started. Supervisor active (Press Ctrl+C to terminate).")

    stop_event = asyncio.Event()

    def _sig_handler(*_):
        logger.info("Termination signal received. Shutting down all services...")
        stop_event.set()

    # Hook signal handlers if supported
    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    except Exception:
        pass

    try:
        while not stop_event.is_set():
            for name, svc in services.items():
                if svc._should_run and not svc.is_alive():
                    exit_code = svc.process.poll() if svc.process else -1
                    svc.restart_count += 1
                    logger.warning(
                        f"[{name}] exited unexpectedly with code {exit_code}! "
                        f"Restart count: {svc.restart_count}. Restarting in 2s..."
                    )
                    await asyncio.sleep(2.0)
                    if not stop_event.is_set():
                        svc.start()

            await asyncio.sleep(1.0)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Exiting supervisor loop...")
    finally:
        for s in services.values():
            s.stop()
        logger.info("All services shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(supervise())
    except KeyboardInterrupt:
        logger.info("Shutdown requested via KeyboardInterrupt.")
