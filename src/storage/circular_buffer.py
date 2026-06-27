# src/storage/circular_buffer.py
import os
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CircularBuffer:
    def __init__(self, clip_dir: str, max_storage_mb: int):
        self.clip_dir = Path(clip_dir)
        self.max_storage_mb = int(max_storage_mb)
        self._retention_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._retention_thread = None
        self._active_files: set[str] = set()

    def register_active_file(self, file_path: str) -> None:
        with self._active_lock:
            self._active_files.add(str(Path(file_path).resolve()))

    def unregister_active_file(self, file_path: str) -> None:
        with self._active_lock:
            self._active_files.discard(str(Path(file_path).resolve()))

    def _is_active_file(self, file_path: Path) -> bool:
        with self._active_lock:
            return str(file_path.resolve()) in self._active_files

    def enforce_retention_policy(self, extension: str = ".avi") -> None:
        if not self.clip_dir.exists():
            return

        files = [p for p in self.clip_dir.rglob(f"*{extension}") if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0)

        max_bytes = self.max_storage_mb * 1024 * 1024
        total_bytes = 0
        for f in files:
            try:
                total_bytes += f.stat().st_size
            except OSError:
                pass

        for file_path in files:
            if total_bytes <= max_bytes:
                break
            if self._is_active_file(file_path):
                continue
            try:
                size = file_path.stat().st_size
                os.remove(file_path)
                total_bytes -= size
                logger.info("Deleted old clip: %s", file_path)
            except OSError:
                logger.warning("Failed deleting clip: %s", file_path)

    def enforce_retention_policy_async(self, extension: str = ".avi") -> None:
        with self._retention_lock:
            if self._retention_thread and self._retention_thread.is_alive():
                return
            self._retention_thread = threading.Thread(
                target=self.enforce_retention_policy,
                kwargs={"extension": extension},
                daemon=True,
                name="RetentionPolicyWorker",
            )
            self._retention_thread.start()
