import logging
import os
import gc
import time
import mmap
import shutil
import queue
import fcntl
import tempfile
import threading
import weakref
import uuid
from typing import Dict, Optional, Tuple, Iterable, List, Any

import torch

LOGGER = logging.getLogger(__name__)


class MemoryProjCache:
    """Per-instance in-memory cache for projections.

    IMPORTANT: projection tensors depend on the underlying volume. Therefore this cache
    must NOT be shared across different volumes / solvers in the same process.

    Stores per-key tensors on CPU by default to avoid GPU memory blow-up.
    """

    def __init__(self):
        self._base_proj_cache: Optional[torch.Tensor] = None
        self._base_proj_cache_side_length: Optional[int] = None
        self._raw_proj_cache: Dict[int, torch.Tensor] = {}

    # ----------------------------
    # SO3 base cache helpers
    # ----------------------------
    def base_proj_cache_ok(self, side_length: int) -> bool:
        return (
            self._base_proj_cache is not None
            and self._base_proj_cache_side_length == int(side_length)
        )

    def get_base_proj(self) -> Optional[torch.Tensor]:
        return self._base_proj_cache

    def set_base_proj(self, proj: torch.Tensor, side_length: int):
        self._base_proj_cache = proj
        self._base_proj_cache_side_length = int(side_length)

    # ----------------------------
    # raw per-key cache helpers
    # ----------------------------
    def has(self, key: int) -> bool:
        return int(key) in self._raw_proj_cache

    def get(
        self, key: int, device: Optional[torch.device] = None
    ) -> Optional[torch.Tensor]:
        t = self._raw_proj_cache.get(int(key), None)
        if t is None:
            return None
        if device is None:
            return t
        return t.to(device)

    def put(self, key: int, tensor: torch.Tensor, store_on_cpu: bool = True):
        if store_on_cpu:
            self._raw_proj_cache[int(key)] = tensor.detach().cpu()
        else:
            self._raw_proj_cache[int(key)] = tensor.detach()

    def find_missing_positions(self, keys: Iterable[int]) -> List[int]:
        missing: List[int] = []
        for i, k in enumerate(keys):
            if int(k) not in self._raw_proj_cache:
                missing.append(i)
        return missing

    def stack_many(self, keys: List[int], *, device: torch.device, dim: int = 0) -> torch.Tensor:
        return torch.stack([self._raw_proj_cache[int(k)].to(device) for k in keys], dim=dim)

    def clear(self):
        self._base_proj_cache = None
        self._base_proj_cache_side_length = None
        self._raw_proj_cache = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SSDProjCache:
    """SSD-backed projection cache with process-wide serialized file operations.

    This cache intentionally uses one shared worker queue/thread per process
    rather than one worker per cache instance. The main goal is to avoid
    same-process read/write/delete races between old and new cache instances
    while a pose searcher refreshes cache directories. Cross-process
    synchronization is still handled via lock files and ready markers on disk;
    the shared worker only guarantees that this process does not issue
    conflicting filesystem operations concurrently.

    As a consequence, async writes and async cleanup are ordered process-wide,
    and ``flush()`` waits for the shared queue to drain rather than flushing
    only one instance in isolation.
    """
    _ID_LOCK = threading.Lock()
    _ID_COUNTER = 0

    _WORKER_LOCK = threading.Lock()
    _WORKER_QUEUE: Optional["queue.Queue[Tuple[str, Tuple[Any, ...]]]"] = None
    _WORKER_THREAD: Optional[threading.Thread] = None

    @classmethod
    def _ensure_worker(cls) -> "queue.Queue[Tuple[str, Tuple[Any, ...]]]":
        """Return the shared worker queue, starting the process-wide worker if needed."""
        with cls._WORKER_LOCK:
            if cls._WORKER_QUEUE is None:
                cls._WORKER_QUEUE = queue.Queue()
            if cls._WORKER_THREAD is None or (not cls._WORKER_THREAD.is_alive()):
                cls._WORKER_THREAD = threading.Thread(
                    target=cls._worker_loop,
                    name="SSDProjCacheWorker",
                    daemon=True,
                )
                cls._WORKER_THREAD.start()
            return cls._WORKER_QUEUE

    @classmethod
    def _enqueue(cls, op: str, *args: Any) -> None:
        """Schedule one filesystem operation on the shared worker queue."""
        q = cls._ensure_worker()
        q.put((op, tuple(args)))

    @classmethod
    def _worker_loop(cls) -> None:
        """Serialize SSD cache filesystem operations within the current process."""
        q = cls._ensure_worker()
        while True:
            op, args = q.get()
            try:
                if op == "write":
                    cache_dir, key, tensor_cpu = args
                    cls._write_item(str(cache_dir), int(key), tensor_cpu)
                elif op == "clear_files":
                    cache_dir, remove_locks = args
                    cls._clear_files(str(cache_dir), bool(remove_locks))
                elif op == "delete_dir":
                    path = str(args[0])
                    cls._delete_dir(path)
                elif op == "stop":
                    return
            except Exception:
                LOGGER.exception("SSD projection cache worker failed during `%s`.", op)
            finally:
                try:
                    q.task_done()
                except Exception:
                    pass

    @staticmethod
    def _delete_dir(path: str) -> None:
        root = os.path.abspath(str(path))
        if root in ("/", ""):
            return
        if os.path.isdir(root):
            shutil.rmtree(root, ignore_errors=True)
        elif os.path.exists(root):
            try:
                os.unlink(root)
            except FileNotFoundError:
                pass

    @staticmethod
    def _clear_files(cache_dir: str, remove_locks: bool) -> None:
        if not os.path.isdir(cache_dir):
            return

        for name in os.listdir(cache_dir):
            if not name.startswith("proj_cache_"):
                continue
            remove = name.endswith((".bin", ".meta", ".done")) or (
                remove_locks and name.endswith(".lock")
            )
            if not remove:
                continue
            try:
                os.unlink(os.path.join(cache_dir, name))
            except FileNotFoundError:
                pass

    @staticmethod
    def _dist_is_rank0() -> bool:
        if not (hasattr(torch, "distributed") and torch.distributed.is_available()):
            return True
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    @classmethod
    def _should_cleanup_path(cls, cache_dir: str) -> bool:
        if cls._dist_is_rank0():
            return True
        base = os.path.basename(os.path.abspath(str(cache_dir)))
        return base.startswith("ssd_proj_cache_")

    @classmethod
    def _new_instance_id(cls) -> str:
        with cls._ID_LOCK:
            cls._ID_COUNTER += 1
            n = cls._ID_COUNTER
        pid = os.getpid()
        suffix = uuid.uuid4().hex[:8]
        return f"{pid}_{n}_{suffix}"

    def __init__(
        self,
        cache_root: Optional[str] = None,
        cache_dir: Optional[str] = None,
        *,
        clear: bool = False,
        auto_clear: bool = True,
    ):
        self._cache_dir: Optional[str] = None
        self._active_dir: Optional[str] = None
        self._closed = False
        self._auto_clear = bool(auto_clear)

        self._finalizer: Optional[weakref.finalize] = None

        if cache_dir is not None:
            self.set_cache_dir(cache_dir, clear=clear)
        elif cache_root is not None:
            inst_id = self._new_instance_id()
            unique_dir = os.path.join(os.path.abspath(str(cache_root)), f"ssd_proj_cache_{inst_id}")
            self.set_cache_dir(unique_dir, clear=clear)

    def __del__(self):
        if not getattr(self, "_auto_clear", False):
            return
        try:
            self.close(delete_root=True, remove_locks=True, async_clear=True)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(delete_root=True, remove_locks=True, async_clear=True)
        return False

    @staticmethod
    def _paths(cache_dir: str, key: int) -> Tuple[str, str, str, str]:
        data_file = os.path.join(cache_dir, f"proj_cache_{key}.bin")
        meta_file = os.path.join(cache_dir, f"proj_cache_{key}.meta")
        done_file = os.path.join(cache_dir, f"proj_cache_{key}.done")
        lock_file = os.path.join(cache_dir, f"proj_cache_{key}.lock")
        return data_file, meta_file, done_file, lock_file

    @classmethod
    def _ready(cls, cache_dir: str, key: int) -> bool:
        data_file, meta_file, done_file, _ = cls._paths(cache_dir, key)
        return os.path.exists(done_file) or (
            os.path.exists(data_file) and os.path.exists(meta_file)
        )

    @staticmethod
    def _finalize_cleanup(cache_dir: Optional[str]):
        if cache_dir is None:
            return
        abs_dir = os.path.abspath(str(cache_dir))
        if not SSDProjCache._should_cleanup_path(abs_dir):
            return
        try:
            SSDProjCache._enqueue("delete_dir", abs_dir)
        except Exception:
            pass

    def set_cache_dir(
        self,
        cache_dir: Optional[str],
        clear: bool = False,
        *,
        is_primary: Optional[bool] = None,
    ):
        self._cache_dir = None if cache_dir is None else os.path.abspath(str(cache_dir))
        self._active_dir = self._cache_dir

        if self._finalizer is not None:
            try:
                self._finalizer.detach()
            except Exception:
                pass
            self._finalizer = None

        if self._cache_dir is None:
            return

        do_clear = self._dist_is_rank0() if is_primary is None else bool(is_primary)
        if clear and do_clear:
            root = self._cache_dir
            if root not in ("/", ""):
                if os.path.isdir(root):
                    shutil.rmtree(root, ignore_errors=True)
                elif os.path.exists(root):
                    try:
                        os.unlink(root)
                    except FileNotFoundError:
                        pass

        os.makedirs(self._cache_dir, exist_ok=True)

        if self._auto_clear:
            self._finalizer = weakref.finalize(
                self, SSDProjCache._finalize_cleanup, self._cache_dir
            )

    def active_dir(self) -> Optional[str]:
        return self._active_dir

    def find_missing_positions(
        self, keys: Iterable[int], cache_dir: Optional[str] = None
    ) -> List[int]:
        cache_dir = self._active_dir if cache_dir is None else cache_dir
        if cache_dir is None:
            return list(range(len(list(keys))))

        missing: List[int] = []
        for i, k in enumerate(keys):
            if not self._ready(cache_dir, int(k)):
                missing.append(i)
        return missing

    @classmethod
    def _write_item(cls, cache_dir: str, key: int, tensor: torch.Tensor) -> bool:
        data_file, meta_file, done_file, lock_file = cls._paths(cache_dir, int(key))

        if os.path.exists(done_file) or (
            os.path.exists(data_file) and os.path.exists(meta_file)
        ):
            return True

        os.makedirs(cache_dir, exist_ok=True)

        with open(lock_file, "a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True

            if os.path.exists(done_file) or (
                os.path.exists(data_file) and os.path.exists(meta_file)
            ):
                return True

            t_cpu = tensor.detach()
            if t_cpu.device.type != "cpu":
                t_cpu = t_cpu.to("cpu")
            t_cpu = t_cpu.contiguous()

            with tempfile.NamedTemporaryFile(
                dir=cache_dir, suffix=".meta", delete=False
            ) as tmp_meta:
                shape_str = "x".join(map(str, t_cpu.shape))
                dtype_str = str(t_cpu.dtype).split(".")[-1]
                tmp_meta.write(f"{shape_str},{dtype_str}".encode())
                tmp_meta_path = tmp_meta.name

            with tempfile.NamedTemporaryFile(
                dir=cache_dir, suffix=".bin", delete=False
            ) as tmp_data:
                tmp_data.write(t_cpu.numpy().tobytes())
                tmp_data_path = tmp_data.name

            os.replace(tmp_meta_path, meta_file)
            os.replace(tmp_data_path, data_file)

            try:
                fd = os.open(done_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.close(fd)
            except FileExistsError:
                pass

        return True

    def put(self, key: int, tensor: torch.Tensor, async_write: bool = True) -> bool:
        cache_dir = self._active_dir
        if cache_dir is None:
            return False

        if not async_write:
            self._write_item(cache_dir, int(key), tensor)
            return True

        t_cpu = tensor.detach()
        if t_cpu.device.type != "cpu":
            t_cpu = t_cpu.to("cpu")
        t_cpu = t_cpu.contiguous()

        try:
            SSDProjCache._enqueue("write", cache_dir, int(key), t_cpu)
        except Exception:
            self._write_item(cache_dir, int(key), t_cpu)
        return True

    def get(
        self,
        key: int,
        device: Optional[torch.device] = None,
        max_wait: float = 10.0,
        poll_interval: float = 0.1,
        raise_on_miss: bool = True,
        raise_on_error: bool = True,
    ) -> Optional[torch.Tensor]:
        cache_dir = self._active_dir
        if cache_dir is None:
            return None

        key = int(key)
        data_file, meta_file, done_file, lock_file = self._paths(cache_dir, key)

        waited = 0.0
        while (not self._ready(cache_dir, key)) and waited < max_wait:
            time.sleep(poll_interval)
            waited += poll_interval

        if not self._ready(cache_dir, key):
            if raise_on_miss:
                raise FileNotFoundError(
                    f"Cache files for key {key} not found after {max_wait} seconds"
                )
            return None

        try:
            with open(lock_file, "a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_SH)

                if not self._ready(cache_dir, key):
                    if raise_on_miss:
                        raise FileNotFoundError(
                            f"Cache files for key {key} disappeared"
                        )
                    return None

                with open(meta_file, "r") as f:
                    shape_str, dtype_str = f.read().strip().split(",")
                shape = tuple(map(int, shape_str.split("x")))
                torch_dtype = getattr(torch, dtype_str)

                with open(data_file, "rb") as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        buf = torch.frombuffer(mm, dtype=torch_dtype)
                        t = buf.view(shape).clone()

            if device is not None:
                t = t.to(device)
            return t

        except Exception as e:
            print(f"[SSDProjCache] Error reading cache for key {key}: {e}")
            if raise_on_error:
                raise
            return None

    def clear(
        self,
        remove_locks: bool = False,
        cache_dir: Optional[str] = None,
        delete_dir: Optional[bool] = None,
        *,
        delete_root: bool = False,
        async_clear: bool = True,
    ):
        """Clear cache files or directories.

        When ``async_clear`` is true, cleanup is enqueued onto the shared
        process-wide worker so that same-process delete/write operations remain
        serialized across cache instances.
        """
        if delete_root:
            cache_dir = self._cache_dir
            delete_dir = True
        else:
            cache_dir = self._active_dir if cache_dir is None else cache_dir

        if cache_dir is None:
            return

        if delete_dir is None:
            delete_dir = False

        if not SSDProjCache._should_cleanup_path(cache_dir):
            return

        if delete_dir:
            if async_clear:
                try:
                    SSDProjCache._enqueue("delete_dir", cache_dir)
                except Exception:
                    SSDProjCache._delete_dir(cache_dir)
                return
            SSDProjCache._delete_dir(cache_dir)
            return

        if async_clear:
            try:
                SSDProjCache._enqueue("clear_files", cache_dir, bool(remove_locks))
            except Exception:
                SSDProjCache._clear_files(cache_dir, bool(remove_locks))
            return

        SSDProjCache._clear_files(cache_dir, bool(remove_locks))

    def close(
        self,
        *,
        delete_root: bool = True,
        remove_locks: bool = True,
        async_clear: bool = False,
        cleanup: bool = True,
    ):
        """Close this cache handle and optionally schedule cleanup.

        Closing one instance does not stop the shared worker because that worker
        may still be serving other SSDProjCache instances in the same process.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if cleanup:
                self.clear(
                    remove_locks=remove_locks,
                    delete_root=delete_root,
                    async_clear=async_clear,
                )
        finally:
            if self._finalizer is not None:
                try:
                    self._finalizer.detach()
                except Exception:
                    pass
                self._finalizer = None

    def flush(self) -> None:
        """Wait until the shared SSD cache worker queue is empty.

        This is intentionally a process-wide flush. It is used before cache
        directory refresh/cleanup boundaries so that older async writes from the
        same process cannot race with later directory changes.
        """
        try:
            q = self._ensure_worker()
            q.join()
        except Exception:
            pass