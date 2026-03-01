"""Generic thread-safe model cache for inference modules.

This module provides ``_InferenceModelCache``, a generic container that
encapsulates the double-checked locking pattern used by both the detection
and classification inference modules.

Design notes:
    The cache uses a module-level ``threading.Lock`` with a fast-path check
    outside the lock (safe under CPython's GIL) and an inner re-check inside
    the lock (correct for free-threaded Python 3.13+ / PEP 703 no-GIL builds).
"""

import threading
from pathlib import Path
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class _InferenceModelCache(Generic[T]):
    """Thread-safe cache for loaded inference models, keyed by resolved path.

    Encapsulates the double-checked locking pattern so that individual
    inference modules (detection, classification) do not duplicate it.

    Type Parameters:
        T: The model type (e.g., ``YOLO`` or ``nn.Module``).

    Example:
        >>> cache: _InferenceModelCache[MyModel] = _InferenceModelCache()
        >>> model = cache.load_or_store("weights/best.pt", lambda p: MyModel(p))
    """

    def __init__(self) -> None:
        """Initialize an empty model cache with its own threading lock."""
        self._cache: dict[str, T] = {}
        self._lock: threading.Lock = threading.Lock()

    def get(self, resolved_key: str) -> T | None:
        """Return a cached model or None if not present.

        Args:
            resolved_key: Resolved absolute path string used as cache key.

        Returns:
            The cached model, or None if not in cache.
        """
        return self._cache.get(resolved_key)

    def load_or_store(self, weights_path: Path | str, loader: Callable[[Path], T]) -> T:
        """Return a cached model, loading it on the first call for a given path.

        Implements double-checked locking: the fast path (no lock) is safe
        under CPython's GIL; the inner re-check is safe for free-threaded
        builds (PEP 703).

        Args:
            weights_path: Path to the model weights file (relative or absolute).
            loader: Callable that receives the resolved Path and returns a
                loaded model instance.  Called at most once per unique path.

        Returns:
            Loaded model instance (either from cache or freshly loaded).
        """
        resolved_path = Path(weights_path).resolve()
        resolved = str(resolved_path)

        # Fast path: safe under CPython's GIL
        if resolved in self._cache:
            return self._cache[resolved]

        with self._lock:
            # Inner re-check for free-threaded Python 3.13+
            if resolved in self._cache:
                return self._cache[resolved]

            model = loader(resolved_path)
            self._cache[resolved] = model

        return model

    def clear(self) -> None:
        """Remove all cached models.

        Thread-safe: acquires the lock before clearing.  After this call,
        the next :meth:`load_or_store` for any path will reload from disk.

        Example:
            >>> cache.clear()
        """
        with self._lock:
            self._cache.clear()
