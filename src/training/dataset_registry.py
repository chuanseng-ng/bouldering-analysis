"""Dataset registry for tracking classification and detection training datasets.

Records each registered training dataset in a JSON file, allowing training runs
to audit which datasets contributed to a given model version and to compute
aggregate sample counts across all registered datasets.

Example:
    >>> from pathlib import Path
    >>> from src.training.dataset_registry import DatasetRegistry
    >>> registry = DatasetRegistry(Path("data/dataset_registry.json"))
    >>> record = registry.add(
    ...     name="roboflow-v2",
    ...     source_path="/path/to/dataset",
    ...     nc=7,
    ...     names=["jug", "crimp"],
    ...     sample_counts={"jug": 50, "crimp": 40},
    ... )
    >>> print(record.nc)
    7
"""

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.logging_config import get_logger

logger = get_logger(__name__)

# Default registry file path (relative to project root).
DEFAULT_REGISTRY_PATH: Path = Path("data/dataset_registry.json")


@dataclass
class DatasetRecord:
    """A single registered training dataset entry.

    Attributes:
        record_id: Unique UUID4 identifier for this record.
        name: Human-readable label for the dataset (e.g. ``"roboflow-v2-crimp-only"``).
        source_path: Absolute path to the dataset root at registration time.
        nc: Number of classes covered by this dataset.
        names: Class names present in this dataset (may be a subset of the
            canonical list).
        sample_counts: Per-class image counts from the training split.
        added_at: ISO-8601 UTC timestamp when this record was added.
        data_hash: SHA-256 fingerprint of ``source_path`` + ``sample_counts``
            used for duplicate detection.
    """

    record_id: str
    name: str
    source_path: str
    nc: int
    names: list[str]
    sample_counts: dict[str, int]
    added_at: str
    data_hash: str


def _compute_data_hash(source_path: str, sample_counts: dict[str, int]) -> str:
    """Compute a deterministic SHA-256 fingerprint for a dataset.

    The fingerprint is derived from ``source_path`` and a sorted JSON
    serialisation of ``sample_counts`` to ensure identical datasets produce
    the same hash regardless of dict insertion order.

    Args:
        source_path: Absolute path to the dataset root.
        sample_counts: Per-class image counts.

    Returns:
        Hex-encoded SHA-256 digest string.

    Example:
        >>> _compute_data_hash("/data/ds", {"jug": 10})  # doctest: +SKIP
        'abc123...'
    """
    counts_str = json.dumps(sample_counts, sort_keys=True)
    payload = f"{source_path}:{counts_str}"
    return hashlib.sha256(payload.encode()).hexdigest()


class DatasetRegistry:
    """File-backed registry of training datasets.

    Persists :class:`DatasetRecord` entries in a JSON file at
    ``registry_path``.  The file is created on the first write.  Reads and
    writes are not thread-safe — intended for use from a single training
    process.

    Args:
        registry_path: Path to the JSON registry file.  Does not need to
            exist yet — it will be created on the first :meth:`add` call.

    Example:
        >>> from pathlib import Path
        >>> registry = DatasetRegistry(Path("/tmp/registry.json"))
        >>> record = registry.add(
        ...     name="test-dataset",
        ...     source_path="/tmp/data",
        ...     nc=2,
        ...     names=["jug", "crimp"],
        ...     sample_counts={"jug": 10, "crimp": 20},
        ... )
        >>> len(registry.list_datasets())
        1
    """

    def __init__(
        self,
        registry_path: Path | str = DEFAULT_REGISTRY_PATH,
    ) -> None:
        """Initialise the registry with a given file path.

        Args:
            registry_path: Path to the JSON file.  Accepts both :class:`Path`
                objects and plain strings.
        """
        self._path = Path(registry_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        source_path: str,
        nc: int,
        names: list[str],
        sample_counts: dict[str, int],
    ) -> DatasetRecord:
        """Register a new dataset and persist it to the JSON file.

        If a record with the same data hash already exists a
        :class:`UserWarning` is emitted and the existing record is returned
        unchanged.

        Args:
            name: Human-readable label for the dataset.
            source_path: Absolute path to the dataset root.
            nc: Number of classes covered by this dataset.
            names: Class names present in this dataset.
            sample_counts: Per-class image counts from the training split.

        Returns:
            The newly created :class:`DatasetRecord`, or the existing record
            if a duplicate fingerprint is detected.

        Example:
            >>> registry = DatasetRegistry(Path("/tmp/reg.json"))
            >>> r = registry.add("ds", "/data", 1, ["jug"], {"jug": 5})
            >>> r.name
            'ds'
        """
        data_hash = _compute_data_hash(source_path, sample_counts)
        records = self._load()

        # Duplicate fingerprint check — warn and return existing record.
        for existing in records:
            if existing.data_hash == data_hash:
                warnings.warn(
                    f"Dataset with same fingerprint already registered as "
                    f"'{existing.name}' (id={existing.record_id}). "
                    "Returning existing record.",
                    UserWarning,
                    stacklevel=2,
                )
                return existing

        record = DatasetRecord(
            record_id=str(uuid4()),
            name=name,
            source_path=source_path,
            nc=nc,
            names=list(names),
            sample_counts=dict(sample_counts),
            added_at=datetime.now(tz=timezone.utc).isoformat(),
            data_hash=data_hash,
        )
        records.append(record)
        self._save(records)

        logger.info(
            "Registered dataset '%s' (id=%s, nc=%d, total=%d samples)",
            name,
            record.record_id,
            nc,
            sum(sample_counts.values()),
        )
        return record

    def list_datasets(self) -> list[DatasetRecord]:
        """Return all registered datasets in insertion order.

        Returns:
            List of :class:`DatasetRecord` instances.  Empty if the registry
            file does not exist or contains no records.

        Example:
            >>> registry = DatasetRegistry(Path("/tmp/empty.json"))
            >>> registry.list_datasets()
            []
        """
        return self._load()

    def get_by_id(self, record_id: str) -> DatasetRecord | None:
        """Retrieve a single record by its UUID.

        Args:
            record_id: UUID4 string from :attr:`DatasetRecord.record_id`.

        Returns:
            :class:`DatasetRecord` if found, else ``None``.

        Example:
            >>> registry = DatasetRegistry(Path("/tmp/reg.json"))
            >>> registry.get_by_id("nonexistent-id") is None
            True
        """
        for record in self._load():
            if record.record_id == record_id:
                return record
        return None

    def remove(self, record_id: str) -> bool:
        """Remove a record by its UUID.

        If the record does not exist the operation is a no-op.

        Args:
            record_id: UUID4 string to remove.

        Returns:
            ``True`` if the record was found and removed, ``False`` if no
            matching record existed.

        Example:
            >>> registry = DatasetRegistry(Path("/tmp/reg.json"))
            >>> registry.remove("nonexistent-id")
            False
        """
        records = self._load()
        filtered = [r for r in records if r.record_id != record_id]
        if len(filtered) == len(records):
            logger.debug("remove: record_id=%s not found, no-op", record_id)
            return False
        self._save(filtered)
        logger.info("Removed dataset record id=%s", record_id)
        return True

    def get_combined_counts(
        self,
        names_filter: list[str] | None = None,
    ) -> dict[str, int]:
        """Return the total sample count per class across all registered datasets.

        Args:
            names_filter: If provided, only include classes whose names appear
                in this list.  If ``None``, all classes from all datasets are
                included.

        Returns:
            Dict mapping class name to total image count, summed across all
            registered datasets.

        Example:
            >>> registry = DatasetRegistry(Path("/tmp/reg.json"))
            >>> registry.get_combined_counts()
            {}
        """
        totals: dict[str, int] = {}
        for record in self._load():
            for cls, count in record.sample_counts.items():
                if names_filter is not None and cls not in names_filter:
                    continue
                totals[cls] = totals.get(cls, 0) + count
        return totals

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[DatasetRecord]:
        """Load all records from the JSON file.

        Returns an empty list if the file does not exist or cannot be parsed.

        Returns:
            List of :class:`DatasetRecord` instances.
        """
        if not self._path.exists():
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw: list[dict[str, Any]] = json.load(f)
            return [DatasetRecord(**entry) for entry in raw]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to load registry from %s: %s", self._path, exc)
            return []

    def _save(self, records: list[DatasetRecord]) -> None:
        """Persist records to the JSON file.

        Creates parent directories if they do not exist.

        Args:
            records: All records to serialise.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in records], f, indent=2)
