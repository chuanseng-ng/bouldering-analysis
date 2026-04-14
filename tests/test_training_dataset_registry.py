"""Tests for the dataset registry module.

Covers CRUD operations, JSON persistence, combined-count aggregation,
duplicate-fingerprint detection, and no-op removal behaviour.
"""

import json
from pathlib import Path

import pytest

from src.training.dataset_registry import (
    DEFAULT_REGISTRY_PATH,
    DatasetRecord,
    DatasetRegistry,
    _compute_data_hash,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    """Return a temp path for the registry JSON file.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path inside tmp_path that does not yet exist.
    """
    return tmp_path / "registry.json"


@pytest.fixture
def registry(registry_path: Path) -> DatasetRegistry:
    """Return a fresh DatasetRegistry backed by a temp file.

    Args:
        registry_path: Temp file path fixture.

    Returns:
        Empty DatasetRegistry instance.
    """
    return DatasetRegistry(registry_path)


@pytest.fixture
def sample_record_kwargs() -> dict:
    """Return keyword arguments for a typical DatasetRegistry.add() call.

    Returns:
        Dict with name, source_path, nc, names, sample_counts.
    """
    return {
        "name": "test-dataset",
        "source_path": "/data/test",
        "nc": 3,
        "names": ["jug", "crimp", "sloper"],
        "sample_counts": {"jug": 10, "crimp": 20, "sloper": 5},
    }


# ============================================================================
# TestComputeDataHash
# ============================================================================


class TestComputeDataHash:
    """Tests for _compute_data_hash()."""

    def test_returns_string(self) -> None:
        """Hash should be a non-empty string."""
        h = _compute_data_hash("/data/ds", {"jug": 10})
        assert isinstance(h, str)
        assert len(h) > 0

    def test_deterministic(self) -> None:
        """Same inputs should always produce the same hash."""
        h1 = _compute_data_hash("/data/ds", {"jug": 10, "crimp": 5})
        h2 = _compute_data_hash("/data/ds", {"jug": 10, "crimp": 5})
        assert h1 == h2

    def test_order_independent(self) -> None:
        """Dict insertion order should not affect the hash."""
        h1 = _compute_data_hash("/data/ds", {"jug": 10, "crimp": 5})
        h2 = _compute_data_hash("/data/ds", {"crimp": 5, "jug": 10})
        assert h1 == h2

    def test_different_paths_produce_different_hashes(self) -> None:
        """Different source_path should produce different hash."""
        h1 = _compute_data_hash("/data/a", {"jug": 10})
        h2 = _compute_data_hash("/data/b", {"jug": 10})
        assert h1 != h2

    def test_different_counts_produce_different_hashes(self) -> None:
        """Different sample_counts should produce different hash."""
        h1 = _compute_data_hash("/data/ds", {"jug": 10})
        h2 = _compute_data_hash("/data/ds", {"jug": 11})
        assert h1 != h2

    def test_sha256_hex_length(self) -> None:
        """SHA-256 hex digest should be 64 characters."""
        h = _compute_data_hash("/path", {"a": 1})
        assert len(h) == 64


# ============================================================================
# TestDatasetRegistryInit
# ============================================================================


class TestDatasetRegistryInit:
    """Tests for DatasetRegistry.__init__."""

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        """DatasetRegistry should accept a Path object."""
        reg = DatasetRegistry(tmp_path / "reg.json")
        assert isinstance(reg, DatasetRegistry)

    def test_accepts_string(self, tmp_path: Path) -> None:
        """DatasetRegistry should accept a plain string path."""
        reg = DatasetRegistry(str(tmp_path / "reg.json"))
        assert isinstance(reg, DatasetRegistry)

    def test_default_registry_path_constant(self) -> None:
        """DEFAULT_REGISTRY_PATH should be the expected path."""
        assert DEFAULT_REGISTRY_PATH == Path("data/dataset_registry.json")

    def test_empty_registry_before_first_write(self, registry: DatasetRegistry) -> None:
        """A fresh registry should have no records."""
        assert registry.list_datasets() == []


# ============================================================================
# TestDatasetRegistryAdd
# ============================================================================


class TestDatasetRegistryAdd:
    """Tests for DatasetRegistry.add()."""

    def test_returns_dataset_record(
        self, registry: DatasetRegistry, sample_record_kwargs: dict
    ) -> None:
        """add() should return a DatasetRecord."""
        record = registry.add(**sample_record_kwargs)
        assert isinstance(record, DatasetRecord)

    def test_record_id_is_string(
        self, registry: DatasetRegistry, sample_record_kwargs: dict
    ) -> None:
        """record_id should be a non-empty string (UUID4)."""
        record = registry.add(**sample_record_kwargs)
        assert isinstance(record.record_id, str)
        assert len(record.record_id) > 0

    def test_record_fields_match_inputs(
        self, registry: DatasetRegistry, sample_record_kwargs: dict
    ) -> None:
        """Returned record should reflect the inputs provided."""
        record = registry.add(**sample_record_kwargs)
        assert record.name == sample_record_kwargs["name"]
        assert record.source_path == sample_record_kwargs["source_path"]
        assert record.nc == sample_record_kwargs["nc"]
        assert record.names == sample_record_kwargs["names"]
        assert record.sample_counts == sample_record_kwargs["sample_counts"]

    def test_record_has_added_at_timestamp(
        self, registry: DatasetRegistry, sample_record_kwargs: dict
    ) -> None:
        """Record should have a non-empty added_at ISO timestamp."""
        record = registry.add(**sample_record_kwargs)
        assert isinstance(record.added_at, str)
        assert len(record.added_at) > 0

    def test_record_has_data_hash(
        self, registry: DatasetRegistry, sample_record_kwargs: dict
    ) -> None:
        """Record should have a 64-char SHA-256 data_hash."""
        record = registry.add(**sample_record_kwargs)
        assert isinstance(record.data_hash, str)
        assert len(record.data_hash) == 64

    def test_creates_registry_file(
        self, registry: DatasetRegistry, registry_path: Path, sample_record_kwargs: dict
    ) -> None:
        """Registry file should be created on first add()."""
        assert not registry_path.exists()
        registry.add(**sample_record_kwargs)
        assert registry_path.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Registry should create nested parent directories if needed."""
        deep_path = tmp_path / "a" / "b" / "c" / "registry.json"
        reg = DatasetRegistry(deep_path)
        reg.add(
            name="ds",
            source_path="/data",
            nc=1,
            names=["jug"],
            sample_counts={"jug": 5},
        )
        assert deep_path.exists()

    def test_multiple_adds_increase_count(self, registry: DatasetRegistry) -> None:
        """Each add() with a unique dataset should increase the record count."""
        registry.add("ds1", "/data/1", 1, ["jug"], {"jug": 5})
        registry.add("ds2", "/data/2", 1, ["crimp"], {"crimp": 3})
        assert len(registry.list_datasets()) == 2

    def test_duplicate_fingerprint_emits_warning(
        self, registry: DatasetRegistry, sample_record_kwargs: dict
    ) -> None:
        """Registering the same dataset twice should emit a UserWarning."""
        registry.add(**sample_record_kwargs)
        with pytest.warns(UserWarning, match="already registered"):
            registry.add(**sample_record_kwargs)

    def test_duplicate_fingerprint_does_not_add_new_record(
        self, registry: DatasetRegistry, sample_record_kwargs: dict
    ) -> None:
        """Duplicate fingerprint should NOT add a second record."""
        registry.add(**sample_record_kwargs)
        with pytest.warns(UserWarning):
            registry.add(**sample_record_kwargs)
        assert len(registry.list_datasets()) == 1

    def test_duplicate_fingerprint_returns_existing_record(
        self, registry: DatasetRegistry, sample_record_kwargs: dict
    ) -> None:
        """Duplicate add() should return the original record (same record_id)."""
        original = registry.add(**sample_record_kwargs)
        with pytest.warns(UserWarning):
            duplicate = registry.add(**sample_record_kwargs)
        assert duplicate.record_id == original.record_id


# ============================================================================
# TestDatasetRegistryListDatasets
# ============================================================================


class TestDatasetRegistryListDatasets:
    """Tests for DatasetRegistry.list_datasets()."""

    def test_empty_before_any_add(self, registry: DatasetRegistry) -> None:
        """Empty registry should return an empty list."""
        assert registry.list_datasets() == []

    def test_returns_list(
        self, registry: DatasetRegistry, sample_record_kwargs: dict
    ) -> None:
        """list_datasets() should return a list."""
        registry.add(**sample_record_kwargs)
        result = registry.list_datasets()
        assert isinstance(result, list)

    def test_returns_all_records_in_insertion_order(
        self, registry: DatasetRegistry
    ) -> None:
        """Records should be returned in insertion order."""
        registry.add("ds1", "/data/1", 1, ["jug"], {"jug": 5})
        registry.add("ds2", "/data/2", 1, ["crimp"], {"crimp": 3})
        registry.add("ds3", "/data/3", 1, ["sloper"], {"sloper": 7})
        names = [r.name for r in registry.list_datasets()]
        assert names == ["ds1", "ds2", "ds3"]

    def test_each_item_is_dataset_record(
        self, registry: DatasetRegistry, sample_record_kwargs: dict
    ) -> None:
        """Each item in the list should be a DatasetRecord."""
        registry.add(**sample_record_kwargs)
        for r in registry.list_datasets():
            assert isinstance(r, DatasetRecord)


# ============================================================================
# TestDatasetRegistryGetById
# ============================================================================


class TestDatasetRegistryGetById:
    """Tests for DatasetRegistry.get_by_id()."""

    def test_returns_none_for_nonexistent_id(self, registry: DatasetRegistry) -> None:
        """get_by_id() should return None for an unknown record_id."""
        result = registry.get_by_id("nonexistent-id")
        assert result is None

    def test_returns_correct_record(
        self, registry: DatasetRegistry, sample_record_kwargs: dict
    ) -> None:
        """get_by_id() should return the record matching the given ID."""
        added = registry.add(**sample_record_kwargs)
        found = registry.get_by_id(added.record_id)
        assert found is not None
        assert found.record_id == added.record_id
        assert found.name == added.name

    def test_returns_none_after_record_removed(
        self, registry: DatasetRegistry, sample_record_kwargs: dict
    ) -> None:
        """get_by_id() should return None once the record has been removed."""
        added = registry.add(**sample_record_kwargs)
        registry.remove(added.record_id)
        assert registry.get_by_id(added.record_id) is None

    def test_finds_correct_record_among_multiple(
        self, registry: DatasetRegistry
    ) -> None:
        """get_by_id() should find the correct record among several."""
        r1 = registry.add("ds1", "/data/1", 1, ["jug"], {"jug": 5})
        r2 = registry.add("ds2", "/data/2", 1, ["crimp"], {"crimp": 3})
        assert registry.get_by_id(r1.record_id).name == "ds1"  # type: ignore[union-attr]
        assert registry.get_by_id(r2.record_id).name == "ds2"  # type: ignore[union-attr]


# ============================================================================
# TestDatasetRegistryRemove
# ============================================================================


class TestDatasetRegistryRemove:
    """Tests for DatasetRegistry.remove()."""

    def test_returns_false_for_nonexistent_id(self, registry: DatasetRegistry) -> None:
        """Removing a nonexistent record should return False (no-op)."""
        result = registry.remove("nonexistent-id")
        assert result is False

    def test_no_op_does_not_raise(self, registry: DatasetRegistry) -> None:
        """Removing a nonexistent record should not raise any exception."""
        registry.remove("nonexistent-id")  # Must not raise

    def test_returns_true_for_existing_record(
        self, registry: DatasetRegistry, sample_record_kwargs: dict
    ) -> None:
        """Removing an existing record should return True."""
        added = registry.add(**sample_record_kwargs)
        result = registry.remove(added.record_id)
        assert result is True

    def test_record_no_longer_present_after_removal(
        self, registry: DatasetRegistry, sample_record_kwargs: dict
    ) -> None:
        """Removed record should not appear in list_datasets()."""
        added = registry.add(**sample_record_kwargs)
        registry.remove(added.record_id)
        assert all(r.record_id != added.record_id for r in registry.list_datasets())

    def test_removes_only_target_record(self, registry: DatasetRegistry) -> None:
        """Removing one record should leave others intact."""
        r1 = registry.add("ds1", "/data/1", 1, ["jug"], {"jug": 5})
        r2 = registry.add("ds2", "/data/2", 1, ["crimp"], {"crimp": 3})
        registry.remove(r1.record_id)
        remaining = registry.list_datasets()
        assert len(remaining) == 1
        assert remaining[0].record_id == r2.record_id

    def test_double_remove_returns_false_second_time(
        self, registry: DatasetRegistry, sample_record_kwargs: dict
    ) -> None:
        """Second remove on same record_id should return False."""
        added = registry.add(**sample_record_kwargs)
        registry.remove(added.record_id)
        result = registry.remove(added.record_id)
        assert result is False


# ============================================================================
# TestDatasetRegistryGetCombinedCounts
# ============================================================================


class TestDatasetRegistryGetCombinedCounts:
    """Tests for DatasetRegistry.get_combined_counts()."""

    def test_returns_empty_dict_for_empty_registry(
        self, registry: DatasetRegistry
    ) -> None:
        """Empty registry should return an empty dict."""
        assert registry.get_combined_counts() == {}

    def test_sums_counts_across_datasets(self, registry: DatasetRegistry) -> None:
        """Counts from multiple datasets should be summed per class."""
        registry.add("ds1", "/data/1", 2, ["jug", "crimp"], {"jug": 10, "crimp": 20})
        registry.add("ds2", "/data/2", 2, ["jug", "crimp"], {"jug": 5, "crimp": 3})
        totals = registry.get_combined_counts()
        assert totals["jug"] == 15
        assert totals["crimp"] == 23

    def test_names_filter_restricts_output(self, registry: DatasetRegistry) -> None:
        """names_filter should include only the specified classes."""
        registry.add(
            "ds1",
            "/data/1",
            3,
            ["jug", "crimp", "sloper"],
            {"jug": 10, "crimp": 20, "sloper": 5},
        )
        totals = registry.get_combined_counts(names_filter=["jug"])
        assert "jug" in totals
        assert "crimp" not in totals
        assert "sloper" not in totals

    def test_names_filter_none_includes_all(self, registry: DatasetRegistry) -> None:
        """names_filter=None should include all classes."""
        registry.add(
            "ds1",
            "/data/1",
            2,
            ["jug", "crimp"],
            {"jug": 10, "crimp": 20},
        )
        totals = registry.get_combined_counts(names_filter=None)
        assert "jug" in totals
        assert "crimp" in totals

    def test_classes_only_in_some_datasets(self, registry: DatasetRegistry) -> None:
        """Classes that appear only in some datasets should have their count only."""
        registry.add("ds1", "/data/1", 1, ["jug"], {"jug": 7})
        registry.add("ds2", "/data/2", 1, ["crimp"], {"crimp": 4})
        totals = registry.get_combined_counts()
        assert totals.get("jug") == 7
        assert totals.get("crimp") == 4


# ============================================================================
# TestDatasetRegistryPersistence
# ============================================================================


class TestDatasetRegistryPersistence:
    """Tests for JSON persistence across registry instances."""

    def test_records_persist_across_instances(
        self, registry_path: Path, sample_record_kwargs: dict
    ) -> None:
        """Records added in one instance should be visible in a new instance."""
        reg1 = DatasetRegistry(registry_path)
        reg1.add(**sample_record_kwargs)

        reg2 = DatasetRegistry(registry_path)
        records = reg2.list_datasets()
        assert len(records) == 1
        assert records[0].name == sample_record_kwargs["name"]

    def test_registry_file_is_valid_json(
        self, registry: DatasetRegistry, registry_path: Path, sample_record_kwargs: dict
    ) -> None:
        """Registry file should be parseable JSON."""
        registry.add(**sample_record_kwargs)
        with open(registry_path, "r") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_json_contains_expected_keys(
        self, registry: DatasetRegistry, registry_path: Path, sample_record_kwargs: dict
    ) -> None:
        """Each JSON entry should contain all DatasetRecord fields."""
        registry.add(**sample_record_kwargs)
        with open(registry_path, "r") as f:
            data = json.load(f)
        entry = data[0]
        for field in (
            "record_id",
            "name",
            "source_path",
            "nc",
            "names",
            "sample_counts",
            "added_at",
            "data_hash",
        ):
            assert field in entry, f"Missing field: {field}"

    def test_corrupted_json_returns_empty_list(self, registry_path: Path) -> None:
        """A corrupt JSON file should yield an empty list (no crash)."""
        registry_path.write_text("not valid json")
        reg = DatasetRegistry(registry_path)
        assert reg.list_datasets() == []

    def test_remove_persists_across_instances(
        self, registry_path: Path, sample_record_kwargs: dict
    ) -> None:
        """Removal should be visible in a subsequent registry instance."""
        reg1 = DatasetRegistry(registry_path)
        added = reg1.add(**sample_record_kwargs)
        reg1.remove(added.record_id)

        reg2 = DatasetRegistry(registry_path)
        assert reg2.list_datasets() == []
