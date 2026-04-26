"""Tests for HuggingFace Hub integration."""

import pytest

from robodata.hub.url import HFDatasetRef, is_hf_url, parse_hf_url


class TestIsHfUrl:
    """Test is_hf_url function."""

    def test_hf_scheme(self):
        assert is_hf_url("hf://lerobot/pusht") is True
        assert is_hf_url("hf://openvla/modified_libero_rlds") is True

    def test_huggingface_scheme(self):
        assert is_hf_url("huggingface://lerobot/pusht") is True

    def test_https_url(self):
        assert is_hf_url("https://huggingface.co/datasets/lerobot/pusht") is True

    def test_local_path(self):
        assert is_hf_url("/path/to/dataset") is False
        assert is_hf_url("./dataset") is False
        assert is_hf_url("dataset") is False

    def test_non_string(self):
        assert is_hf_url(None) is False  # type: ignore
        assert is_hf_url(123) is False  # type: ignore


class TestParseHfUrl:
    """Test parse_hf_url function."""

    def test_hf_scheme_basic(self):
        ref = parse_hf_url("hf://lerobot/pusht")
        assert ref.repo_id == "lerobot/pusht"
        assert ref.revision is None
        assert ref.subset is None

    def test_hf_scheme_with_revision(self):
        ref = parse_hf_url("hf://lerobot/pusht@main")
        assert ref.repo_id == "lerobot/pusht"
        assert ref.revision == "main"
        assert ref.subset is None

    def test_hf_scheme_with_commit(self):
        ref = parse_hf_url("hf://lerobot/pusht@abc123")
        assert ref.repo_id == "lerobot/pusht"
        assert ref.revision == "abc123"

    def test_huggingface_scheme(self):
        ref = parse_hf_url("huggingface://openvla/modified_libero_rlds")
        assert ref.repo_id == "openvla/modified_libero_rlds"

    def test_https_url(self):
        ref = parse_hf_url("https://huggingface.co/datasets/lerobot/pusht")
        assert ref.repo_id == "lerobot/pusht"

    def test_https_url_with_trailing_slash(self):
        ref = parse_hf_url("https://huggingface.co/datasets/lerobot/pusht/")
        assert ref.repo_id == "lerobot/pusht"

    def test_invalid_scheme(self):
        with pytest.raises(ValueError, match="Invalid HuggingFace URL scheme"):
            parse_hf_url("http://example.com/dataset")

    def test_invalid_repo_id_format(self):
        with pytest.raises(ValueError, match="Invalid repo_id format"):
            parse_hf_url("hf://just_one_part")


class TestHFDatasetRef:
    """Test HFDatasetRef dataclass."""

    def test_basic_ref(self):
        ref = HFDatasetRef(repo_id="lerobot/pusht")
        assert ref.repo_id == "lerobot/pusht"
        assert ref.revision is None
        assert ref.subset is None

    def test_ref_with_all_fields(self):
        ref = HFDatasetRef(
            repo_id="org/dataset",
            revision="v1.0",
            subset="train",
        )
        assert ref.repo_id == "org/dataset"
        assert ref.revision == "v1.0"
        assert ref.subset == "train"
