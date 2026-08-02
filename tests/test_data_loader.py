from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path("code").resolve()))
from src.data_loader import DatasetBundle, DatasetValidationError


@pytest.fixture
def dataset_copy(tmp_path):
    target = tmp_path / "dataset"
    shutil.copytree("dataset", target)
    return target


def rewrite(root, table, change):
    path = root / f"{table}.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    change(frame)
    frame.to_csv(path, index=False)


def assert_error(root, code):
    with pytest.raises(DatasetValidationError) as caught:
        DatasetBundle.load(root)
    assert code in {issue.code for issue in caught.value.report.errors}
    assert caught.value.report.as_dict()["valid"] is False


def test_missing_required_column(dataset_copy):
    rewrite(dataset_copy, "users", lambda frame: frame.drop(columns=["user_id"], inplace=True))
    assert_error(dataset_copy, "missing_columns")


def test_duplicate_primary_id(dataset_copy):
    rewrite(dataset_copy, "messages", lambda frame: frame.__setitem__("message_id", [frame.message_id.iloc[0]] * 2 + frame.message_id.iloc[2:].tolist()))
    assert_error(dataset_copy, "duplicate_key")


def test_invalid_boolean_domain(dataset_copy):
    rewrite(dataset_copy, "business_accounts", lambda frame: frame.__setitem__("verified", ["yes"] + frame.verified.iloc[1:].tolist()))
    assert_error(dataset_copy, "invalid_boolean")


def test_malformed_timestamp(dataset_copy):
    rewrite(dataset_copy, "messages", lambda frame: frame.__setitem__("created_at", ["not-a-date"] + frame.created_at.iloc[1:].tolist()))
    assert_error(dataset_copy, "invalid_timestamp")


def test_media_path_cannot_escape_dataset(dataset_copy):
    rewrite(dataset_copy, "images", lambda frame: frame.__setitem__("file_path", ["../../etc/passwd"] + frame.file_path.iloc[1:].tolist()))
    assert_error(dataset_copy, "media_path_traversal")


def test_missing_mapped_media_file(dataset_copy):
    mapped = pd.read_csv(dataset_copy / "images.csv", dtype=str).file_path.iloc[0]
    (dataset_copy / mapped).unlink()
    assert_error(dataset_copy, "missing_media_file")


def test_media_id_must_match_declared_type(dataset_copy):
    def change(frame):
        row = frame.index[frame.media_type.eq("image")][0]
        frame.loc[row, "media_id"] = pd.read_csv(dataset_copy / "voice_notes.csv", dtype=str).voice_note_id.iloc[0]

    rewrite(dataset_copy, "messages", change)
    assert_error(dataset_copy, "media_type_mismatch")


def test_invalid_conversation_type(dataset_copy):
    rewrite(dataset_copy, "messages", lambda frame: frame.__setitem__("conversation_type", ["channel"] + frame.conversation_type.iloc[1:].tolist()))
    assert_error(dataset_copy, "invalid_category")


def test_absent_optional_business_relationship_is_warning(dataset_copy):
    messages = pd.read_csv(dataset_copy / "messages.csv", dtype=str, keep_default_na=False)
    message = messages[messages.conversation_type.eq("business")].iloc[0]

    def remove_relationship(frame):
        drop = frame.user_id.eq(message.user_id) & frame.business_id.eq(message.business_id)
        frame.drop(frame.index[drop], inplace=True)

    rewrite(dataset_copy, "user_business_history", remove_relationship)
    bundle = DatasetBundle.load(dataset_copy)
    warnings = bundle.validation_report.warnings
    assert any(issue.code == "missing_optional_relationship" and issue.table == "messages" for issue in warnings)


def test_incoming_conversation_specific_nullability(dataset_copy):
    def change(frame):
        row = frame.index[frame.conversation_type.eq("business")][0]
        frame.loc[row, "sender_user_id"] = frame.loc[0, "user_id"]

    rewrite(dataset_copy, "messages", change)
    assert_error(dataset_copy, "conversation_nullability")
