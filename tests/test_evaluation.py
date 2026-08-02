from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
from evaluation.main import build_training_table, leakage_groups, normalized_template
from src.data_loader import DatasetBundle


def test_template_normalization_removes_variable_numbers_and_urls():
    assert normalized_template("Pay 42 at https://example.test/a") == "pay <num> at <url>"
    assert normalized_template("Pay 99 at https://other.test/b") == "pay <num> at <url>"


def test_training_table_has_labels_features_and_leakage_groups():
    frame = build_training_table(DatasetBundle.load(Path("dataset")))
    assert len(frame) == 412
    assert set(frame.weak_action) == {"notify", "digest", "mute"}
    assert {"template", "leakage_group", "group_type", "business_category", "load_notifications_sent"} <= set(frame)
    assert frame.leakage_group.nunique() > 5
