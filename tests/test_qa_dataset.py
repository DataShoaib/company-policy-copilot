import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "data" / "eval"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from qa_dataset import QA_ITEMS, dataset, get_subset


class TestDatasetStructure:
    def test_dataset_not_empty(self):
        assert len(QA_ITEMS) > 0

    def test_question_answer_lists_same_length(self):
        assert len(dataset["question"]) == len(dataset["answer"])

    def test_all_items_have_unique_ids(self):
        ids = [item.id for item in QA_ITEMS]
        assert len(ids) == len(set(ids))

    def test_all_question_types_are_valid(self):
        valid = {"factual", "numeric", "multi_hop", "paraphrase", "unanswerable"}
        for item in QA_ITEMS:
            assert item.question_type in valid

    def test_all_difficulties_are_valid(self):
        valid = {"easy", "medium", "hard"}
        for item in QA_ITEMS:
            assert item.difficulty in valid

    def test_unanswerable_items_have_no_source_doc(self):
        for item in QA_ITEMS:
            if item.question_type == "unanswerable":
                assert item.source_doc == "none"


class TestGroundTruthAccuracy:
    def test_casual_leave_is_12_days_not_18(self):
        item = next(i for i in QA_ITEMS if i.id == "leave-01")
        assert "12 days" in item.ground_truth
        assert "18 days" not in item.ground_truth

    def test_maternity_leave_is_26_weeks(self):
        item = next(i for i in QA_ITEMS if i.id == "leave-03")
        assert "26 weeks" in item.ground_truth

    def test_paternity_leave_is_15_days(self):
        item = next(i for i in QA_ITEMS if i.id == "leave-07")
        assert "15 days" in item.ground_truth

    def test_pf_employer_contribution_is_12_percent(self):
        item = next(i for i in QA_ITEMS if i.id == "comp-01")
        assert "12%" in item.ground_truth


class TestGetSubset:
    def test_filter_by_question_type(self):
        numeric = get_subset(question_type="numeric")
        assert all(i.question_type == "numeric" for i in numeric)
        assert len(numeric) > 0

    def test_filter_by_category(self):
        leave = get_subset(category="leave")
        assert all(i.category == "leave" for i in leave)

    def test_filter_by_multiple_criteria(self):
        hard_numeric = get_subset(question_type="numeric", difficulty="hard")
        assert all(i.question_type == "numeric" and i.difficulty == "hard" for i in hard_numeric)

    def test_unmatched_filter_returns_empty(self):
        assert get_subset(category="nonexistent_category") == []
