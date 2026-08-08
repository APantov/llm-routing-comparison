"""The bridge from an arbitrary query to a task dict."""

from __future__ import annotations

import pytest

from router_agent import live


class TestInferDomain:
    @pytest.mark.parametrize("query", [
        "Write a python function to reverse a list",
        "Debug this traceback for me",
        "```\nfor x in y:\n  pass\n```",
        "Refactor this SQL query",
    ])
    def test_code(self, query):
        assert live.infer_domain(query) == "code"

    @pytest.mark.parametrize("query", [
        "Solve for x: 2x + 5 = 13",
        "Evaluate $\\int_0^1 x^2 dx$",
        "Prove the Pythagorean theorem",
        "What is 17 * 23?",
    ])
    def test_math(self, query):
        assert live.infer_domain(query) == "math"

    @pytest.mark.parametrize("query", [
        "Summarise this email for me",
        "What is the capital of France?",
        "Write a haiku about autumn",
    ])
    def test_general(self, query):
        assert live.infer_domain(query) == "general"

    def test_supplied_tests_are_conclusive(self):
        # Prose says maths, asserts say code. The asserts win, because a caller
        # who supplied them is unambiguously asking for a function.
        assert live.infer_domain(
            "Solve for x", tests=["assert f(1) == 1"]
        ) == "code"


class TestSynthesizeTask:
    def test_id_is_stable_across_calls(self):
        a = live.synthesize_task("What is 2+2?", domain="math")
        b = live.synthesize_task("What is 2+2?", domain="math")
        assert a["id"] == b["id"]

    def test_id_varies_with_query_and_domain(self):
        a = live.synthesize_task("What is 2+2?", domain="math")
        b = live.synthesize_task("What is 3+3?", domain="math")
        c = live.synthesize_task("What is 2+2?", domain="general")
        assert a["id"] != b["id"] != c["id"] and a["id"] != c["id"]

    def test_carries_no_ground_truth(self):
        """The property the whole serving layer depends on.

        If an `answer` key ever appeared here, `graders.grade` would start
        silently returning verdicts for served queries and the distinction
        between `verified` and `correct` would collapse.
        """
        task = live.synthesize_task("What is 2+2?", domain="math")
        assert "answer" not in task["grader_payload"]
        assert task.get("grader") is None

    def test_marks_itself_live(self):
        task = live.synthesize_task("hello", domain="general")
        assert task["_live"] is True

    def test_no_difficulty_label(self):
        """A live query has no `level`, and must not pretend to."""
        task = live.synthesize_task("Solve x^2 = 4", domain="math")
        assert "level" not in task["predict_features"]

    def test_tests_are_attached(self):
        task = live.synthesize_task(
            "reverse a list", domain="code", tests=["assert f([1]) == [1]"]
        )
        assert task["grader"] == "test_program"
        assert task["grader_payload"]["tests"] == ["assert f([1]) == [1]"]
        assert task["predict_features"]["n_asserts"] == 1

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            live.synthesize_task("   ")

    def test_rejects_unknown_domain(self):
        with pytest.raises(ValueError):
            live.synthesize_task("hi", domain="physics")


class TestPredictIsHardLive:
    def test_long_prompt_is_hard(self):
        task = live.synthesize_task("x" * 200, domain="general")
        assert live.predict_is_hard_live(task) is True

    def test_short_simple_prompt_is_easy(self):
        task = live.synthesize_task("What is 2+2?", domain="math")
        assert live.predict_is_hard_live(task) is False

    def test_marker_word_is_hard(self):
        task = live.synthesize_task("prove it", domain="math")
        assert live.predict_is_hard_live(task) is True

    def test_a_difficulty_label_has_no_effect(self):
        """A shipped `level` must not change the verdict. The inverse of the
        test this replaces, and it encodes why the benchmark policy was deleted.

        Until 8 August 2026 this function delegated to policies.predict_is_hard
        whenever a `level` was present, so the "query text only" promise held
        only for queries that happened to lack the field. That predicate read a
        difficulty annotation written by someone who had already solved the
        problem, and under MIN_MATH_LEVEL = 5 it was a constant - which is what
        made the `predictive` policy `always_expensive` on the whole maths half.

        A serving router sees query text. Adding a label must be inert.
        """
        task = live.synthesize_task("What is 2+2?", domain="math")
        baseline = live.predict_is_hard_live(task)
        assert baseline is False

        for level in (1, 3, 5, 99):
            labelled = dict(task)
            labelled["predict_features"] = {**task.get("predict_features", {}),
                                            "level": level}
            assert live.predict_is_hard_live(labelled) is baseline, level

    def test_benchmark_tasks_are_routed_on_text_alone(self, benchmark_task):
        """Real evaluation tasks carry a level; it must still be ignored."""
        maths = [t for t in benchmark_task if t["domain"] == "math"]
        if not maths:
            import pytest as _pytest
            _pytest.skip("no math tasks")
        for task in maths[:20]:
            stripped = dict(task)
            stripped["predict_features"] = {
                k: v for k, v in task.get("predict_features", {}).items()
                if k != "level"
            }
            assert live.predict_is_hard_live(task) == live.predict_is_hard_live(stripped)
