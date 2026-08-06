"""Verification without ground truth."""

from __future__ import annotations

import pytest

from router_agent import live, verifiers


class TestCanonicalAnswer:
    def test_math_uses_boxed_protocol(self):
        a = verifiers._canonical_answer("blah $\\boxed{42}$ blah", "math")
        b = verifiers._canonical_answer("different words, $\\boxed{42}$", "math")
        assert a == b == "42"

    def test_math_distinguishes_answers(self):
        a = verifiers._canonical_answer("$\\boxed{42}$", "math")
        b = verifiers._canonical_answer("$\\boxed{43}$", "math")
        assert a != b

    def test_code_ignores_formatting_and_comments(self):
        """Two draws differing only in whitespace or comments are one program.

        Counting them as disagreement would make the verifier escalate on
        cosmetic noise, which is expensive and wrong.
        """
        a = verifiers._canonical_answer(
            "```python\ndef f(x):\n    return x  # doubles\n```", "code"
        )
        b = verifiers._canonical_answer(
            "```python\ndef f(x):\n        return x\n```", "code"
        )
        assert a == b

    def test_code_distinguishes_logic(self):
        a = verifiers._canonical_answer("```python\ndef f(x): return x\n```", "code")
        b = verifiers._canonical_answer("```python\ndef f(x): return -x\n```", "code")
        assert a != b

    def test_empty_is_none(self):
        assert verifiers._canonical_answer("", "math") is None
        assert verifiers._canonical_answer("   ", "general") is None


class TestSelect:
    def test_one_shot_policies_do_not_verify(self, cfg):
        """The fix that keeps the cost comparison honest.

        A predictive router cannot act on a verification result, so charging it
        the cascade's verification cost would make cascading look artificially
        competitive - and that comparison is the repository's whole subject.
        """
        task = live.synthesize_task("What is 2+2?", domain="math")
        for policy in ("predictive", "always_cheap", "always_expensive"):
            assert verifiers.select(task, cfg(policy=policy)) == "none"

    def test_cascade_gets_self_consistency_by_default(self, cfg):
        task = live.synthesize_task("What is 2+2?", domain="math")
        assert verifiers.select(task, cfg(policy="cascade")) == "self_consistency"

    def test_tests_chosen_only_when_execution_allowed(self, cfg):
        task = live.synthesize_task(
            "reverse a list", domain="code", tests=["assert f([1]) == [1]"]
        )
        assert verifiers.select(
            task, cfg(policy="cascade", allow_code_execution=True)
        ) == "tests"
        # Default is off, and the fallback is the proxy rather than a refusal.
        assert verifiers.select(
            task, cfg(policy="cascade", allow_code_execution=False)
        ) == "self_consistency"

    def test_explicit_verifier_overrides(self, cfg):
        task = live.synthesize_task("What is 2+2?", domain="math")
        assert verifiers.select(
            task, cfg(policy="always_cheap", verifier="self_consistency")
        ) == "self_consistency"


class TestVerifyNone:
    def test_accepts_and_costs_nothing(self, cfg):
        task = live.synthesize_task("hi", domain="general")
        check = verifiers.verify_none(task, "some answer", "cheap", cfg())
        assert check.accepted is True
        assert check.cost_usd == 0.0
        assert check.confidence is None


class TestVerifyTests:
    def test_refuses_without_opt_in(self, cfg):
        """Executing model-generated code must be opt-in, not a default."""
        task = live.synthesize_task(
            "f", domain="code", tests=["assert f(1) == 1"]
        )
        with pytest.raises(PermissionError, match="disabled by default"):
            verifiers.verify_tests(task, "```python\ndef f(x): return x\n```",
                                   "cheap", cfg(allow_code_execution=False))

    def test_requires_tests(self, cfg):
        task = live.synthesize_task("f", domain="code")
        with pytest.raises(ValueError, match="needs tests"):
            verifiers.verify_tests(task, "code", "cheap",
                                   cfg(allow_code_execution=True))

    def test_passing_code_is_accepted(self, cfg):
        task = live.synthesize_task(
            "double", domain="code", tests=["assert f(2) == 4"]
        )
        check = verifiers.verify_tests(
            task, "```python\ndef f(x):\n    return x * 2\n```",
            "cheap", cfg(allow_code_execution=True),
        )
        assert check.accepted is True
        assert check.cost_usd == 0.0  # the perfect verifier is free

    def test_failing_code_is_rejected(self, cfg):
        task = live.synthesize_task(
            "double", domain="code", tests=["assert f(2) == 4"]
        )
        check = verifiers.verify_tests(
            task, "```python\ndef f(x):\n    return x + 1\n```",
            "cheap", cfg(allow_code_execution=True),
        )
        assert check.accepted is False


class TestSelfConsistency:
    def test_unverifiable_when_tier_refuses_temperature(self, cfg, use_ladder):
        """Sonnet 5 and Opus 5 reject `temperature`, so they cannot be resampled.

        The verifier must say "not measured" rather than "unanimous" - a
        verifier that always accepts is worse than no verifier, because it is
        invisible.
        """
        use_ladder("claude")
        from router_agent import verifiers as v
        task = live.synthesize_task("What is 2+2?", domain="math")
        check = v.verify_self_consistency(
            task, "$\\boxed{4}$", "expensive", cfg()
        )
        assert check.accepted is False
        assert check.confidence is None
        assert check.detail.get("unverifiable") is True
        assert check.cost_usd == 0.0

    def test_measures_agreement_on_a_temperature_tier(self, cfg, use_ladder):
        use_ladder("deepseek")
        from router_agent import verifiers as v
        task = live.synthesize_task("What is 2+2?", domain="math")
        check = v.verify_self_consistency(
            task, "$\\boxed{A}$", "cheap", cfg(self_consistency_k=5)
        )
        assert check.confidence is not None
        assert 0.0 <= check.confidence <= 1.0
        assert check.detail["k"] == 5
        # k-1 extra draws were paid for.
        assert check.cost_usd > 0.0
