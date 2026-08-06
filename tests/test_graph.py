"""The routing graph: the cycle, the guards, and the accounting.

The tests that matter most here are the accounting ones. A cascade is a loop,
and the characteristic bug in a loop with accumulated state is that the second
lap overwrites the first lap's spend instead of adding to it - which produces
a router that under-reports its own cost. On a project whose subject is what
routing costs, that is the bug worth having tests for.
"""

from __future__ import annotations

import pytest

from router_agent.engine import route


class TestCascadeMechanics:
    def test_cascade_starts_at_the_cheapest_rung(self, cfg):
        import models
        out = route("What is 2+2?", cfg=cfg(policy="cascade"))
        assert out.events[0]["data"]["start_tier"] == models.TIERS[0]

    def test_always_expensive_starts_at_the_top(self, cfg):
        import models
        out = route("What is 2+2?", cfg=cfg(policy="always_expensive"))
        assert out.final_tier == models.TIERS[-1]

    def test_one_shot_policies_never_escalate(self, cfg):
        for policy in ("predictive", "always_cheap", "always_expensive"):
            out = route("What is 2+2?", cfg=cfg(policy=policy,
                                                agreement_threshold=1.0))
            assert out.escalations == 0, policy

    def test_one_shot_policies_make_exactly_one_call(self, cfg):
        """No verification means one call, which is what makes them cheap."""
        for policy in ("predictive", "always_cheap", "always_expensive"):
            out = route("What is 2+2?", cfg=cfg(policy=policy))
            assert len(out.calls) == 1, policy
            assert out.verifier == "none", policy

    def test_cascade_escalates_when_verification_fails(self, cfg, use_ladder):
        use_ladder("claude")
        from router_agent.engine import route as r
        # Unanimity is unachievable on a tier that refuses temperature, so the
        # cascade is forced up the ladder.
        out = r("Compute the integral of x^2", cfg=cfg(
            policy="cascade", agreement_threshold=1.0, max_cost_usd=10.0))
        assert out.escalations >= 1
        assert out.final_tier != "cheap"


class TestAccounting:
    def test_cost_accumulates_across_the_cycle(self, cfg, use_ladder):
        """The reducer test. Total must equal the sum of the parts."""
        use_ladder("claude")
        from router_agent.engine import route as r
        out = r("Compute the integral of x^2", cfg=cfg(
            policy="cascade", agreement_threshold=1.0, max_cost_usd=10.0))

        assert out.escalations >= 1, "test needs a run that actually escalated"
        summed = sum(c["cost_usd"] for c in out.calls)
        assert out.cost_usd == pytest.approx(summed, rel=1e-9)

    def test_escalating_costs_more_than_not(self, cfg, use_ladder):
        use_ladder("claude")
        from router_agent.engine import route as r
        cheap = r("What is 2+2?", cfg=cfg(policy="always_cheap"))
        escalated = r("Compute the integral of x^2", cfg=cfg(
            policy="cascade", agreement_threshold=1.0, max_cost_usd=10.0))
        assert escalated.cost_usd > cheap.cost_usd

    def test_every_call_is_recorded(self, cfg, use_ladder):
        use_ladder("claude")
        from router_agent.engine import route as r
        out = r("Compute the integral of x^2", cfg=cfg(
            policy="cascade", agreement_threshold=1.0, max_cost_usd=10.0))
        answer_calls = [c for c in out.calls if c["kind"] == "answer"]
        assert len(answer_calls) == out.escalations + 1

    def test_mock_mode_spends_no_real_money(self, cfg):
        out = route("What is 2+2?", cfg=cfg(policy="cascade"))
        assert out.backend_cost_usd == 0.0
        assert out.simulated is True

    def test_backend_cost_is_summed_per_call(self, cfg, use_ladder):
        """Backend spend is per call, not inferred from the run's mode.

        The bug this pins: reporting the whole attributed cost as backend spend
        whenever a run touched a provider at all. A real run that partially
        replays pays for some calls and reads the rest from cache, so that
        would bill cached calls as money spent - and `cost_usd` versus
        `backend_cost_usd` is exactly the distinction this repository keeps.
        """
        use_ladder("claude")
        from router_agent.engine import route as r
        out = r("Compute the integral of x^2", cfg=cfg(
            policy="cascade", agreement_threshold=1.0, max_cost_usd=10.0))

        assert out.backend_cost_usd == pytest.approx(
            sum(c["backend_cost_usd"] for c in out.calls)
        )
        # Every call carries the field, and it never exceeds what was charged.
        for c in out.calls:
            assert "backend_cost_usd" in c
            assert c["backend_cost_usd"] <= c["cost_usd"] + 1e-12
        assert out.backend_cost_usd <= out.cost_usd + 1e-12


class TestGuards:
    def test_budget_refuses_escalation_before_spending(self, cfg, use_ladder):
        use_ladder("claude")
        from router_agent.engine import route as r
        out = r("Compute the integral of x^2", cfg=cfg(
            policy="cascade", agreement_threshold=1.0, max_cost_usd=0.0005))
        assert out.stop_reason == "budget_exceeded"
        assert out.escalations == 0
        # The guard must fire BEFORE the expensive call, not after it.
        assert out.cost_usd <= 0.0005 + 0.01  # cheap call already paid for
        assert all(c["tier"] == "cheap" for c in out.calls)

    def test_max_escalations_bounds_the_loop(self, cfg, use_ladder):
        use_ladder("claude")
        from router_agent.engine import route as r
        out = r("Compute the integral of x^2", cfg=cfg(
            policy="cascade", agreement_threshold=1.0,
            max_cost_usd=10.0, max_escalations=1))
        assert out.escalations == 1
        assert out.stop_reason == "max_escalations"

    @pytest.mark.parametrize("ladder", ["deepseek", "claude", "wide"])
    def test_loop_always_terminates_on_the_ladder(self, cfg, use_ladder, ladder):
        """The cycle terminates, and never runs off the top of the ladder.

        Asserted as an invariant rather than as a specific escalation path: on
        a ladder whose cheap rung accepts a temperature the mock may reach
        unanimous agreement and legitimately stop at the bottom, so pinning a
        particular final tier would be testing the mock's luck rather than the
        graph's control flow.
        """
        models = use_ladder(ladder)
        from router_agent.engine import route as r

        out = r("Compute the integral of x^2", cfg=cfg(
            policy="cascade", agreement_threshold=1.0, max_cost_usd=10.0))

        assert out.final_tier in models.TIERS
        assert out.escalations <= len(models.TIERS) - 1
        assert out.stop_reason in (
            "verified", "exhausted_ladder", "max_escalations",
            "budget_exceeded", "unverified_final",
        )
        # One answer call per rung visited, and no more.
        answer_calls = [c for c in out.calls if c["kind"] == "answer"]
        assert len(answer_calls) == out.escalations + 1


class TestHumanApproval:
    def test_pauses_and_can_be_denied(self, cfg, use_ladder):
        use_ladder("claude")
        from router_agent.engine import resume, route as r

        c = cfg(policy="cascade", agreement_threshold=1.0,
                max_cost_usd=10.0, require_approval_above_usd=0.0)
        out = r("Compute the integral of x^2", cfg=c)
        assert out.stop_reason == "awaiting_approval"
        assert out.interrupted is not None
        assert "projected_usd" in out.interrupted

        denied = resume(out.thread_id, approved=False, cfg=c)
        assert denied.stop_reason == "approval_denied"
        assert denied.escalations == 0

    def test_approval_lets_the_escalation_proceed(self, cfg, use_ladder):
        use_ladder("claude")
        from router_agent.engine import resume, route as r

        c = cfg(policy="cascade", agreement_threshold=1.0,
                max_cost_usd=10.0, require_approval_above_usd=0.0)
        out = r("Compute the integral of x^2", cfg=c)
        approved = resume(out.thread_id, approved=True, cfg=c)
        assert approved.escalations >= 1


class TestOutcomeContract:
    def test_there_is_no_correct_field(self, cfg):
        """Serving has no ground truth, so it must not report correctness."""
        out = route("What is 2+2?", cfg=cfg())
        assert not hasattr(out, "correct")
        assert "correct" not in out.to_dict()

    def test_verified_meaning_is_always_present(self, cfg):
        out = route("What is 2+2?", cfg=cfg())
        assert out.to_dict()["verified_meaning"]

    def test_unmeasured_agreement_is_not_reported_as_disagreement(
        self, cfg, use_ladder
    ):
        use_ladder("claude")
        from router_agent.engine import route as r
        out = r("What is 2+2?", cfg=cfg(
            policy="always_expensive", verifier="self_consistency"))
        assert out.confidence is None
        assert "could NOT be measured" in out.verified_meaning
