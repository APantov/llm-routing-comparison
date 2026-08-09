"""The findings module, and the guarantee that it never invents a number.

These tests are regression protection for a specific failure this repository
has already had once: on 6 August 2026 a grader bug moved the routable fraction
from 13.0% to 15.0%. Any figure that had been transcribed into source would
have silently kept saying 13% - so `findings` recomputes from the committed
data, and these tests check that the recomputation still matches the documented
result.
"""

from __future__ import annotations

import json

import pytest

from router_agent import findings


class TestProbe:
    def test_probe_loads(self):
        probe = findings.load_probe()
        if probe is None:
            pytest.skip("results.probe.jsonl not present in this checkout")
        assert probe.n > 0

    def test_matches_the_documented_result(self):
        """The figures STATUS.md and README both quote.

        Updated 8 August 2026 for the quarantine. results.probe.jsonl still
        holds all 200 original rows - nothing measured is deleted here - but
        load_probe drops the 5 unpassable code tasks, so these are now over
        n=95. The old values were n=100, routable 15.0%, both_fail 6.

        The interesting move is both_fail 6 -> 1: five of the six tasks that
        "defeated both rungs" were unpassable rather than hard, which is what
        pushed the rescue rate from 71.4% to 93.8%.
        """
        probe = findings.load_probe()
        if probe is None:
            pytest.skip("results.probe.jsonl not present in this checkout")

        assert probe.n == 95
        assert probe.both_ok == 77
        assert probe.routable == 15
        assert probe.both_fail == 1
        assert probe.inverted == 2
        assert probe.routable_pct == pytest.approx(15.8, abs=0.1)
        assert probe.ceiling_pct == pytest.approx(17.9, abs=0.1)
        assert probe.rescue_rate == pytest.approx(0.938, abs=0.001)
        assert probe.cheap_acc == pytest.approx(0.832, abs=0.001)
        assert probe.expensive_acc == pytest.approx(0.968, abs=0.001)

    def test_quarantined_tasks_never_reach_a_reported_figure(self):
        """The quarantine is not only a build-time filter.

        results.probe.jsonl predates it and still contains every broken task,
        so load_probe has to drop them at the point of use. If this fails, a
        rerun is counting tasks whose expected answers cannot be derived from
        their prompt - and since they were ALL of always_expensive's failures,
        they set the ceiling for every policy in the project.
        """
        from build_taskset import QUARANTINED

        raw = [
            json.loads(line)
            for line in findings.PROBE_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        present = {r["task_id"] for r in raw} & set(QUARANTINED)
        assert present, "probe no longer contains them; this test is now vacuous"

        probe = findings.load_probe()
        assert probe.n == len({r["task_id"] for r in raw}) - len(present)

    def test_confidence_interval_matches_documented(self):
        probe = findings.load_probe()
        if probe is None:
            pytest.skip("results.probe.jsonl not present in this checkout")
        lo, hi = probe.ci95()
        assert lo == pytest.approx(9.8, abs=0.1)
        assert hi == pytest.approx(24.4, abs=0.1)

    def test_cells_partition_the_task_set(self):
        probe = findings.load_probe()
        if probe is None:
            pytest.skip("results.probe.jsonl not present in this checkout")
        assert (probe.both_ok + probe.routable
                + probe.both_fail + probe.inverted) == probe.n

    def test_per_domain_matches_documented(self):
        probe = findings.load_probe()
        if probe is None:
            pytest.skip("results.probe.jsonl not present in this checkout")
        assert probe.by_domain["code"]["routable_pct"] == pytest.approx(14.3, abs=0.1)
        assert probe.by_domain["math"]["routable_pct"] == pytest.approx(16.7, abs=0.1)
        # RETRACTED 8 August 2026: "code is the harder domain" was an artefact.
        # All 5 of its both_fail tasks were unpassable, not hard, so the code
        # half now has ZERO both_fail and a 100% rescue rate - the clean cascade
        # structure. Pinned here because it is the claim most likely to creep
        # back into the docs. See SHIP_PLAN.md section 0.1.
        assert probe.by_domain["code"]["both_fail"] == 0
        assert probe.by_domain["math"]["both_fail"] == 1

    def test_never_reports_simulated_rows_as_measured(self, tmp_path):
        """A mock row must not be able to contaminate a 'measured' figure."""
        import json
        path = tmp_path / "fake.jsonl"
        rows = [
            {"task_id": "t1", "policy": "always_cheap", "correct": False,
             "domain": "math", "simulated": True},
            {"task_id": "t1", "policy": "always_expensive", "correct": True,
             "domain": "math", "simulated": True},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        assert findings.load_probe(str(path)) is None

    def test_missing_file_degrades_rather_than_crashes(self, tmp_path):
        assert findings.load_probe(str(tmp_path / "nope.jsonl")) is None


class TestPriceRatios:
    """Exact arithmetic over the price table - no run, no mode, no task set."""

    @pytest.mark.parametrize("ladder,list_r,eff_r", [
        # Haiku $1.00 -> Opus $5.00 in; Opus is on the newer tokenizer (1.30).
        ("claude", 5.0, 6.5),
        # Both DeepSeek rungs share a tokenizer, so list == effective.
        ("deepseek", 3.11, 3.11),
        # DeepSeek $0.14 -> Opus $5.00, plus the tokenizer asymmetry.
        ("wide", 35.71, 46.43),
    ])
    def test_matches_the_documented_ratios(self, ladder, list_r, eff_r):
        r = findings.price_ratios(ladder)
        assert r["list_ratio"] == pytest.approx(list_r, abs=0.02)
        assert r["effective_ratio"] == pytest.approx(eff_r, abs=0.02)

    def test_effective_is_never_below_list_on_a_mixed_tokenizer_ladder(self):
        """The asymmetry works against escalation, so it can only widen the gap."""
        for ladder in ("claude", "wide"):
            r = findings.price_ratios(ladder)
            assert r["effective_ratio"] >= r["list_ratio"]

    def test_unknown_ladder_returns_none(self):
        assert findings.price_ratios("nonexistent") is None

    def test_ratios_are_ordered_deepseek_claude_wide(self):
        order = [findings.price_ratios(l)["effective_ratio"]
                 for l in ("deepseek", "claude", "wide")]
        assert order == sorted(order)


class TestRatioVerdict:
    def test_sign_flips_across_ladders(self):
        """The repository's headline: cascading pays in proportion to the gap.

        Asserted on the SIGN only. The magnitudes moved substantially in the
        6 August task-set rebuild (claude -12% -> about -46%), which is why
        `ratio_verdict` derives them from a frontier run when one is on disk
        rather than quoting a constant. The sign survived; pinning a magnitude
        here would just re-create the staleness in the test suite.
        """
        assert findings.ratio_verdict("deepseek")["verdict"] == "route"
        assert findings.ratio_verdict("claude")["verdict"] == "cascade"
        assert findings.ratio_verdict("wide")["verdict"] == "cascade"

        # Positive means the cascade costs MORE than always-best.
        assert findings.ratio_verdict("deepseek")["cascade_vs_always_best_pct"] > 0
        assert findings.ratio_verdict("wide")["cascade_vs_always_best_pct"] < 0

    def test_every_verdict_declares_where_it_came_from(self):
        """The flag must track its source, not pin the project to a moment.

        This asserted `economics_simulated is True` unconditionally until
        August 2026, which encoded "the economics have never been run against
        real models" as if it were an invariant. It is a fact about the
        project's state, and the first replay or real frontier run falsifies
        it - so the test would have gone red at precisely the moment the
        project succeeded, which is the worst time for a test to fail.

        What is genuinely invariant is that the flag agrees with where the
        number came from. Assert that instead.
        """
        for ladder in ("deepseek", "claude", "wide"):
            v = findings.ratio_verdict(ladder)
            assert v["economics_source"] in ("frontier.jsonl", "historical")
            assert isinstance(v["economics_simulated"], bool)

            if v["economics_source"] == "historical":
                # The stored constants were recorded from mock runs and cannot
                # become real retrospectively, so this one IS permanent.
                assert v["economics_simulated"] is True
            else:
                live = findings.frontier_economics()
                assert v["economics_simulated"] == live["simulated"]

    def test_only_wide_claims_real_accuracy_data(self):
        assert findings.ratio_verdict("wide")["accuracy_data_is_real"] is True
        assert findings.ratio_verdict("claude")["accuracy_data_is_real"] is False
        assert findings.ratio_verdict("deepseek")["accuracy_data_is_real"] is False

    def test_unknown_ladder_is_honest_about_it(self):
        v = findings.ratio_verdict("nonexistent")
        assert v["known"] is False
        assert "not in models.LADDERS" in v["note"]


class TestFrontierEconomics:
    def test_missing_file_degrades_rather_than_crashes(self, tmp_path):
        findings.frontier_economics.cache_clear()
        assert findings.frontier_economics(str(tmp_path / "nope.jsonl")) is None

    def test_incomplete_file_is_rejected(self, tmp_path):
        """A frontier missing a family cannot produce a comparison."""
        import json as _json
        p = tmp_path / "partial.jsonl"
        p.write_text(_json.dumps({
            "family": "cascade", "cost_per_task": 0.01, "accuracy": 0.9,
            "ladder": "claude", "simulated": True,
        }), encoding="utf-8")
        findings.frontier_economics.cache_clear()
        assert findings.frontier_economics(str(p)) is None

    def test_derives_sign_and_labels_simulation(self, tmp_path):
        import json as _json
        rows = [
            {"family": "always_cheap", "cost_per_task": 0.001, "accuracy": 0.70},
            {"family": "always_expensive", "cost_per_task": 0.010, "accuracy": 0.90},
            # A cascade that reaches the same accuracy for a quarter of the money.
            {"family": "cascade", "cost_per_task": 0.0025, "accuracy": 0.90},
            {"family": "random", "cost_per_task": 0.001, "accuracy": 0.70},
            {"family": "random", "cost_per_task": 0.010, "accuracy": 0.90},
        ]
        for r in rows:
            r.update(ladder="claude", simulated=True, n=50, split="eval")
        p = tmp_path / "frontier.jsonl"
        p.write_text("\n".join(_json.dumps(r) for r in rows), encoding="utf-8")

        findings.frontier_economics.cache_clear()
        econ = findings.frontier_economics(str(p))
        assert econ["ladder"] == "claude"
        assert econ["verdict"] == "cascade"
        assert econ["cascade_vs_always_best_pct"] == pytest.approx(-75.0)
        assert econ["simulated"] is True


class TestSummary:
    def test_carries_caveats(self):
        s = findings.summary("wide")
        assert s["caveats"]
        assert any("nine" in c or "mock" in c for c in s["caveats"])

    def test_verifier_transfer_is_stated(self):
        s = findings.summary("wide")
        assert s["verifiers"]["self_consistency"]["transfers_to_production"] is True
        assert s["verifiers"]["tests"]["transfers_to_production"] is False
