"""The findings module, and the guarantee that it never invents a number.

These tests are regression protection for a specific failure this repository
has already had once: on 6 August 2026 a grader bug moved the routable fraction
from 13.0% to 15.0%. Any figure that had been transcribed into source would
have silently kept saying 13% - so `findings` recomputes from the committed
data, and these tests check that the recomputation still matches the documented
result.
"""

from __future__ import annotations

import pytest

from router_agent import findings


class TestProbe:
    def test_probe_loads(self):
        probe = findings.load_probe()
        if probe is None:
            pytest.skip("results.probe.jsonl not present in this checkout")
        assert probe.n > 0

    def test_matches_the_documented_result(self):
        """The figures STATUS.md and README both quote."""
        probe = findings.load_probe()
        if probe is None:
            pytest.skip("results.probe.jsonl not present in this checkout")

        assert probe.n == 100
        assert probe.both_ok == 77
        assert probe.routable == 15
        assert probe.both_fail == 6
        assert probe.inverted == 2
        assert probe.routable_pct == pytest.approx(15.0)
        assert probe.ceiling_pct == pytest.approx(17.0)
        assert probe.rescue_rate == pytest.approx(0.714, abs=0.001)
        assert probe.cheap_acc == pytest.approx(0.79)
        assert probe.expensive_acc == pytest.approx(0.92)

    def test_confidence_interval_matches_documented(self):
        probe = findings.load_probe()
        if probe is None:
            pytest.skip("results.probe.jsonl not present in this checkout")
        lo, hi = probe.ci95()
        assert lo == pytest.approx(9.3, abs=0.1)
        assert hi == pytest.approx(23.3, abs=0.1)

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
        assert probe.by_domain["code"]["routable_pct"] == pytest.approx(12.5)
        assert probe.by_domain["math"]["routable_pct"] == pytest.approx(16.7, abs=0.1)
        # Code is the harder domain, which reverses the project's original
        # assumption and is worth pinning down.
        assert probe.by_domain["code"]["both_fail"] == 5
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


class TestRatioVerdict:
    def test_sign_flips_across_ladders(self):
        """The repository's headline: cascading pays in proportion to the gap."""
        assert findings.ratio_verdict("deepseek")["verdict"] == "route"
        assert findings.ratio_verdict("claude")["verdict"] == "cascade"
        assert findings.ratio_verdict("wide")["verdict"] == "cascade"

        # Positive means the cascade costs MORE than always-best.
        assert findings.ratio_verdict("deepseek")["cascade_vs_always_best_pct"] > 0
        assert findings.ratio_verdict("wide")["cascade_vs_always_best_pct"] < 0

    def test_wider_ratio_means_bigger_saving(self):
        rows = [findings.RATIO_FINDING[k] for k in ("deepseek", "claude", "wide")]
        ratios = [r["effective_ratio"] for r in rows]
        savings = [r["cascade_vs_always_best_pct"] for r in rows]
        assert ratios == sorted(ratios)
        # Monotonically more favourable to the cascade as the ratio widens.
        assert savings == sorted(savings, reverse=True)

    def test_unknown_ladder_is_honest_about_it(self):
        v = findings.ratio_verdict("nonexistent")
        assert v["known"] is False
        assert "crossover" in v["note"] or "3.0" in v["note"]

    def test_only_wide_claims_real_accuracy_data(self):
        assert findings.RATIO_FINDING["wide"]["measured"] is True
        assert findings.RATIO_FINDING["claude"]["measured"] is False
        assert findings.RATIO_FINDING["deepseek"]["measured"] is False


class TestSummary:
    def test_carries_caveats(self):
        s = findings.summary("wide")
        assert s["caveats"]
        assert any("nine" in c or "mock" in c for c in s["caveats"])

    def test_verifier_transfer_is_stated(self):
        s = findings.summary("wide")
        assert s["verifiers"]["self_consistency"]["transfers_to_production"] is True
        assert s["verifiers"]["tests"]["transfers_to_production"] is False
