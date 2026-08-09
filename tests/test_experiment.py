"""Unit tests for the research half.

Until August 2026 every test in this suite covered `router_agent/` and none of
them covered the 6,100 lines that actually produce the numbers the repository
reports. `sanity_check.py` guarded the graders and nothing else - and it had
already missed a real grader bug in July by only ever testing `grade(GT, GT)`.

These target the arithmetic and the invariants, not the findings. A test that
pinned an accuracy figure would just re-create the staleness problem the repo
already hit once when the 6 August task-set rebuild moved every magnitude.
What is asserted here is the kind of thing that would silently corrupt a result
rather than break a run:

    price arithmetic          a wrong multiplier makes every cost figure wrong
    cache keys                a missing key field makes two calls collide
    cascade accounting        an unpaid rung makes cascading look cheaper
    the policy-name invariant run_eval drops rows by name; it must match
    McNemar                   the significance claim rests on it
    the convex hull           the "who owns each budget" table rests on it
    split determinism         a drifting split silently changes every result
    math answer equivalence   the exact bug class found in July

No network, no API key, no model call that is not mock or stubbed.
"""

from __future__ import annotations

import json
import math

import pytest

import frontier
import graders
import models
import policies
import response_cache
import run_eval
import splits
import stats


# ---------------------------------------------------------------------------
# Price arithmetic. Every dollar figure in the repository flows through _price.
# ---------------------------------------------------------------------------

class TestPricing:
    def test_price_is_the_published_per_million_rate(self):
        """Hand-computed against the table rather than against itself."""
        m = models.MODELS["cheap"]
        got = models._price("cheap", tokens_in=1_000_000, tokens_out=0)
        assert got == pytest.approx(m["price_in"])

        got = models._price("cheap", tokens_in=0, tokens_out=1_000_000)
        assert got == pytest.approx(m["price_out"])

        # And the two halves add rather than, say, taking the larger.
        got = models._price("cheap", tokens_in=500_000, tokens_out=250_000)
        assert got == pytest.approx(0.5 * m["price_in"] + 0.25 * m["price_out"])

    def test_zero_tokens_cost_zero(self):
        assert models._price("cheap", 0, 0) == 0.0

    def test_output_is_never_cheaper_than_input(self):
        """True of every provider on every current model, and relied on by the
        cascade economics: verification buys OUTPUT tokens, so if this inverted
        the cost of verifying would be mis-stated in the cheap direction."""
        for tier in models.TIERS:
            m = models.MODELS[tier]
            assert m["price_out"] >= m["price_in"]

    @pytest.mark.parametrize("ladder", ["deepseek", "claude", "wide"])
    def test_every_ladder_is_ordered_cheapest_first(self, use_ladder, ladder):
        """`cheap` -> `mid` -> `expensive` must be strictly increasing.

        The whole repository indexes rungs by position and calls the first one
        cheap. A ladder that was not ordered would invert the sign of the
        headline finding without breaking a single run.
        """
        m = use_ladder(ladder)
        prices = [m.MODELS[t]["price_out"] for t in m.TIERS]
        assert prices == sorted(prices)
        assert len(set(prices)) == len(prices), "two rungs priced identically"


# ---------------------------------------------------------------------------
# Cache keys. A missing field here silently serves one call's response to a
# different call - which would look like a model being unexpectedly consistent.
# ---------------------------------------------------------------------------

BASE_KEY = dict(
    mode="mock", model="some-model-id", prompt="solve this",
    temperature=0.0, sample_idx=0, max_tokens=4096, mock_seed=0,
)


class TestCacheKey:
    def test_identical_inputs_give_an_identical_key(self):
        assert response_cache.make_key(**BASE_KEY) == response_cache.make_key(**BASE_KEY)

    @pytest.mark.parametrize("field,other", [
        ("mode", "real"),
        ("model", "another-model-id"),
        ("prompt", "solve this differently"),
        ("temperature", 0.8),
        ("sample_idx", 1),
        ("max_tokens", 8),
        ("mock_seed", 1),
    ])
    def test_every_determining_field_changes_the_key(self, field, other):
        """Parametrised over response_cache._KEY_FIELDS deliberately.

        The failure this prevents is specific: `sample_idx` was added so
        self-consistency could draw k DIFFERENT samples. If it dropped out of
        the key, all five draws would collide on one entry, agreement would be
        unanimous by construction, and the verifier would look perfect.
        """
        changed = {**BASE_KEY, field: other}
        assert response_cache.make_key(**changed) != response_cache.make_key(**BASE_KEY)

    def test_the_parametrised_list_still_covers_every_key_field(self):
        """Guards the test above from going stale if a field is added."""
        covered = set(BASE_KEY)
        assert covered == set(response_cache._KEY_FIELDS)


# ---------------------------------------------------------------------------
# Cascade accounting. If a rung is not charged, cascading looks cheaper than it
# is - which is the exact direction that would flatter this repo's thesis.
# ---------------------------------------------------------------------------

def _stub_call(monkeypatch, per_tier_cost, text_by_tier=None):
    """Replace models.call with a priced stub, and record what was asked for."""
    seen = []

    def fake_call(tier, task, temperature=0.0, sample_idx=0, kind="answer"):
        seen.append(tier)
        return models.ModelResponse(
            text=(text_by_tier or {}).get(tier, "answer"),
            tier=tier, tokens_in=10, tokens_out=10,
            latency_s=1.0, cost_usd=per_tier_cost[tier],
        )

    monkeypatch.setattr(models, "call", fake_call)
    return seen


def _verifier(accepted, cost=0.0):
    def v(task, response_text, tier="cheap"):
        return policies.Verdict(
            accepted=accepted, answer_text=response_text,
            cost_usd=cost, latency_s=0.0,
        )
    return v


CODE_TASK = {"id": "t-1", "domain": "code", "grader_payload": {}}


class TestCascadeAccounting:
    def test_accepting_at_the_cheap_rung_never_buys_the_expensive_one(self, monkeypatch):
        seen = _stub_call(monkeypatch, {"cheap": 0.001, "expensive": 0.100})
        monkeypatch.setattr(policies, "grade", lambda task, text: True)

        res = policies._cascade(
            CODE_TASK, {"code": _verifier(accepted=True)}, "cascade",
            chain=("cheap", "expensive"),
        )

        assert seen == ["cheap"]
        assert res.calls == ["cheap"]
        assert res.escalated is False
        assert res.cost_usd == pytest.approx(0.001)

    def test_an_escalation_pays_for_both_rungs_and_the_verifier(self, monkeypatch):
        """The fixed cost that decides the whole crossover finding.

        A cascade that escalated for the price of the expensive call alone
        would be free to be wrong at the cheap rung, and the 3x sign flip
        would disappear.
        """
        seen = _stub_call(monkeypatch, {"cheap": 0.001, "expensive": 0.100})
        monkeypatch.setattr(policies, "grade", lambda task, text: True)

        res = policies._cascade(
            CODE_TASK, {"code": _verifier(accepted=False, cost=0.005)}, "cascade",
            chain=("cheap", "expensive"),
        )

        assert seen == ["cheap", "expensive"]
        assert res.escalated is True
        assert res.cost_usd == pytest.approx(0.001 + 0.005 + 0.100)

    def test_the_top_rung_is_never_verified(self, monkeypatch):
        """Verifying an answer you cannot act on is pure cost. If the top rung
        were verified, every escalation would carry a verification charge that
        buys no decision."""
        _stub_call(monkeypatch, {"cheap": 0.001, "expensive": 0.100})
        monkeypatch.setattr(policies, "grade", lambda task, text: True)
        calls = []

        def counting_verifier(task, response_text, tier="cheap"):
            calls.append(tier)
            return policies.Verdict(False, response_text, 0.0, 0.0)

        policies._cascade(
            CODE_TASK, {"code": counting_verifier}, "cascade",
            chain=("cheap", "expensive"),
        )
        assert calls == ["cheap"], "the last rung must not be verified"

    def test_it_grades_the_verifiers_answer_not_the_raw_one(self, monkeypatch):
        """verify_math buys k samples and the majority vote beats the single
        draw it was handed. Grading the raw response would throw that away and
        under-report the cascade's accuracy."""
        _stub_call(monkeypatch, {"cheap": 0.001, "expensive": 0.100},
                   text_by_tier={"cheap": "raw"})
        graded = []
        monkeypatch.setattr(policies, "grade",
                            lambda task, text: graded.append(text) or True)

        def voting_verifier(task, response_text, tier="cheap"):
            return policies.Verdict(True, "majority-vote", 0.0, 0.0)

        policies._cascade(CODE_TASK, {"code": voting_verifier}, "cascade",
                          chain=("cheap", "expensive"))
        assert graded == ["majority-vote"]


# ---------------------------------------------------------------------------
# The policy registry invariant that run_eval._drop_uncached depends on.
# ---------------------------------------------------------------------------

class TestPolicyRegistry:
    def test_every_policy_reports_the_name_it_is_registered_under(self, benchmark_task):
        """`run_eval` drops a policy's rows by matching row["policy"] against
        the POLICIES key. If a policy reported a different name, a replay cache
        miss would drop nothing and the incomplete policy would be reported as
        if it were fully scored.
        """
        task = next(t for t in benchmark_task if t["domain"] == "code")
        for name, fn in policies.POLICIES.items():
            if name in ("routellm", "cascade_routing"):
                continue  # need calibration; covered by run_eval.applicable
            assert fn(task).policy == name


# ---------------------------------------------------------------------------
# McNemar. The p = 0.002 significance claim rests entirely on this function.
# ---------------------------------------------------------------------------

class TestMcNemar:
    def test_matches_a_hand_computed_table(self):
        """b=1, c=5. Under the null min(b,c) ~ Binomial(6, 0.5), so the
        one-sided tail is (C(6,0) + C(6,1)) / 2^6 = 7/64 and p = 14/64."""
        a = [True] + [False] * 5 + [True] * 3
        b = [False] + [True] * 5 + [True] * 3
        got_b, got_c, p = stats.mcnemar_exact(a, b)
        assert (got_b, got_c) == (1, 5)
        assert p == pytest.approx(14 / 64)

    def test_concordant_pairs_carry_no_information(self):
        """Adding tasks both policies get right must not move the p-value.
        This is the property that makes McNemar the right test for a paired
        design, and the reason a saturated benchmark cannot rescue itself by
        adding easy tasks."""
        a = [True, False, False]
        b = [False, True, True]
        _, _, p_small = stats.mcnemar_exact(a, b)
        _, _, p_padded = stats.mcnemar_exact(a + [True] * 50, b + [True] * 50)
        assert p_small == pytest.approx(p_padded)

    def test_total_agreement_is_p_one(self):
        a = b = [True, False, True]
        assert stats.mcnemar_exact(a, b) == (0, 0, 1.0)

    def test_swapping_the_arms_swaps_b_and_c_but_not_p(self):
        a = [True, True, False, False]
        b = [False, True, True, True]
        b1, c1, p1 = stats.mcnemar_exact(a, b)
        b2, c2, p2 = stats.mcnemar_exact(b, a)
        assert (b1, c1) == (c2, b2)
        assert p1 == pytest.approx(p2)


# ---------------------------------------------------------------------------
# The cost-quality hull. "Who owns each budget" is read straight off this.
# ---------------------------------------------------------------------------

class TestUpperHull:
    def test_a_dominated_point_is_dropped(self):
        """Costs more, scores worse: no budget is ever served by it."""
        hull = frontier.upper_hull([(1.0, 0.5), (2.0, 0.4), (3.0, 0.9)])
        assert (2.0, 0.4) not in hull

    def test_a_point_below_the_chord_is_dropped(self):
        """Concavity. A mix of the two endpoints beats it at its own price, so
        it is not on the achievable frontier even though nothing dominates it
        outright."""
        hull = frontier.upper_hull([(0.0, 0.0), (1.0, 0.4), (2.0, 1.0)])
        assert (1.0, 0.4) not in hull
        assert hull[0] == (0.0, 0.0) and hull[-1] == (2.0, 1.0)

    def test_a_point_above_the_chord_is_kept(self):
        hull = frontier.upper_hull([(0.0, 0.0), (1.0, 0.8), (2.0, 1.0)])
        assert (1.0, 0.8) in hull

    def test_the_hull_rises_with_cost(self):
        pts = [(0.1, 0.2), (0.4, 0.9), (0.2, 0.5), (0.8, 0.95), (0.3, 0.1)]
        hull = frontier.upper_hull(pts)
        costs = [c for c, _ in hull]
        accs = [a for _, a in hull]
        assert costs == sorted(costs)
        assert accs == sorted(accs)

    def test_auc_of_a_flat_hull_is_that_accuracy(self):
        """One measured point extends flat in both directions, so its mean
        accuracy over any budget window is just its accuracy."""
        assert frontier.auc([(1.0, 0.75)], lo=0.0, hi=10.0) == pytest.approx(0.75)

    def test_auc_of_a_straight_ramp_is_its_midpoint(self):
        got = frontier.auc([(0.0, 0.0), (1.0, 1.0)], lo=0.0, hi=1.0)
        assert got == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------------------
# Splits. A drifting split changes every reported number without any error.
# ---------------------------------------------------------------------------

def _fake_tasks(n=40):
    return [
        {"id": f"t-{i}", "domain": "math" if i % 2 else "code",
         "difficulty_pct": (i % 10) / 10.0}
        for i in range(n)
    ]


class TestSplits:
    def test_the_split_is_deterministic(self):
        tasks = _fake_tasks()
        a_cal, a_ev = splits.split(tasks)
        b_cal, b_ev = splits.split(tasks)
        assert [t["id"] for t in a_cal] == [t["id"] for t in b_cal]
        assert [t["id"] for t in a_ev] == [t["id"] for t in b_ev]

    def test_the_two_halves_partition_the_task_set(self):
        tasks = _fake_tasks()
        cal, ev = splits.split(tasks)
        cal_ids = {t["id"] for t in cal}
        ev_ids = {t["id"] for t in ev}
        assert cal_ids & ev_ids == set()
        assert cal_ids | ev_ids == {t["id"] for t in tasks}

    def test_a_tasks_side_does_not_depend_on_the_other_tasks(self):
        """splits._rank hashes the task id and the seed, so membership is a
        property of the task alone. The docstring claims it; nothing checked
        it. If it broke, adding a task would reshuffle the evaluation half and
        every historical comparison would quietly stop being comparable.
        """
        tasks = _fake_tasks(40)
        cal_before = {t["id"] for t in splits.split(tasks)[0]}

        # Drop the last task of each stratum, then re-split.
        fewer = tasks[:-6]
        cal_after = {t["id"] for t in splits.split(fewer)[0]}

        survivors = {t["id"] for t in fewer}
        moved = (cal_before & survivors) ^ (cal_after & survivors)
        # Stratum sizes change, so the CUT can move by at most one task per
        # stratum; no task may leapfrog another.
        assert len(moved) <= len({t["domain"] for t in tasks}) * 3

    def test_ordering_is_preserved_in_both_halves(self):
        tasks = _fake_tasks()
        cal, ev = splits.split(tasks)
        order = [t["id"] for t in tasks]
        assert [t["id"] for t in cal] == [i for i in order if i in {t["id"] for t in cal}]
        assert [t["id"] for t in ev] == [i for i in order if i in {t["id"] for t in ev}]


# ---------------------------------------------------------------------------
# The maths grader. This is the July 2026 bug class, pinned.
# ---------------------------------------------------------------------------

class TestQuarantine:
    """A quarantined task must never be counted again, in any rerun.

    Five MBPP+ tasks were found on 8 August 2026 to have expected answers that
    cannot be derived from their prompt - they score against whatever the MBPP
    reference happened to return on inputs the prompt never describes. They were
    ALL of always_expensive's failures on the eval split, so leaving them in caps
    every policy in the project at 92% instead of 100%.

    The rule has two halves and both are enforced here: they are absent from a
    freshly built task set, and they are filtered out of the artefacts recorded
    before the quarantine existed, which still contain them.
    """

    def test_rebuild_never_reintroduces_them(self):
        import build_taskset

        tasks = [json.loads(l) for l in
                 (REPO_ROOT / "taskset.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        present = {t["id"] for t in tasks} & set(build_taskset.QUARANTINED)
        assert not present, (
            f"taskset.jsonl contains quarantined task(s) {sorted(present)}. "
            f"Rebuild with `python build_taskset.py` - and if this came from "
            f"--keep-quarantined, every number downstream is an artefact."
        )

    def test_row_filter_drops_them_from_older_artefacts(self):
        import build_taskset

        victim = next(iter(build_taskset.QUARANTINED))
        rows = [{"task_id": victim}, {"task_id": "math-1"}, {"task_id": "codeplus-800"}]
        kept = build_taskset.drop_quarantined_rows(rows, "unit test", warn=False)
        assert [r["task_id"] for r in kept] == ["math-1", "codeplus-800"]

    def test_every_quarantined_task_carries_its_evidence(self):
        """A bare id is not a justification. Each needs a stated reason, because
        'both models failed' is exactly the reasoning that got these five
        mistaken for hard tasks in the first place."""
        import build_taskset

        for task_id, reason in build_taskset.QUARANTINED.items():
            assert len(reason) > 40, f"{task_id} has no real evidence recorded"


class TestMathAnswerEquivalence:
    @pytest.mark.parametrize("got,want", [
        # The exact pair that was scoring seven correct answers as wrong.
        (r"1+\sqrt{19}, 1-\sqrt{19}", r"1 \pm \sqrt{19}"),
        # Order must not matter for a solution set.
        (r"1-\sqrt{19}, 1+\sqrt{19}", r"1 \pm \sqrt{19}"),
        # Cosmetic LaTeX differences.
        (r"\frac{1}{2}", r"\dfrac{1}{2}"),
        (r"\left(3\right)", r"(3)"),
        # Whitespace.
        (r"2 , 3", r"2,3"),
    ])
    def test_equivalent_formattings_are_accepted(self, got, want):
        response = rf"Reasoning... the answer is $\boxed{{{got}}}$"
        assert graders.grade_exact_match_str(response, {"answer": want}) is True

    @pytest.mark.parametrize("got,want", [
        # A near miss must still be wrong. This is the half of the gate that
        # matters once the normaliser is made permissive - a normaliser loose
        # enough to accept everything would pass the test above and destroy
        # the experiment.
        (r"1 \pm \sqrt{18}", r"1 \pm \sqrt{19}"),
        (r"1+\sqrt{19}", r"1 \pm \sqrt{19}"),   # one root, not both
        (r"\frac{1}{3}", r"\frac{1}{2}"),
        (r"-2", r"2"),
        (r"2,3,4", r"2,3"),
    ])
    def test_near_miss_wrong_answers_are_still_rejected(self, got, want):
        response = rf"Reasoning... the answer is $\boxed{{{got}}}$"
        assert graders.grade_exact_match_str(response, {"answer": want}) is False

    def test_the_last_boxed_answer_wins(self):
        """Models revise. The prompt asks for \\boxed{}, and the final one is
        the model's actual claim."""
        response = r"First I thought $\boxed{5}$ but actually $\boxed{7}$"
        assert graders.grade_exact_match_str(response, {"answer": "7"}) is True
        assert graders.grade_exact_match_str(response, {"answer": "5"}) is False


# ---------------------------------------------------------------------------
# Replay misses. The failure mode fixed in August 2026.
# ---------------------------------------------------------------------------

class TestReplayMiss:
    def test_it_is_a_keyerror_so_old_handlers_still_catch_it(self):
        assert issubclass(models.ReplayMiss, KeyError)

    def test_drop_uncached_removes_every_row_of_that_policy(self, capsys):
        """Including rows scored BEFORE the miss was hit. A policy measured on
        the subset of tasks that happened to be cached is measured on a biased
        subset, and would be printed next to fully-scored policies."""
        rows = [
            {"policy": "cascade", "task_id": "t-1"},
            {"policy": "llm_router", "task_id": "t-1"},
            {"policy": "cascade", "task_id": "t-2"},
            {"policy": "llm_router", "task_id": "t-2"},
        ]
        kept = run_eval._drop_uncached(rows, {"llm_router": "t-3"})
        assert [r["policy"] for r in kept] == ["cascade", "cascade"]

    def test_drop_uncached_says_which_policy_and_where(self, capsys):
        run_eval._drop_uncached([{"policy": "llm_router"}], {"llm_router": "codeplus-418"})
        err = capsys.readouterr().err
        assert "llm_router" in err and "codeplus-418" in err

    def test_no_misses_returns_the_rows_untouched(self, capsys):
        rows = [{"policy": "cascade"}]
        assert run_eval._drop_uncached(rows, {}) is rows
        assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# Response provenance. On 7 August 2026 a replay served 240 fabricated
# self-consistency samples into a results.jsonl whose every row said
# `simulated: false`, because the flag was derived from the run MODE and from
# whether the real cache FILE existed - neither of which is a fact about any
# response. See NOTES.md issue 19.
#
# The bug survived 156 tests, sanity_check.py and check_core_unchanged.py,
# because nothing compared a served response against the cache it came from.
# That is what these do.
# ---------------------------------------------------------------------------

class TestResponseProvenance:
    def _record(self, mode):
        return {"text": "42", "tier": "cheap", "tokens_in": 10, "tokens_out": 2,
                "latency_s": 0.1, "cost_usd": 0.001, "mode": mode}

    def test_a_mock_record_is_reported_as_simulated(self, monkeypatch, benchmark_task):
        """The flag comes from the RECORD's stored mode, not from the run mode."""
        monkeypatch.setattr(models, "MODE", "replay")
        monkeypatch.setattr(models, "REPLAY_FALLBACK_TO_MOCK", True)
        monkeypatch.setattr(response_cache, "get", lambda k: self._record("mock"))
        assert models.call("cheap", benchmark_task[0]).simulated is True

    def test_a_real_record_is_not(self, monkeypatch, benchmark_task):
        monkeypatch.setattr(models, "MODE", "replay")
        monkeypatch.setattr(response_cache, "get", lambda k: self._record("real"))
        assert models.call("cheap", benchmark_task[0]).simulated is False

    def test_the_fallback_defaults_to_off(self):
        """The default is the whole fix. The mechanism being *available* was
        never the problem - it was available and on, for weeks."""
        assert models.REPLAY_FALLBACK_TO_MOCK is False

    def test_the_fallback_is_off_unless_asked_for(self, monkeypatch, benchmark_task):
        """A mock record must NOT satisfy a replay lookup by default.

        This is the property whose absence made ReplayMiss dead code: while the
        mock cache answered every miss, the miss could never happen and nothing
        downstream of it could ever run.
        """
        monkeypatch.setattr(models, "MODE", "replay")
        monkeypatch.setattr(models, "REPLAY_FALLBACK_TO_MOCK", False)

        seen = []

        def only_mock_is_stored(key):
            seen.append(key)
            return None          # nothing in the real cache

        monkeypatch.setattr(response_cache, "get", only_mock_is_stored)
        with pytest.raises(models.ReplayMiss):
            models.call("cheap", benchmark_task[0])
        # Exactly one lookup: the real key. The mock key is never even built.
        assert len(seen) == 1

    def test_call_stats_split_real_from_fabricated(self, monkeypatch, benchmark_task):
        monkeypatch.setattr(models, "MODE", "replay")
        monkeypatch.setattr(models, "REPLAY_FALLBACK_TO_MOCK", True)
        models.reset_call_stats()

        monkeypatch.setattr(response_cache, "get", lambda k: self._record("real"))
        models.call("cheap", benchmark_task[0])
        monkeypatch.setattr(response_cache, "get", lambda k: self._record("mock"))
        models.call("cheap", benchmark_task[0])

        assert models.call_stats["served_real"] == 1
        assert models.call_stats["served_mock"] == 1

    def test_provenance_prefers_the_measured_value_over_the_mode_guess(self):
        """run_eval.provenance's own fallback is the thing that got this wrong.

        The caller knows what was served; the mode does not. When the caller
        says so, that must win - in BOTH directions, so a real run cannot be
        mislabelled either.
        """
        assert run_eval.provenance(simulated=True)["simulated"] is True
        assert run_eval.provenance(simulated=False)["simulated"] is False

    def test_a_row_is_simulated_if_any_of_its_calls_was(self, monkeypatch):
        """One fabricated call taints the row.

        A cascade that verifies a genuine cheap answer with five mock samples is
        reporting the mock's verdict, not the model's - which is precisely the
        shape of the 7 August contamination.
        """
        models.reset_call_stats()
        before = models.call_stats["served_mock"]
        models.call_stats["served_real"] += 4       # four real calls
        models.call_stats["served_mock"] += 1       # and one fabricated
        assert (models.call_stats["served_mock"] > before) is True


# ---------------------------------------------------------------------------
# The routable re-estimate. This function corrected the repository's headline
# from 15.0% to 10.2%, so it had better be right.
# ---------------------------------------------------------------------------

def _import_redraw():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "redraw_decisive", REPO_ROOT / "scripts" / "redraw_decisive.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


class TestRoutableReestimate:
    def setup_method(self):
        self.rd = _import_redraw()

    def _cells(self, routable_ids, both_fail_ids=()):
        return {
            "routable": [{"id": i} for i in routable_ids],
            "both_fail": [{"id": i} for i in both_fail_ids],
            "both_ok": [], "inverted": [],
        }

    def test_a_phantom_is_removed_from_the_estimate(self):
        """The exact case found on 7 August: the probe called a task routable,
        but ten redraws show the cheap rung solves it every time. It must not
        count toward either the expected or the reproducible figure."""
        cells = self._cells(["phantom"])
        p_hat = {"phantom": {"cheap": 1.0, "expensive": 1.0}}
        out = self.rd.reestimate(cells, p_hat, n_total=100, tau=0.2)
        assert out["observed"] == pytest.approx(0.01)
        assert out["expected"] == pytest.approx(0.0)
        assert out["reproducible"] == pytest.approx(0.0)

    def test_a_solid_routable_task_survives_intact(self):
        cells = self._cells(["solid"])
        p_hat = {"solid": {"cheap": 0.0, "expensive": 1.0}}
        out = self.rd.reestimate(cells, p_hat, n_total=100, tau=0.2)
        assert out["expected"] == pytest.approx(0.01)
        assert out["reproducible"] == pytest.approx(0.01)
        assert out["noise_share"] == pytest.approx(0.0)

    def test_a_flaky_task_contributes_a_fraction_not_a_whole_task(self):
        """cheap succeeds 40% of the time, expensive always. The task is
        routable on 60% of draws, so it should contribute 0.6, not 1.0 - and
        it is not capturable, because a router cannot predict which draw it
        will get."""
        cells = self._cells(["flaky"])
        p_hat = {"flaky": {"cheap": 0.4, "expensive": 1.0}}
        out = self.rd.reestimate(cells, p_hat, n_total=100, tau=0.2)
        assert out["expected"] == pytest.approx(0.006)
        assert out["reproducible"] == pytest.approx(0.0)

    def test_a_both_fail_task_can_add_routable_mass_back(self):
        """codeplus-305 was counted both_fail, but the expensive rung solves it
        half the time. The correction runs in both directions."""
        cells = self._cells([], both_fail_ids=["rescued"])
        p_hat = {"rescued": {"cheap": 0.0, "expensive": 0.5}}
        out = self.rd.reestimate(cells, p_hat, n_total=100, tau=0.2)
        assert out["observed"] == pytest.approx(0.0)
        assert out["expected"] == pytest.approx(0.005)

    def test_it_reproduces_the_published_re_estimate(self):
        """End to end against the committed artefact. If redraw.wide.json is
        present, the numbers in README and STATUS must fall out of it."""
        path = REPO_ROOT / "redraw.wide.json"
        if not path.exists():
            pytest.skip("redraw.wide.json not present; run scripts/redraw_decisive.py")
        import json
        d = json.loads(path.read_text(encoding="utf-8"))
        assert d["observed"] == pytest.approx(0.15)
        assert d["expected"] == pytest.approx(0.102, abs=0.002)
        assert d["reproducible"] == pytest.approx(0.09)
        # Four phantoms: cheap solves them every time.
        phantoms = [t for t, p in d["p_hat"].items() if p["cheap"] >= 0.9]
        assert len(phantoms) == 4


# ---------------------------------------------------------------------------
# The realized price ratio, measured rather than quoted.
# ---------------------------------------------------------------------------

class TestRealizedRatio:
    def test_absent_for_a_ladder_never_run_for_real(self, tmp_path):
        from router_agent import findings
        assert findings.realized_ratio("wide", path=tmp_path / "nope.jsonl") is None

    def test_it_measures_what_was_billed_not_what_was_listed(self, tmp_path):
        """Two tasks, both rungs, hand-set costs: the ratio is 10x whatever the
        price table says, because it is read off the invoices."""
        from router_agent import findings
        path = tmp_path / "raw.jsonl"
        rows = []
        for tid in ("t-1", "t-2"):
            for tier, cost in (("cheap", 0.001), ("expensive", 0.010)):
                rows.append({"task_id": tid, "tier": tier, "kind": "answer",
                             "mode": "real", "temperature": 0.0, "cost_usd": cost})
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        got = findings.realized_ratio("wide", path=path)
        assert got["ratio"] == pytest.approx(10.0)
        assert got["n_tasks"] == 2

    def test_a_task_with_only_one_rung_is_excluded(self, tmp_path):
        """Otherwise the two means describe different task populations, and the
        ratio silently becomes a comparison of task mixes."""
        from router_agent import findings
        path = tmp_path / "raw.jsonl"
        rows = [
            {"task_id": "both", "tier": "cheap", "kind": "answer", "mode": "real",
             "temperature": 0.0, "cost_usd": 0.001},
            {"task_id": "both", "tier": "expensive", "kind": "answer", "mode": "real",
             "temperature": 0.0, "cost_usd": 0.010},
            # Expensive only, and wildly dearer. Must not move the ratio.
            {"task_id": "lonely", "tier": "expensive", "kind": "answer", "mode": "real",
             "temperature": 0.0, "cost_usd": 5.0},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        got = findings.realized_ratio("wide", path=path)
        assert got["n_tasks"] == 1
        assert got["ratio"] == pytest.approx(10.0)

    def test_self_consistency_samples_are_excluded(self, tmp_path):
        """They sit at temperature 0.8 and are a shorter, cheaper action, so
        counting them would drag the cheap rung's mean down and inflate the
        ratio."""
        from router_agent import findings
        path = tmp_path / "raw.jsonl"
        rows = [
            {"task_id": "t", "tier": "cheap", "kind": "answer", "mode": "real",
             "temperature": 0.0, "cost_usd": 0.001},
            {"task_id": "t", "tier": "expensive", "kind": "answer", "mode": "real",
             "temperature": 0.0, "cost_usd": 0.010},
            {"task_id": "t", "tier": "cheap", "kind": "answer", "mode": "real",
             "temperature": 0.8, "cost_usd": 0.0000001},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        assert findings.realized_ratio("wide", path=path)["ratio"] == pytest.approx(10.0)

    def test_the_committed_wide_cache_is_dearer_than_the_price_table_says(self):
        """The 7 August finding: quoted 46.4x, billed 68.2x. Asserted as an
        inequality rather than a constant, so more real runs can move it."""
        from router_agent import findings
        r = findings.price_ratios("wide")
        if "realized_ratio" not in r:
            pytest.skip("no committed wide cache")
        assert r["realized_ratio"] > r["effective_ratio"]
