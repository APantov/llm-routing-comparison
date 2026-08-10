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
    """A quarantined task must never come back, in any artefact.

    Five MBPP+ tasks were found on 8 August 2026 to have expected answers that
    cannot be derived from their prompt - they score against whatever the MBPP
    reference happened to return on inputs the prompt never describes. They were
    ALL of always_expensive's failures on the eval split, so leaving them in
    capped every policy at 92% instead of 100%.

    On 9 August 2026 every trace of them was deleted rather than filtered
    (`scripts/purge_quarantined.py`). This class is the tripwire that keeps it
    that way: nothing downstream screens them out any more, so a reintroduced id
    would be counted rather than ignored.
    """

    def test_no_artefact_mentions_them(self):
        """The whole rule, in one assertion, over every file that stores a task id."""
        import build_taskset

        quarantined = set(build_taskset.QUARANTINED)
        targets = [
            "taskset.jsonl", "results.jsonl", "results.probe.jsonl",
            "cache/raw_calls.wide.jsonl", "cache/raw_calls.claude.jsonl",
            "cache/raw_calls.deepseek.jsonl", "cache/routellm_scores.jsonl",
            "redraw.wide.json",
        ]
        offenders = {}
        for rel in targets:
            path = REPO_ROOT / rel
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            hits = sorted(q for q in quarantined if q in text)
            if hits:
                offenders[rel] = hits
        assert not offenders, (
            f"quarantined tasks are back in {offenders}.\n"
            f"They are unpassable, not hard, and nothing filters them any more. "
            f"Re-run `python scripts/purge_quarantined.py --go`, and work out "
            f"what put them back before trusting any number from this run."
        )

    def test_rebuild_never_reintroduces_them(self):
        import build_taskset

        pool = build_taskset.load_mbppplus()
        assert set(build_taskset.QUARANTINED) <= {t["id"] for t in pool}, (
            "a quarantined id is no longer in the MBPP+ pool; if the source "
            "changed, re-verify the quarantine rather than carrying it forward"
        )
        kept = build_taskset.drop_quarantined([{"id": t["id"]} for t in pool])
        assert not ({t["id"] for t in kept} & set(build_taskset.QUARANTINED))

    def test_every_quarantined_task_carries_its_evidence(self):
        """A bare id is not a justification. Each needs a stated reason, because
        'both models failed' is exactly the reasoning that got these five
        mistaken for hard tasks in the first place - and the responses that
        would let someone re-derive it have now been deleted."""
        import build_taskset

        for task_id, reason in build_taskset.QUARANTINED.items():
            assert len(reason) > 40, f"{task_id} has no real evidence recorded"

    def test_nothing_passable_is_quarantined(self):
        """The bar tightened on 10 August: every rung's p-hat must be 0.

        `codeplus-305` was quarantined on 8 August while `redraw.wide.json` in
        the same commit recorded its expensive rung at p-hat 0.5. A task the
        cheap rung reliably fails and the expensive rung solves half the time is
        ROUTABLE - the most valuable kind of task here - and quarantining it
        deleted the signal the project exists to measure.

        This reads whatever draw data is on disk and fails if any quarantined id
        was ever observed passing. It is deliberately cheap to satisfy and hard
        to argue with: one passing draw is a disproof of "unpassable".
        """
        import json

        import build_taskset

        quarantined = set(build_taskset.QUARANTINED)
        offenders = {}
        for name in ("redraw.wide.json", "redraw.claude.json",
                     "redraw.deepseek.json"):
            path = REPO_ROOT / name
            if not path.exists():
                continue
            p_hat = json.loads(path.read_text(encoding="utf-8")).get("p_hat", {})
            for task_id, rungs in p_hat.items():
                if task_id not in quarantined:
                    continue
                passing = {r: p for r, p in rungs.items() if p}
                if passing:
                    offenders[task_id] = passing
        assert not offenders, (
            f"quarantined but observed passing: {offenders}.\n"
            f"'Unpassable' is disproved by a single passing draw. Either the "
            f"quarantine is wrong - see build_taskset.UNQUARANTINED for the "
            f"codeplus-305 reversal - or the draw data is."
        )

    def test_a_reversal_records_its_evidence_too(self):
        """A quarantine decision is recorded with its evidence, and a reversal
        is a quarantine decision. Otherwise the next reader sees a task that was
        removed and then silently restored, with no way to tell which call was
        the considered one."""
        import build_taskset

        for task_id, reason in build_taskset.UNQUARANTINED.items():
            assert task_id not in build_taskset.QUARANTINED, (
                f"{task_id} is in both QUARANTINED and UNQUARANTINED")
            assert len(reason) > 40, f"{task_id} reversal has no evidence recorded"


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
# The spend guards.
#
# Both of these were real hazards on 9 August 2026, before the largest paid
# batch in the project's history. The cap printed "stopping early" and carried
# on, so a half-measured task set would have been written to results.jsonl and
# read as complete; and ROUTER_CACHE=0 would have paid for every response and
# discarded it, because put() is a no-op with the cache off.
#
# Neither had a test, which is how they survived. A guard nothing exercises is
# the same shape of defect as the silent mock fallback that fabricated 240
# self-consistency samples.
# ---------------------------------------------------------------------------

class TestSpendCap:
    def test_it_binds_in_real_mode(self, monkeypatch, benchmark_task):
        """Raises rather than returning, and raises BEFORE reaching a backend."""
        monkeypatch.setattr(models, "MODE", "real")
        monkeypatch.setattr(models, "MAX_SPEND_USD", 1.0)
        monkeypatch.setattr(models, "backend_spend_usd", 1.5)
        monkeypatch.setattr(response_cache, "get", lambda k: None)
        monkeypatch.setattr(
            models, "_real_call",
            lambda *a, **k: pytest.fail("reached a backend past the spend cap"))

        with pytest.raises(models.SpendCapExceeded):
            models.call("cheap", benchmark_task[0])

    def test_it_does_not_bind_in_mock_or_replay(self, monkeypatch, benchmark_task):
        """Mock and replay charge nothing, so a cap on them is a false positive.

        Not a detail: mock cost_usd is modelled from synthetic token counts and
        accumulates exactly like real cost, so a mode-blind cap would abort free
        mock runs over a large task set for no reason at all.
        """
        monkeypatch.setattr(models, "MODE", "mock")
        monkeypatch.setattr(models, "MAX_SPEND_USD", 0.0)
        monkeypatch.setattr(models, "backend_spend_usd", 99.0)
        monkeypatch.setattr(response_cache, "get", lambda k: None)
        monkeypatch.setattr(response_cache, "put", lambda k, r: None)
        assert models.call("cheap", benchmark_task[0]) is not None

    def test_it_counts_backend_spend_not_attributed_cost(self, monkeypatch, benchmark_task):
        """A cache hit charges the policy but not the card.

        This is the distinction the cap turns on. Attributed cost grows with the
        number of policies and is identical in replay, so capping on it would
        abort a free replay of a large task set.
        """
        monkeypatch.setattr(models, "MODE", "replay")
        models.reset_call_stats()
        assert models.backend_spend_usd == 0.0

        record = {"text": "42", "tier": "cheap", "tokens_in": 10, "tokens_out": 2,
                  "latency_s": 0.1, "cost_usd": 0.50, "mode": "real"}
        monkeypatch.setattr(response_cache, "get", lambda k: record)

        r = models.call("cheap", benchmark_task[0])
        assert r.cost_usd == 0.50          # the policy is charged
        assert models.backend_spend_usd == 0.0   # the card is not

    def test_reset_clears_it(self, monkeypatch):
        monkeypatch.setattr(models, "backend_spend_usd", 3.0)
        models.reset_call_stats()
        assert models.backend_spend_usd == 0.0

    def test_run_eval_still_exposes_the_name(self):
        """STATUS.md and README point readers at run_eval.MAX_SPEND_USD."""
        assert run_eval.MAX_SPEND_USD == models.MAX_SPEND_USD

    def test_the_exception_is_not_aliased_at_import(self):
        """`use_ladder` reloads models, which rebuilds the class object.

        An alias captured at import would then name a class nothing raises, and
        the handler in run_eval would silently stop catching - a spend guard
        that fails open. Both entry points reach through `models.` instead.
        """
        assert not hasattr(run_eval, "SpendCapExceeded")


class TestCacheOffIsRefusedOnThePaidPath:
    def test_real_mode_is_refused(self, monkeypatch):
        """With the cache off, put() discards - so a real run pays and keeps
        nothing. The most expensive failure available in this repository."""
        monkeypatch.setattr(response_cache, "ENABLED", False)
        with pytest.raises(SystemExit):
            response_cache.configure("real", "wide")

    def test_replay_mode_is_refused(self, monkeypatch):
        """Replay IS a cache read; with the cache off it measures nothing."""
        monkeypatch.setattr(response_cache, "ENABLED", False)
        with pytest.raises(SystemExit):
            response_cache.configure("replay", "wide")

    def test_mock_mode_is_allowed(self, monkeypatch):
        """The switch exists to count un-deduplicated calls. That still works."""
        monkeypatch.setattr(response_cache, "ENABLED", False)
        response_cache.configure("mock", "wide")
        # Put the module back the way the rest of the suite expects it.
        response_cache.configure(models.MODE, models.LADDER)


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

    def test_the_correction_only_ever_shrinks_the_headline(self):
        """The ordering that makes the redraw worth buying, as an invariant.

        These assertions used to be magnitudes - n_graded == 94, 15 p_hat
        entries, observed == 0.149. They went stale twice, and the second time
        (the code half going from 35 tasks to 366) they said nothing useful:
        a pinned number cannot tell "the task set moved" from "the correction
        broke".

        What must hold on any task set is the DIRECTION. `observed` counts one
        draw per cell and cannot distinguish "the cheap model cannot do this"
        from "the cheap model usually can and missed once", so it overstates.
        `expected` averages fresh draws. `reproducible` further requires the
        behaviour to be reliable at both rungs. Each step can only remove mass
        it cannot re-confirm, so:

            observed  >=  expected  >=  reproducible

        If that ever inverted, the re-estimate would be manufacturing routing
        opportunities rather than pricing the noise in them, which is the exact
        failure this script exists to prevent.
        """
        path = REPO_ROOT / "redraw.wide.json"
        if not path.exists():
            pytest.skip("redraw.wide.json not present; run scripts/redraw_decisive.py")
        import json
        d = json.loads(path.read_text(encoding="utf-8"))

        assert d["observed"] >= d["expected"] - 1e-9, (
            f"fresh draws found MORE routable mass than the single-draw "
            f"cross-tab: observed {d['observed']:.4f} < expected "
            f"{d['expected']:.4f}")
        assert d["expected"] >= d["reproducible"] - 1e-9, (
            f"requiring reliability found more mass than not requiring it: "
            f"expected {d['expected']:.4f} < reproducible "
            f"{d['reproducible']:.4f}")
        assert 0.0 <= d["reproducible"] <= 1.0
        assert d["noise_share"] == pytest.approx(
            (d["expected"] - d["reproducible"]) / d["expected"], abs=1e-6), (
            "noise_share must be the share of `expected` that `reproducible` "
            "drops, or it is not the quantity its name claims")

    def test_phantoms_are_reported_not_silently_kept(self):
        """A task whose cheap rung solves it on every fresh draw was never a
        routing opportunity - the single-draw cross-tab manufactured it.

        `math-96` was the original: its cheap greedy draw was cut off at
        max_tokens, so the cross-tab read cheap-wrong / expensive-right and
        called it routable, while ten fresh cheap draws solved it 10 times out
        of 10. Phantoms are why `observed` is not the number to quote.
        """
        path = REPO_ROOT / "redraw.wide.json"
        if not path.exists():
            pytest.skip("redraw.wide.json not present; run scripts/redraw_decisive.py")
        import json
        d = json.loads(path.read_text(encoding="utf-8"))
        phantoms = [t for t, p in d["p_hat"].items()
                    if p.get("cheap") is not None and p["cheap"] >= 0.9]
        # Not a bound on how many there are - that is a property of the task
        # set. The invariant is that having them is compatible with the
        # correction shrinking, which the test above asserts.
        assert isinstance(phantoms, list)
        for t in phantoms:
            assert d["p_hat"][t]["cheap"] >= 0.9

    def test_truncated_draws_left_the_denominator(self):
        """p_hat must be a share of GRADEABLE draws, not of requested ones.

        A response cut off at max_tokens never reached its answer, so grading it
        False counts it in the numerator's complement AND the denominator.
        Dropping it must remove both - and the bookkeeping has to agree: every
        draw missing from `draws_used` is one the run reported dropping.
        """
        path = REPO_ROOT / "redraw.wide.json"
        if not path.exists():
            pytest.skip("redraw.wide.json not present; run scripts/redraw_decisive.py")
        import json
        d = json.loads(path.read_text(encoding="utf-8"))

        requested = d["draws"]
        missing = 0
        for task_id, row in d["draws_used"].items():
            for tier, used in row.items():
                assert 0 <= used <= requested, (
                    f"{task_id}/{tier} used {used} of {requested} draws")
                missing += requested - used
        assert missing == d["truncated_draws_dropped"], (
            f"{missing} draw(s) are missing from draws_used but the run "
            f"reported dropping {d['truncated_draws_dropped']}. A draw that "
            f"vanishes without being counted as truncated is a draw graded "
            f"as a failure somewhere.")

        # A rung with every draw truncated must be None, never 0.0 - the
        # difference between "unmeasured" and "the model got it wrong".
        for task_id, row in d["draws_used"].items():
            for tier, used in row.items():
                if used == 0:
                    assert d["p_hat"][task_id][tier] is None, (
                        f"{task_id}/{tier} has no gradeable draws but reports "
                        f"p_hat {d['p_hat'][task_id][tier]}")


# ---------------------------------------------------------------------------
# Truncation is a MISSING measurement, not a wrong answer.
#
# A response cut off at max_tokens never reaches its \boxed{}, so the grader
# scores it False for a reason that is nothing to do with the model. The damage
# is directional rather than random: truncations land on the longest
# derivations, which are the hardest tasks, which are exactly the ones that
# decide the routable fraction. A truncated CHEAP answer reads as "the cheap
# rung failed" and inflates `routable`; a truncated EXPENSIVE one manufactures
# a `both_fail`.
#
# Raising the cap is not the fix - max_tokens is in the cache key, so it would
# strand all 980 committed responses and re-charge $4.2352.
# ---------------------------------------------------------------------------

class TestTruncationIsUnmeasured:
    def test_the_stored_flag_wins_when_present(self):
        assert models.is_truncated({"truncated": True, "tokens_out": 1}) is True
        assert models.is_truncated({"truncated": False, "tokens_out": 99999}) is False

    def test_it_falls_back_to_the_token_count(self):
        """The load-bearing branch, not a courtesy: all three truncations on
        disk predate the `truncated` field and carry None."""
        assert models.is_truncated({"tokens_out": models.MAX_TOKENS}) is True
        assert models.is_truncated({"tokens_out": models.MAX_TOKENS - 1}) is False

    def test_the_fallback_uses_the_right_cap_per_kind(self):
        """A routing call is capped at 8 tokens, not MAX_TOKENS. Judging it by
        the answer cap would call every routing response un-truncated."""
        assert models.is_truncated(
            {"kind": "route", "tokens_out": models.ROUTER_MAX_TOKENS}) is True
        assert models.is_truncated(
            {"kind": "route", "tokens_out": models.ROUTER_MAX_TOKENS - 1}) is False

    def test_a_truncated_greedy_draw_leaves_the_cross_tab(self, wide_verdicts):
        """routable.real_verdicts must not classify what it could not measure.

        Dropping the tier drops the task, because a cross-tab cell needs both
        rungs. That is the correct arithmetic - an unmeasured pair is not a
        cell, it is an absence.
        """
        # math-96's cheap greedy draw hit the cap. It must have no cheap
        # verdict at all, rather than a False one.
        assert "cheap" not in wide_verdicts.get("math-96", {})

    def test_every_surviving_pair_is_reachable_and_unique(self, wide_verdicts):
        """One greedy draw per (task, tier), and it is the one the cache serves.

        real_verdicts reads the raw JSONL rather than going through the cache,
        so before models.is_reachable it also graded 73 rows stranded at the old
        max_tokens=2048 - and the LAST such row on disk silently won.
        """
        complete = [t for t, v in wide_verdicts.items()
                    if "cheap" in v and "expensive" in v]
        # 60 maths tasks, minus math-96 whose cheap rung is unmeasured.
        assert len(complete) == 59
        assert all(t.startswith("math-") for t in complete)


class TestReachability:
    def test_a_record_under_a_superseded_cap_is_unreachable(self, benchmark_task):
        """The orphan case, reproduced from the committed cache.

        max_tokens is in the cache key and NOT in the record, so it cannot be
        read off a row - it has to be inferred by recomputing the key. These
        rows are real, paid for, gradeable, and permanently unservable.
        """
        import json
        import response_cache
        task = next(t for t in benchmark_task if t["id"] == "math-96")
        prompt = models.build_prompt(task, "answer")
        old = response_cache.make_key(
            mode="real", model="deepseek-v4-flash", prompt=prompt,
            temperature=0.0, sample_idx=0, max_tokens=2048, mock_seed=None)
        now = response_cache.make_key(
            mode="real", model="deepseek-v4-flash", prompt=prompt,
            temperature=0.0, sample_idx=0, max_tokens=models.MAX_TOKENS,
            mock_seed=None)
        assert old != now

        rec = {"key": old, "mode": "real", "model": "deepseek-v4-flash",
               "temperature": 0.0, "sample_idx": 0, "kind": "answer"}
        assert models.is_reachable(rec, task) is False
        assert models.is_reachable({**rec, "key": now}, task) is True

    def test_a_record_missing_key_fields_is_not_assumed_reachable(self, benchmark_task):
        assert models.is_reachable({"key": "whatever"}, benchmark_task[0]) is False


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


class TestLadderScopedOutputs:
    """Every artefact-writing entry point must accept an output path.

    The project's thesis is that the sign of the cascading-vs-routing result
    depends on the PRICE RATIO between rungs, and a price ratio is a property of
    a ladder. Testing that claim means holding three ladders' numbers side by
    side - so the moment any analysis script hard-codes its output filename,
    running a second ladder silently overwrites the first.

    That is not hypothetical. `run_eval.guard_regression` exists because on
    8 August 2026 a `deepseek` run overwrote a complete nine-policy `wide` run
    with 47 rows of one policy. The guard caught it; the underlying cause was one
    fixed path shared by every ladder.

    This is a structural tripwire rather than a behavioural test: it fails when
    someone adds a new analysis script with a fixed output path, which is the
    mistake, not the symptom.
    """

    WRITERS = [
        ("run_eval.py", "--out"),
        ("frontier.py", "--out"),
        ("sweep_degraded.py", "--out"),
        ("stats.py", "--results"),
        ("plot.py", "--frontier"),
    ]

    @pytest.mark.parametrize("script,flag", WRITERS)
    def test_it_accepts_an_output_override(self, script, flag):
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / script), "--help"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert proc.returncode == 0, f"{script} --help failed:\n{proc.stderr}"
        assert flag in proc.stdout, (
            f"{script} has no {flag}, so every ladder writes to the same file "
            f"and the second run destroys the first. See "
            f"scripts/run_all_ladders.py."
        )

    def test_the_driver_covers_every_ladder_models_defines(self):
        """A ladder that exists but is never run is an unmeasured comparison.

        `always_mid` only exists on the three-rung `claude` ladder, so a driver
        that quietly skipped it would leave one policy permanently unmeasured
        while every summary still said "all policies".
        """
        import models

        src = (REPO_ROOT / "scripts" / "run_all_ladders.py").read_text(
            encoding="utf-8")
        assert "models.LADDERS" in src, (
            "run_all_ladders.py should read the ladder list from models rather "
            "than hard-coding it, so a new ladder is picked up automatically"
        )
        assert models.LADDERS, "models defines no ladders"


class TestScorecard:
    """The per-policy error attribution must agree with the accuracy it explains.

    `scorecard.py` re-derives what each policy did from two independent sources:
    the result rows, and the cheap-vs-expensive cross-tab built from the raw
    cache. Those can disagree, and when they do every number it prints is wrong.
    Both disagreements found while writing it were real:

      the oracle rescued two `routable` tasks with `calls = ['cheap'] * 5` -
      cheap self-consistency, a third action that a binary escalated/not reading
      files as a missed rescue while the row says `correct: true`;

      `wasted_escalation` tasks are CORRECT - on a both_ok task both rungs get
      it right, so the waste is the money, not the answer.
    """

    def test_the_outcome_buckets_partition_every_task(self):
        import scorecard

        seen = set()
        for cell in ("both_ok", "routable", "both_fail", "inverted"):
            for used_expensive in (True, False):
                for correct in (True, False):
                    name, _ = scorecard.outcome_of(cell, used_expensive, correct)
                    assert name in scorecard.ORDER, (
                        f"{name} is not in ORDER, so it would be silently "
                        f"dropped from the report")
                    seen.add(name)
        assert seen <= set(scorecard.ORDER)

    def test_a_correct_outcome_is_never_also_a_wasteful_mistake_about_accuracy(self):
        """WASTEFUL is about money and CORRECT_OUTCOMES about answers, so
        `wasted_escalation` belongs to both. That is the distinction the
        reconciliation check exists to protect."""
        import scorecard

        assert "wasted_escalation" in scorecard.CORRECT_OUTCOMES
        assert "wasted_escalation" in scorecard.WASTEFUL
        assert "missed_rescue" not in scorecard.CORRECT_OUTCOMES
        assert "harmful_escalation" not in scorecard.CORRECT_OUTCOMES

    def test_resampling_a_routable_task_right_is_not_a_missed_rescue(self):
        """The oracle case, as a unit test."""
        import scorecard

        name, mistake = scorecard.outcome_of("routable", False, correct=True)
        assert name == "rescued_by_resampling" and not mistake
        name, mistake = scorecard.outcome_of("routable", False, correct=False)
        assert name == "missed_rescue" and mistake

    def test_the_buckets_reproduce_accuracy_on_a_synthetic_set(self):
        """The reconciliation invariant, tested without grading 366 tasks.

        This is the arithmetic that was wrong - `wasted_escalation` was left out
        of the correct-answer total - and it is a property of the bucketing, not
        of any particular corpus. One row is placed in every reachable
        (cell, action, correct) combination, so a bucket that stops implying its
        own correctness fails here in milliseconds rather than in three minutes.
        """
        import scorecard

        rows, cells, expected_correct = [], {}, 0
        i = 0
        for cell in ("both_ok", "routable", "both_fail", "inverted"):
            for used_expensive in (True, False):
                # What the cross-tab says the outcome must be: escalating gets
                # the expensive rung's verdict, staying cheap gets the cheap
                # rung's. `rescued_by_resampling` is the one case where a policy
                # beats that, and it is covered by the unit test above.
                correct = ({"both_ok": True, "routable": True,
                            "both_fail": False, "inverted": False}[cell]
                           if used_expensive else
                           {"both_ok": True, "routable": False,
                            "both_fail": False, "inverted": True}[cell])
                tid = f"t{i}"
                i += 1
                cells[tid] = cell
                rows.append({
                    "task_id": tid, "policy": "p", "domain": "code",
                    "correct": correct, "cost_usd": 0.01,
                    "calls": ["cheap", "expensive"] if used_expensive else ["cheap"],
                })
                expected_correct += correct

        acc = scorecard.score(rows, cells)
        d = acc["p"]["all"]
        assert d["n"] == len(rows)
        assert d["correct"] == expected_correct
        got = sum(d["outcomes"][k] for k in scorecard.CORRECT_OUTCOMES)
        assert got == expected_correct, (
            f"buckets say {got} correct, rows say {expected_correct}. "
            f"An outcome moved in or out of CORRECT_OUTCOMES.")

    @pytest.mark.slow
    def test_it_reconciles_against_the_committed_results(self):
        """End to end on real data: the buckets must sum to the reported
        accuracy for every policy.

        Marked slow because it grades every code task through a subprocess -
        3 seconds at 95 tasks, 168 at 426. The fast test above covers the
        arithmetic; this covers the join against the real cross-tab.
        """
        import json
        import subprocess
        import sys

        if not (REPO_ROOT / "results.jsonl").exists():
            pytest.skip("no results.jsonl")
        out = REPO_ROOT / "tests" / "_scorecard_tmp.json"
        try:
            proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scorecard.py"),
                 "--json", str(out)],
                capture_output=True, text=True, cwd=REPO_ROOT,
            )
            assert proc.returncode == 0, proc.stderr
            assert "BUCKETS DO NOT RECONCILE" not in proc.stdout, proc.stdout
            data = json.loads(out.read_text(encoding="utf-8"))
            for policy, groups in data["policies"].items():
                g = groups["all"]
                right = sum(g[k] for k in scorecard_correct())
                assert abs(right / g["n"] - g["accuracy"]) < 1e-9, (
                    f"{policy}: buckets do not reproduce its accuracy")
        finally:
            if out.exists():
                out.unlink()


def scorecard_correct():
    import scorecard
    return scorecard.CORRECT_OUTCOMES


class TestScorecardLadder:
    """The scorecard must configure the ladder from the FILE, not the shell.

    `models` reads ROUTER_LADDER at module scope and builds MODELS/TIERS once.
    Scoring results.claude.jsonl while the environment still said `wide` made
    every cached response fail models.is_reachable - the model ids did not match
    the ladder - so the cross-tab came back empty and the report divided by zero.

    A silently empty cross-tab is the dangerous version of this: with one more
    task classified it would have printed a full table of outcomes computed from
    almost no data, and nothing in the output would have said so.
    """

    def test_it_reads_the_ladder_from_the_results_file(self):
        src = (REPO_ROOT / "scorecard.py").read_text(encoding="utf-8")
        set_at = src.index('os.environ["ROUTER_LADDER"] = ladder')
        import_at = src.index("import run_eval", set_at - 2000)
        assert set_at < import_at, (
            "scorecard.py must set ROUTER_LADDER BEFORE importing run_eval, "
            "which imports models, which reads it at module scope"
        )

    def test_an_empty_crosstab_refuses_rather_than_reports(self):
        """Zero classified tasks must stop the run, not produce a table."""
        src = (REPO_ROOT / "scorecard.py").read_text(encoding="utf-8")
        assert "No task could be classified" in src
        i = src.index("if not total:")
        j = src.index("dynamic range:")
        assert i < j, "the guard must come before the first division by `total`"


class TestCrossLadderVerdicts:
    """`routable.real_verdicts` must see what the CACHE would serve, not one file.

    The ladder is deliberately absent from the cache key, so a response bought
    for one ladder serves any ladder whose rung uses the same model - that is
    what made three ladders affordable here. But it also means a ladder's own
    file is not where its responses live: on 10 August 2026
    raw_calls.claude.jsonl held haiku and sonnet and no Opus, because Opus had
    been bought for `wide`. Reading one file returned a cross-tab with ZERO
    tasks in it for a fully measured ladder.

    The second trap is subtler and worse. Reading all the files but trusting the
    recorded `tier` label would file Opus answers as the deepseek ladder's top
    rung - `wide`'s "expensive" is Opus, `deepseek`'s is v4-pro - and produce a
    clean-looking cross-tab comparing two different models. Rows must be matched
    to rungs by MODEL, which is what response_cache.make_key actually hashes.
    """

    def test_it_reads_every_file_the_cache_would_serve(self):
        src = (REPO_ROOT / "routable.py").read_text(encoding="utf-8")
        assert "_sibling_real_paths" in src, (
            "real_verdicts must consult the same sibling caches "
            "response_cache.configure() does, or it sees fewer responses than "
            "the experiment serves"
        )

    def test_it_matches_rungs_by_model_not_by_recorded_tier(self):
        src = (REPO_ROOT / "routable.py").read_text(encoding="utf-8")
        assert "tier_of_model" in src
        body = src[src.index("def real_verdicts"):src.index("def report(")]
        assert 'd["tier"]' not in body, (
            "a row's `tier` belongs to the ladder it was RECORDED under; using "
            "it across files compares different models under one label"
        )

    @pytest.mark.parametrize("ladder", ["wide", "claude", "deepseek"])
    def test_every_ladder_resolves_its_own_rungs(self, use_ladder, ladder):
        """The mapping must cover each ladder's rungs with distinct models."""
        m = use_ladder(ladder)
        ids = [m.MODELS[t]["id"] for t in m.TIERS]
        assert len(set(ids)) == len(ids), (
            f"{ladder} maps two rungs to one model id, so a response could not "
            f"be attributed to a rung unambiguously"
        )
