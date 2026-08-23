"""The MCP surface.

Tool *descriptions* are tested as well as tool behaviour. A description is the
only thing a client model reads when deciding whether to call a tool, so an
empty or vague one is a functional defect, not a documentation nit.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _import_mcp_call():
    spec = importlib.util.spec_from_file_location(
        "mcp_call", REPO_ROOT / "scripts" / "mcp_call.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def server():
    from router_agent.mcp_server import mcp
    return mcp


class TestRegistration:
    async def test_expected_tools_are_registered(self, server):
        names = {t.name for t in await server.list_tools()}
        assert names == {
            "route_query", "resume_routing", "estimate_cost",
            "compare_policies", "explain_routing",
        }

    async def test_expected_resources_are_registered(self, server):
        uris = {str(r.uri) for r in await server.list_resources()}
        assert uris == {
            "routing://ladders",
            "routing://findings/probe",
            "routing://findings/verifiers",
            "routing://policies",
        }

    async def test_prompt_is_registered(self, server):
        names = {p.name for p in await server.list_prompts()}
        assert "choose_routing_policy" in names

    async def test_every_tool_has_a_substantial_description(self, server):
        """A client model routes on the description alone."""
        for tool in await server.list_tools():
            assert tool.description, f"{tool.name} has no description"
            assert len(tool.description) > 120, (
                f"{tool.name} description is too thin to route on"
            )

    async def test_route_query_description_warns_about_verified(self, server):
        """The one caveat a caller must not miss."""
        tool = next(t for t in await server.list_tools() if t.name == "route_query")
        assert "verifier" in tool.description.lower()
        assert "not a correctness claim" in tool.description.lower()


class TestTools:
    async def test_explain_routing_returns_the_verdict(self, server):
        """The tool serves whatever that ladder's frontier measured.

        Pinned to "route" for deepseek at one point, from a mock-era
        constant that the measurement later contradicted. What the tool owes a
        caller is a verdict traceable to a real run, not a particular one.
        """
        r = await server.call_tool("explain_routing", {"ladder": "deepseek"})
        assert not r.is_error
        d = r.structured_content
        assert d["ratio"]["verdict"] in ("cascade", "route")
        assert d["ratio"]["economics_source"] == "frontier.deepseek.jsonl"
        assert d["ratio"]["economics_simulated"] is False

    async def test_estimate_cost_calls_no_model(self, server):
        from llm_routing import models
        before = models.call_stats["requested"]
        r = await server.call_tool("estimate_cost", {"query": "What is 2+2?"})
        assert not r.is_error
        assert models.call_stats["requested"] == before, (
            "estimate_cost must be free - it made a model call"
        )
        policies = {e["policy"] for e in r.structured_content["estimates"]}
        assert policies == {
            "always_cheap", "always_expensive", "predictive", "cascade"
        }

    async def test_estimate_orders_policies_sensibly(self, server):
        r = await server.call_tool("estimate_cost", {"query": "What is 2+2?"})
        by = {e["policy"]: e for e in r.structured_content["estimates"]}
        assert by["always_cheap"]["max_usd"] < by["always_expensive"]["min_usd"]
        # A cascade's floor is above always_cheap: it also pays to verify.
        assert by["cascade"]["min_usd"] >= by["always_cheap"]["min_usd"]

    async def test_route_query_returns_a_trace(self, server):
        r = await server.call_tool("route_query", {"query": "What is 2+2?"})
        assert not r.is_error
        d = r.structured_content
        assert d["trace"]
        assert [e["node"] for e in d["trace"]][0] == "classify"
        assert d["trace"][-1]["node"] == "finalize"

    async def test_route_query_never_claims_correctness(self, server):
        r = await server.call_tool("route_query", {"query": "What is 2+2?"})
        d = r.structured_content
        assert "correct" not in d
        assert d["verified_meaning"]

    async def test_compare_policies_prices_each(self, server):
        r = await server.call_tool(
            "compare_policies",
            {"query": "What is 2+2?", "policies": ["always_cheap", "always_expensive"]},
        )
        d = r.structured_content
        assert len(d["results"]) == 2
        assert d["cheapest_policy"] == "always_cheap"
        assert d["caveat"]


class TestHumanApproval:
    """The pause is only useful if the client can also finish the run."""

    @pytest.fixture
    def approving(self, monkeypatch):
        """A server whose escalations all need a human."""
        monkeypatch.setenv("ROUTER_APPROVAL_USD", "0")
        from router_agent.mcp_server import mcp
        return mcp

    async def _paused(self, server):
        r = await server.call_tool(
            "route_query",
            {"query": "Compute the integral of x^2", "domain": "math"},
        )
        d = r.structured_content
        assert d["stop_reason"] == "awaiting_approval", d["stop_reason"]
        return d

    async def test_route_query_pauses_with_a_resumable_handle(self, approving):
        d = await self._paused(approving)
        assert d["thread_id"], "a paused run must hand back something to resume"
        assert d["interrupted"]["kind"] == "escalation_approval"
        # The caller is being asked to approve a SPEND, so the payload has to
        # say how much before they can answer.
        assert d["interrupted"]["projected_usd"] > 0
        assert d["escalations"] == 0

    async def test_approving_escalates_and_denying_does_not(self, approving):
        paused = await self._paused(approving)
        denied = await approving.call_tool(
            "resume_routing",
            {"thread_id": paused["thread_id"], "approved": False},
        )
        assert denied.structured_content["stop_reason"] == "approval_denied"
        assert denied.structured_content["escalations"] == 0

        paused = await self._paused(approving)
        ok = await approving.call_tool(
            "resume_routing",
            {"thread_id": paused["thread_id"], "approved": True},
        )
        assert ok.structured_content["escalations"] >= 1
        assert ok.structured_content["cost_usd"] > denied.structured_content["cost_usd"]

    async def test_every_escalation_needs_its_own_approval(self, approving):
        """Regression: resume used to report the next pause as a finished run.

        Approval is per-escalation - the escalate node clears `approved` on the
        way through - so on the three-rung claude ladder the first approval
        walks the graph straight into the second interrupt. `resume` did not
        check `__interrupt__` the way `route` did, and returned an outcome with
        `interrupted=None` and an EMPTY stop_reason, which any drain loop reads
        as finished. A two-rung ladder never reaches the second pause, which is
        why this went unnoticed.
        """
        paused = await self._paused(approving)
        again = (await approving.call_tool(
            "resume_routing",
            {"thread_id": paused["thread_id"], "approved": True},
        )).structured_content
        assert again["stop_reason"] == "awaiting_approval"
        assert again["interrupted"]["to_tier"] == "expensive"
        assert again["escalations"] == 1

    async def test_a_fully_approved_run_finishes_at_the_top_rung(self, approving):
        d = await self._paused(approving)
        approvals = 0
        while d["stop_reason"] == "awaiting_approval":
            approvals += 1
            assert approvals <= 4, "drain loop is not terminating"
            d = (await approving.call_tool(
                "resume_routing",
                {"thread_id": d["thread_id"], "approved": True},
            )).structured_content

        assert approvals == 2, "three rungs is two escalations to approve"
        assert d["escalations"] == 2
        # Same shape as a route_query that never paused.
        assert d["trace"][-1]["node"] == "finalize"
        assert set(d["trace"][0]) == {"node", "detail"}, "trace must be summarised"
        assert "correct" not in d, "serving has no ground truth"
        assert d["verified_meaning"]

    async def test_unknown_thread_is_explained_not_crashed(self, approving):
        """LangGraph answers an unknown thread with a bare KeyError('config')."""
        r = await approving.call_tool(
            "resume_routing", {"thread_id": "deadbeefcafe", "approved": True},
        )
        assert not r.is_error
        assert r.structured_content["error"] == "no_suspended_run"

    async def test_resume_is_refused_when_nothing_can_pause(self, server, monkeypatch):
        monkeypatch.delenv("ROUTER_APPROVAL_USD", raising=False)
        r = await server.call_tool(
            "resume_routing", {"thread_id": "whatever", "approved": True},
        )
        assert r.structured_content["error"] == "approval_not_enabled"

    async def test_resume_is_annotated_as_spending(self, server):
        """It authorises an escalation, so it is not read-only."""
        tool = next(
            t for t in await server.list_tools() if t.name == "resume_routing"
        )
        assert tool.annotations.read_only_hint is False
        assert "approved=false" in (tool.description or "")


class TestListingRenders:
    """`--list` is the first thing anyone runs, and it used to lie.

    It took `description[:78]` and appended a full stop, so any description
    longer than that came back chopped mid-word wearing a period:
    `compare what each return.` reads as a finished sentence and is not one.
    The resource rows were cut at 100 characters with no indicator at all.
    Both rendered without error, which is exactly why nothing caught them -
    the same gap tests/test_figures.py exists to close for the charts.
    """

    @pytest.fixture(scope="class")
    def cli(self):
        return _import_mcp_call()

    def test_a_decimal_does_not_end_a_sentence(self, cli):
        assert cli.first_sentence(
            "Ratios run to 3.11x here. Then more."
        ) == "Ratios run to 3.11x here."

    def test_text_with_no_sentence_end_survives_whole(self, cli):
        assert cli.first_sentence("no full stop here") == "no full stop here"

    def test_entry_never_invents_or_drops_a_word(self, cli):
        text = (
            "Route one query through a cost-aware cascade and return the "
            "answer with a full cost and verification trace."
        )
        rendered = cli.entry("route_query", text, 18)
        # Strip the label column, then the row must be the text back verbatim.
        body = " ".join(rendered.replace("route_query", "", 1).split())
        assert body == text

    def test_entry_wraps_rather_than_truncates(self, cli):
        text = "word " * 60
        rendered = cli.entry("tool", text.strip(), 18)
        assert chr(10) in rendered, "a long description must wrap, not vanish"
        assert len(" ".join(rendered.split())) >= len(text.strip())

    async def test_every_advertised_description_renders_complete(self, server, cli):
        """No rendered row may end mid-sentence."""
        rows = [(t.name, t.description or "") for t in await server.list_tools()]
        rows += [
            (str(r.uri), r.description or "") for r in await server.list_resources()
        ]
        for name, desc in rows:
            shown = cli.first_sentence(desc)
            assert shown.endswith("."), f"{name}: rendered row does not end a sentence"
            assert shown in " ".join(desc.split()), (
                f"{name}: rendered text is not a verbatim prefix of the description"
            )
            rendered = cli.entry(name, shown, 30)
            body = " ".join(rendered.replace(name, "", 1).split())
            assert body == shown, f"{name}: wrapping altered the text"


class TestResources:
    async def _read(self, server, uri):
        chunks = list(await server.read_resource(uri))
        return json.loads(chunks[0].content)

    async def test_ladders_resource_lists_every_ladder(self, server):
        from llm_routing import models
        d = await self._read(server, "routing://ladders")
        assert set(d["ladders"]) == set(models.LADDERS)

    async def test_ladders_resource_flags_verifiability(self, server):
        """Which rungs can be resampled is a real operational constraint."""
        d = await self._read(server, "routing://ladders")
        claude = d["ladders"]["claude"]["rungs"]
        assert claude[0]["verifiable_by_self_consistency"] is True   # Haiku
        assert claude[-1]["verifiable_by_self_consistency"] is False  # Opus 5

    async def test_verifiers_resource_states_the_transfer_gap(self, server):
        d = await self._read(server, "routing://findings/verifiers")
        assert d["verifiers"]["tests"]["transfers_to_production"] is False
        assert d["verifiers"]["self_consistency"]["transfers_to_production"] is True

    async def test_policies_resource_explains_the_tradeoff(self, server):
        d = await self._read(server, "routing://policies")
        assert "cascade" in d and "predictive" in d
        assert "wins_when" in d["cascade"]
