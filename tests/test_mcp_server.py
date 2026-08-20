"""The MCP surface.

Tool *descriptions* are tested as well as tool behaviour. A description is the
only thing a client model reads when deciding whether to call a tool, so an
empty or vague one is a functional defect, not a documentation nit.
"""

from __future__ import annotations

import json

import pytest

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
            "route_query", "estimate_cost", "compare_policies", "explain_routing"
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
