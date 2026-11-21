"""Gemini Managed Agents integration for autonomous hospital unit auditing."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from hospital_twin.schemas import SimulationComparison, UnitMap
from hospital_twin.world_model import analyze_floorplan_delegation, rank_bottlenecks

logger = logging.getLogger(__name__)


class ManagedAgentAuditRequest(BaseModel):
    unit_map: UnitMap
    comparison: Optional[SimulationComparison] = None
    goal: str = "Audit unit bottlenecks and recommend optimal mobile + stationary robotics deployment."


class ManagedAgentAuditResponse(BaseModel):
    mode: Literal["managed_agent", "fallback"]
    agent_id: str
    interaction_id: str
    recommended_fleet: str
    key_findings: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    summary: str


def _build_fallback_audit(
    unit: UnitMap,
    comparison: Optional[SimulationComparison],
) -> ManagedAgentAuditResponse:
    delegation = analyze_floorplan_delegation(unit)
    findings = [
        f"Unit has {len(unit.rooms)} rooms and {len(unit.doors)} doorways.",
        f"Delegation split: {delegation.mobile_count} mobile, {delegation.stationary_count} stationary, "
        f"{delegation.human_only_count} human-only, {delegation.rd_gap_count} R&D gaps.",
    ]
    if comparison is not None:
        bottlenecks = rank_bottlenecks(unit, comparison)
        findings.append(
            f"Shift simulation saved {comparison.nurse_minutes_saved:.1f} nurse walking minutes "
            f"(avg wait {comparison.robot_assisted.avg_delivery_wait_s:.1f}s)."
        )
        if bottlenecks:
            findings.append(f"Top bottleneck: {bottlenecks[0].source} ({bottlenecks[0].impact_minutes:.1f}m).")

    actions = [
        f"{c.title} ({c.estimated_bom_usd}): {c.target_problem}"
        for c in delegation.custom_rd_concepts
    ]
    rec_fleet = "mixed_fleet" if delegation.rd_gap_count > 0 else "aethon_tug"
    return ManagedAgentAuditResponse(
        mode="fallback",
        agent_id="amed-local-auditor",
        interaction_id="local-fallback",
        recommended_fleet=rec_fleet,
        key_findings=findings,
        action_items=actions,
        summary=f"Recommended fleet: {rec_fleet}. {findings[1]}",
    )


def run_managed_agent_audit(req: ManagedAgentAuditRequest) -> ManagedAgentAuditResponse:
    """Create and invoke a Gemini Managed Agent (`client.agents` + `client.interactions`) to audit the unit."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _build_fallback_audit(req.unit_map, req.comparison)

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        delegation = analyze_floorplan_delegation(req.unit_map)
        context_payload: dict[str, Any] = {
            "goal": req.goal,
            "rooms": len(req.unit_map.rooms),
            "doors": len(req.unit_map.doors),
            "delegation_summary": {
                "mobile": delegation.mobile_count,
                "stationary": delegation.stationary_count,
                "human_only": delegation.human_only_count,
                "rd_gaps": delegation.rd_gap_count,
            },
            "custom_rd_concepts": [c.model_dump() for c in delegation.custom_rd_concepts],
        }
        if req.comparison is not None:
            context_payload["simulation"] = {
                "nurse_minutes_saved": req.comparison.nurse_minutes_saved,
                "avg_wait_s": req.comparison.robot_assisted.avg_delivery_wait_s,
                "flagged_exceptions": req.comparison.robot_assisted.flagged_exceptions,
            }

        agent = client.agents.create(
            agent_config={"model": "gemini-4-neon"},
            description="Amed Autonomous Hospital Unit Robotics Auditor",
            system_instruction=(
                "You are an autonomous hospital robotics systems engineer. "
                "Given unit floor-plan delegation metrics and shift simulation results, "
                "return strict JSON with keys: recommended_fleet (string), key_findings (list of strings), "
                "action_items (list of strings), summary (string)."
            ),
        )
        agent_id = getattr(agent, "id", None) or getattr(agent, "name", None) or "amed-managed-agent"

        interaction = client.interactions.create(
            agent=getattr(agent, "name", agent_id),
            input=json.dumps(context_payload),
        )
        interaction_id = getattr(interaction, "id", None) or getattr(interaction, "name", None) or "interaction-1"

        raw_text = ""
        if hasattr(interaction, "outputs") and interaction.outputs:
            last = interaction.outputs[-1]
            raw_text = getattr(last, "text", "") or str(last)
        elif hasattr(interaction, "text"):
            raw_text = interaction.text or ""

        parsed = json.loads(raw_text) if raw_text.strip().startswith("{") else {}
        fallback = _build_fallback_audit(req.unit_map, req.comparison)
        return ManagedAgentAuditResponse(
            mode="managed_agent",
            agent_id=str(agent_id),
            interaction_id=str(interaction_id),
            recommended_fleet=str(parsed.get("recommended_fleet") or fallback.recommended_fleet),
            key_findings=list(parsed.get("key_findings") or fallback.key_findings),
            action_items=list(parsed.get("action_items") or fallback.action_items),
            summary=str(parsed.get("summary") or raw_text[:280] or fallback.summary),
        )
    except Exception as exc:
        logger.warning("Managed Agents API failed, using fallback: %s", exc)
        return _build_fallback_audit(req.unit_map, req.comparison)
