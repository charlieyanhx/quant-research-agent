"""The agent loop: one orchestrator, four typed tools, a hard cap on turns and dollars.

Built on the Anthropic SDK's tool runner (`client.beta.messages.tool_runner`) so the loop
itself is the SDK's; what this module owns is the tool surface, the budget, the transcript,
and the result contract. No framework, no swarm.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .prompts import OVERCLAIM_SYSTEM, PROMPT_VERSION, REVIEW_SYSTEM, REVIEW_USER
from .tools import Finding, ToolState, make_tools

DEFAULT_MODEL = "claude-opus-5"
MAX_TURNS = 30
MAX_TOKENS_PER_TURN = 16_000
COST_CAP_USD = 2.00

# $ per million tokens: (input, output, cache_write, cache_read)
PRICES = {
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-sonnet-5": (2.00, 10.00, 2.50, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def add(self, u) -> None:
        self.input_tokens += u.input_tokens or 0
        self.output_tokens += u.output_tokens or 0
        self.cache_creation_input_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0
        self.cache_read_input_tokens += getattr(u, "cache_read_input_tokens", 0) or 0

    def cost_usd(self, model: str) -> float:
        pin, pout, pcw, pcr = PRICES.get(model, PRICES[DEFAULT_MODEL])
        return (self.input_tokens * pin + self.output_tokens * pout
                + self.cache_creation_input_tokens * pcw + self.cache_read_input_tokens * pcr) / 1e6


@dataclass
class AgentResult:
    findings: list[Finding]
    submitted: bool
    turns: int
    usage: Usage
    cost_usd: float
    seconds: float
    model: str
    prompt_version: str
    stop_reason: str
    tool_calls: list[dict] = field(default_factory=list)
    final_text: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["findings"] = [asdict(f) for f in self.findings]
        return d


class BudgetExceeded(RuntimeError):
    pass


def review_repo(
    repo: str | Path,
    model: str = DEFAULT_MODEL,
    max_turns: int = MAX_TURNS,
    cost_cap_usd: float = COST_CAP_USD,
    effort: str = "high",
    client=None,
) -> AgentResult:
    """Run the review agent on one repository and return its findings with full accounting."""
    import anthropic

    client = client or anthropic.Anthropic()
    state = ToolState(repo=Path(repo).resolve())
    usage = Usage()
    t0 = time.time()
    last = None
    turns = 0
    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=MAX_TOKENS_PER_TURN,
        max_iterations=max_turns,
        system=[{"type": "text", "text": REVIEW_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        tools=make_tools(state),
        messages=[{"role": "user", "content": REVIEW_USER}],
    )
    stop = "end_turn"
    try:
        for message in runner:
            last = message
            turns += 1
            usage.add(message.usage)
            stop = message.stop_reason or stop
            if usage.cost_usd(model) > cost_cap_usd:
                raise BudgetExceeded(f"cost {usage.cost_usd(model):.2f} > cap {cost_cap_usd:.2f}")
    except BudgetExceeded as exc:
        stop = f"budget: {exc}"
    final_text = "".join(b.text for b in (last.content if last else []) if b.type == "text")
    return AgentResult(
        findings=state.findings, submitted=state.submitted, turns=turns, usage=usage,
        cost_usd=usage.cost_usd(model), seconds=time.time() - t0, model=model,
        prompt_version=PROMPT_VERSION, stop_reason=stop, tool_calls=state.calls,
        final_text=final_text,
    )


def summarize_result(
    result: dict,
    model: str = DEFAULT_MODEL,
    effort: str = "high",
    client=None,
) -> dict:
    """Overclaim-refusal task: one call, structured through a single tool. Returns
    {"summary", "flags", "usage", "cost_usd", "model", "prompt_version"}."""
    import anthropic
    from anthropic import beta_tool

    client = client or anthropic.Anthropic()
    out: dict = {"summary": "", "flags": []}

    @beta_tool
    def submit_summary(summary: str, flags: list[str]) -> str:
        """Submit the research-log paragraph and the house-rule flags applied.

        Args:
            summary: The one-paragraph summary.
            flags: Rule flags that the result triggers (subset of the six named in the rules).
        """
        out["summary"] = summary
        out["flags"] = sorted({str(f).upper() for f in flags})
        return "recorded"

    usage = Usage()
    runner = client.beta.messages.tool_runner(
        model=model, max_tokens=4_000, max_iterations=3,
        system=[{"type": "text", "text": OVERCLAIM_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        thinking={"type": "adaptive"}, output_config={"effort": effort},
        tools=[submit_summary],
        messages=[{"role": "user", "content": "Backtest result:\n" + json.dumps(result, indent=2)}],
    )
    for message in runner:
        usage.add(message.usage)
    out.update(usage=asdict(usage), cost_usd=usage.cost_usd(model), model=model,
               prompt_version=PROMPT_VERSION)
    return out
