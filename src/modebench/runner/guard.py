"""The two guards that stop a run before it spends money or leaks data.

Both guards are code, not convention. The command line calls them, and the
executor calls them again, so a caller that skips the command line cannot
skip the guards.
"""

from collections.abc import Iterable, Sequence

from modebench.config import Mode, ProviderConfig
from modebench.errors import CostCeilingExceeded, PrivacyViolation

FREE_SUFFIX = ":free"


def privacy_problems(mode: Mode, provider: ProviderConfig) -> list[str]:
    """Return the reasons for which `mode` must not see private data.

    A `:free` model is never acceptable: a free endpoint can keep the prompts
    or train on them. An OpenRouter route must say `data_collection = "deny"`
    in its raw parameters. A direct API has no such parameter, so the mode
    must say `private_data_ok = true`, which is the statement of the operator
    that the account has a data policy that permits private data.
    """
    if provider.privacy == "local":
        return []
    problems: list[str] = []
    if FREE_SUFFIX in mode.model.lower():
        problems.append("the model is a :free variant")
    if provider.privacy == "openrouter":
        routing = mode.params.get("provider")
        denies = isinstance(routing, dict) and routing.get("data_collection") == "deny"
        if not denies:
            problems.append('the route does not set provider.data_collection = "deny"')
    elif not mode.private_data_ok:
        problems.append("the mode does not set private_data_ok = true")
    return problems


def enforce_privacy(
    has_private_data: bool, routes: Iterable[tuple[Mode, ProviderConfig]]
) -> None:
    """Raise PrivacyViolation if private data would go to a route that is not safe.

    The run stops as a whole. It does not continue with the safe modes only,
    because a partial run that looks complete is a worse result than no run.
    """
    if not has_private_data:
        return
    lines: list[str] = []
    for mode, provider in routes:
        for problem in privacy_problems(mode, provider):
            lines.append(f"- {mode.id}: {problem}")
    if lines:
        details = "\n".join(lines)
        raise PrivacyViolation(
            "A private dataset is selected, and these modes must not receive it:\n"
            f"{details}\n"
            "Remove the private dataset, remove the modes, or correct their routes."
        )


def enforce_cost_ceiling(
    estimate_usd: float, ceiling_usd: float, unknown_price_modes: Sequence[str] = ()
) -> None:
    """Raise CostCeilingExceeded if the estimate is above the ceiling or not complete."""
    if unknown_price_modes:
        names = ", ".join(unknown_price_modes)
        raise CostCeilingExceeded(
            f"The cost estimate is not complete. These modes have no price: {names}. "
            "Add a [modes.price] table, or run the preflight so that the catalogue gives it."
        )
    if estimate_usd > ceiling_usd:
        raise CostCeilingExceeded(
            f"The estimate is {estimate_usd:.2f} USD and the ceiling is {ceiling_usd:.2f} USD. "
            "Use a smaller profile, select fewer modes, or increase the ceiling."
        )


def check_running_cost(spent_usd: float, ceiling_usd: float) -> bool:
    """Return True if the run must stop because the money spent is above the ceiling."""
    return spent_usd > ceiling_usd
