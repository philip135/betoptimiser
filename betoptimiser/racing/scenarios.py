"""
Racing Scenarios – enumerate all ordered finish permutations.
"""

from __future__ import annotations

import itertools
import logging

logger = logging.getLogger(__name__)


def runner_names(catalogues: list) -> list[str]:
    """Deduplicate runner names across WIN / PLACE catalogues for one race."""
    names = []
    seen: set[str] = set()
    for cat in catalogues:
        for r in cat.runners:
            if r.runner_name not in seen:
                names.append(r.runner_name)
                seen.add(r.runner_name)
    return names


def build_scenarios(
    runners: list[str],
    max_places: int = 3,
) -> tuple[list[str], list[tuple[str, ...]]]:
    """
    Enumerate ALL ordered finish scenarios for a race.

    Each scenario is a tuple of runner names: (1st, 2nd, 3rd, ...) of
    length min(max_places, len(runners)).

    For N runners and depth D, this is P(N, D) = N! / (N-D)! scenarios.

    Returns
    -------
    scenario_labels : list[str]
        Human-readable: "1st=X | 2nd=Y | 3rd=Z"
    scenarios : list[tuple[str, ...]]
        Each tuple = (runner_1st, runner_2nd, runner_3rd, ...)
    """
    n = len(runners)
    depth = min(max_places, n)

    scenario_labels: list[str] = []
    scenarios: list[tuple[str, ...]] = []

    ordinals = ["1st", "2nd", "3rd", "4th", "5th", "6th"]

    for perm in itertools.permutations(runners, depth):
        label = " | ".join(
            f"{ordinals[k] if k < len(ordinals) else f'{k+1}th'}={perm[k]}"
            for k in range(depth)
        )
        scenarios.append(perm)
        scenario_labels.append(label)

    logger.debug(
        f"Racing scenarios: {n} runners, depth={depth} → "
        f"{len(scenarios)} ordered scenarios"
    )
    return scenario_labels, scenarios
