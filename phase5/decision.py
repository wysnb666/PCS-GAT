from __future__ import annotations

from dataclasses import dataclass

from MPC4plus.phase5.decomposition import Phase5Buckets


R_CONST = (15 + 505 ** 0.5) / 20


@dataclass
class Phase5Decision:
    recurse_on_Gc: bool
    lhs_value: float
    rhs_value: float
    has_critical: bool
    reason: str


def decide_phase5_route(buckets: Phase5Buckets) -> Phase5Decision:
    """
    Step 4 / 5 of Section 3.6:

        if there exists a critical component and
           sum_{i=1}^5 i|Ki| <= (5/7) r (|K1,c| + 2|K2,c|),
        recurse on Gc;
        otherwise finish directly.
    """
    lhs = (
        1 * len(buckets.K1) +
        2 * len(buckets.K2) +
        3 * len(buckets.K3) +
        4 * len(buckets.K4) +
        5 * len(buckets.K5)
    )
    rhs = (5 / 7) * R_CONST * (len(buckets.K1c) + 2 * len(buckets.K2c))

    has_critical = (len(buckets.K1c) + len(buckets.K2c)) > 0
    recurse = has_critical and (lhs <= rhs)

    if recurse:
        reason = "critical components exist and lhs <= rhs, recurse on G_c"
    elif not has_critical:
        reason = "no critical component exists, finish directly"
    else:
        reason = "critical components exist but lhs > rhs, finish directly"

    return Phase5Decision(
        recurse_on_Gc=recurse,
        lhs_value=lhs,
        rhs_value=rhs,
        has_critical=has_critical,
        reason=reason,
    )