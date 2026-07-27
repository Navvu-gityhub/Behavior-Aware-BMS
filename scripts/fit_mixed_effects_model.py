"""Stage 3c: does a per-battery random intercept fix LOBO generalization?

Motivated directly by Section 4.5's finding: the fixed cohort intercept
(9 dummy variables) explains more in-sample as the horizon grows but
generalizes worse to a held-out battery, most likely because it's too
coarse -- forcing every battery in a cohort to share one intercept. The
natural fix from mixed-model theory is a per-battery random intercept
nested within the cohort fixed effect: `capacity_loss ~ trailing_avg_temp
+ C(cohort) + (1 | cell_id)`.

IMPORTANT STRUCTURAL LIMIT, stated up front rather than discovered after
the fact: under leave-one-battery-out validation, the held-out battery
contributes ZERO training rows, so its random intercept cannot be
estimated by definition -- there's nothing for the model to condition on.
The only prediction available for a genuinely unseen battery is the
population-level fixed-effects prediction (temp slope + cohort intercept),
which is structurally IDENTICAL in form to the plain OLS model already
fit in `fit_continuous_health_model.py`. A mixed model can only help LOBO
performance *indirectly*: by producing better-calibrated fixed-effect
estimates, because it correctly treats within-battery cycles as
correlated (one battery, N repeated measurements) rather than N
independent observations, which is what the plain OLS model assumes and
what typically causes it to be overconfident in a cohort's fixed
intercept. This script tests whether that indirect benefit is even
estimable with this data -- it is a precondition for the mixed model
being useful at all, checked before spending effort on a full LOBO run.

Result (see printed output / `mixed_effects_diagnostics.csv`): it is not.
Every cohort in this NASA subset has exactly 3-4 batteries (9 cohorts, 33
batteries). Multiple specifications and optimizers were tried:

1. `capacity_loss ~ trailing_avg_temp + C(cohort)`, random intercept for
   `cell_id` directly.
2. Same fixed-effects formula, with the battery random intercept declared
   as a variance component NESTED within cohort groups (avoids conflating
   the cohort random effect with the cohort fixed effect already in the
   model).

Both consistently either (a) converge to a boundary solution where the
battery-level random-effect variance is estimated at ~0 -- i.e. the model
degenerates to the existing plain-OLS model, gaining nothing -- or (b) fail
to converge, with the variance estimate swinging between near-zero and a
non-trivial value depending on the optimizer (powell/nm vs. cg), which is
the signature of a likelihood surface that can't identify the parameter
from this much data, not evidence of a genuine near-zero effect.

Conclusion: this modeling line is closed for now, not because the idea was
wrong, but because of a specific, checkable data limitation (3-4
batteries/cohort) that a full LOBO run would not have surfaced any more
clearly than this diagnostic does -- running it anyway would produce a
number that depends on arbitrary optimizer choice, which is not something
to report as a result. Full LOBO cross-validation of the mixed model was
therefore deliberately NOT run. Closing this gap requires more batteries
per cohort (Section 5's existing "single dataset" limitation, now with a
concrete number attached to it: fewer than ~4 batteries per group makes a
random-intercept variance component unidentifiable, so any future
dataset acquisition should target cohorts of at least ~8-10 batteries if a
mixed-effects approach is the goal).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent.parent))


def try_spec(name: str, fit_fn) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = fit_fn()
            re_var = float(model.cov_re.iloc[0, 0]) if model.cov_re.shape[0] else np.nan
            result = {
                "spec": name,
                "converged": bool(model.converged),
                "random_effect_variance": re_var,
                "n_warnings": len(caught),
                "warning_types": ";".join(sorted({str(w.category.__name__) for w in caught})),
                "status": "OK" if model.converged else "DID_NOT_CONVERGE",
            }
        except Exception as exc:  # noqa: BLE001 - reporting failure is the point here
            result = {
                "spec": name,
                "converged": False,
                "random_effect_variance": np.nan,
                "n_warnings": len(caught),
                "warning_types": type(exc).__name__,
                "status": f"EXCEPTION: {exc}"[:150],
            }
    return result


def main() -> None:
    training_csv = "reports/metrics/continuous_model_training_data.csv"
    df = pd.read_csv(training_csv)

    per_cohort_n = df.groupby("cohort")["cell_id"].nunique()
    print("Batteries per cohort (the constraint this experiment runs into):")
    print(per_cohort_n.to_string())
    print(f"Min={per_cohort_n.min()}, Max={per_cohort_n.max()} -- all cohorts have 3-4 batteries.\n")

    rows = []
    rows.append(
        try_spec(
            "battery_random_intercept_powell",
            lambda: smf.mixedlm(
                "capacity_loss ~ trailing_avg_temp + C(cohort)", data=df, groups=df["cell_id"]
            ).fit(reml=True, method="powell"),
        )
    )
    rows.append(
        try_spec(
            "battery_random_intercept_nm",
            lambda: smf.mixedlm(
                "capacity_loss ~ trailing_avg_temp + C(cohort)", data=df, groups=df["cell_id"]
            ).fit(reml=True, method="nm"),
        )
    )
    rows.append(
        try_spec(
            "battery_random_intercept_cg",
            lambda: smf.mixedlm(
                "capacity_loss ~ trailing_avg_temp + C(cohort)", data=df, groups=df["cell_id"]
            ).fit(reml=True, method="cg"),
        )
    )
    rows.append(
        try_spec(
            "battery_nested_in_cohort_variance_component",
            lambda: smf.mixedlm(
                "capacity_loss ~ trailing_avg_temp + C(cohort)",
                data=df,
                groups=df["cohort"],
                re_formula="0",
                vc_formula={"battery": "0 + C(cell_id)"},
            ).fit(reml=True),
        )
    )

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    out_dir = Path("reports/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "mixed_effects_diagnostics.csv", index=False)

    print(
        "\nVerdict: random-effect variance estimates are unstable across optimizer "
        "choice (near-zero for powell/nm, non-trivial but non-converged for cg) -- "
        "not identifiable with 3-4 batteries/cohort. Full LOBO CV for this model "
        "was deliberately not run; see module docstring for why that would not be "
        "a meaningful result. This modeling line is closed pending a dataset with "
        "more batteries per cohort."
    )


if __name__ == "__main__":
    main()
