"""Semi-empirical Arrhenius degradation model, and why it is the transfer hypothesis.

THE PROBLEM THIS ADDRESSES
--------------------------
This project established two things about temperature. It is the one
behavioural signal with a real, cohort-controlled, correctly-signed
relationship to measured capacity fade (rho 0.21-0.22, p < 0.0001, 7 of 7
cells across independent cohorts). And a model built on it as a *fitted linear
coefficient* does not survive leave-one-cohort-out: ADR 0002 measured
rho = -0.295 on an unseen protocol, with the apparent skill living in the
per-cohort intercepts rather than the temperature term.

Those findings are usually read as "temperature matters but does not
transfer." This module proposes a different reading: **a linear temperature
coefficient is the wrong functional form, and its failure to transfer is what
that wrongness looks like.**

Capacity fade is rate-limited by thermally activated side reactions —
principally SEI growth. The rate of such a reaction is not linear in
temperature. It follows Arrhenius:

    k(T) = A * exp(-Ea / (R * T))

with T in kelvin. A coefficient fitted linearly against degrees Celsius is a
local tangent to an exponential, and the slope of a tangent depends on where
it was taken. Fit it on a 4 C cohort and apply it at 43 C and it is simply the
wrong number — which is a complete mechanical explanation for a coefficient
that reverses sign across cohorts, requiring no appeal to memorisation.

WHY THIS IS A FALSIFIABLE CLAIM AND NOT A STORY
-----------------------------------------------
`Ea` is an activation energy in J/mol. It is a property of the chemistry, not
of the experiment, so unlike a fitted linear slope it *should* transfer across
protocols of the same cell type. That is a prediction, and the existing
leave-one-cohort-out harness tests it directly: if the Arrhenius
parameterisation is right, its LOBO-to-LOCO degradation should be materially
smaller than a linear model's on the same data.

It also has an external check no purely fitted model has. Published activation
energies for lithium-ion capacity fade cluster in roughly 20-80 kJ/mol
(Wang et al. 2011 report ~31 kJ/mol for LFP cycling; SEI-growth studies
commonly land in the 30-60 kJ/mol band). A fit landing far outside that is
evidence the model is absorbing something other than thermal activation, and
`PLAUSIBLE_EA_RANGE` makes that check explicit rather than leaving it to a
reader's judgement.

THE PREDICTION WAS TESTED HERE AND IT FAILED
--------------------------------------------
It should be said plainly, because this module was written to advance the
hypothesis above and the data did not support it.

Fitted on the NASA cycle-level frame, this model returns **Ea = -31 kJ/mol**:
negative, meaning degradation *slows* with heat. Per-cohort fits scatter from
-568 to +183 kJ/mol, with one of nine cohorts inside the published range.
Correcting for the reversible thermal measurement artifact documented in
`thermal_confound.py` does not rescue it — Ea moves to -25.4 kJ/mol and over a third
the observations are lost, because after correction many cells show no net
fade to take a logarithm of.

The implementation is not at fault; `tests/test_physics.py` recovers a known
Ea from synthetic data to within 1%. What the fit is faithfully reporting is
that **an activation energy is not identifiable from this frame**, for three
compounding reasons:

1. Capacity is not measured at a common temperature, so the target itself
   carries a large reversible thermal offset that runs opposite to real
   thermal degradation (`thermal_confound.py`).
2. Ambient temperature is nearly collinear with protocol — the 4 C cells *are*
   the COLD4C cohorts, with their own loads, cutoffs and cell batches — so a
   thermal coefficient and a protocol effect cannot be separated.
3. Within-cohort temperature variation, which is the only contrast free of
   (2), is an order of magnitude smaller than the between-cohort spread.

So the honest status of this module is: correct code, physically motivated
form, **refused on the available data**. `assess_identifiability` performs the
three checks above and returns that refusal before anything is fitted, in the
same spirit as `adaptive/datasets.assess_suitability`. The requirement it
implies — reference discharges at a common temperature, and temperature varied
*within* protocol — is a concrete acceptance criterion for the next dataset,
alongside the "8-10+ batteries per cohort" criterion Section 4.6 established.

THE MODEL
---------
Cumulative fade after N cycles at effective temperature T:

    fade(N, T) = A * exp(-Ea / (R * T)) * N^z

Taking logarithms makes it linear in its parameters:

    ln(fade) = ln(A) - (Ea/R) * (1/T) + z * ln(N)

so ordinary least squares on the design matrix [1, 1/T, ln N] recovers ln(A),
-(Ea/R) and z directly, and Ea = -R * beta_(1/T). The power-law exponent z
carries its own physical reading: z = 0.5 is diffusion-limited SEI growth,
z = 1 is reaction-limited.

TWO HONEST CAVEATS
------------------
**Log-space fitting is biased on back-transformation.** Least squares in log
space estimates the conditional *median*, so exponentiating gives a median
prediction, not a mean. `smearing` applies Duan's nonparametric correction
when a mean is wanted. It is off by default because the harness scores rank
and R-squared, where the median is the appropriate point estimate, and
applying an unrequested correction would make these numbers non-comparable
with the rest of the project's results.

**Only positive fade can be fitted.** Cycles recording zero or negative fade
(measurement noise, or a genuine capacity-recovery rest effect) cannot be
log-transformed and are dropped, with the count reported on the fit. That is a
real selection effect: dropping the low tail biases the intercept upward. It
is reported rather than hidden because the alternative — adding an epsilon to
make the logarithm defined — silently invents data at exactly the point where
the measurement is least trustworthy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.bms.adaptive.validation import FitFn

# Molar gas constant, J/(mol*K).
GAS_CONSTANT = 8.314462618

KELVIN_OFFSET = 273.15

# Published activation energies for lithium-ion capacity fade. Outside this
# band the fit is reported as implausible — see the module docstring.
PLAUSIBLE_EA_RANGE = (20_000.0, 80_000.0)

# Wider band beyond which the fit is treated as failed rather than merely
# suspect. A negative Ea would mean degradation slowing with temperature.
ADMISSIBLE_EA_RANGE = (0.0, 200_000.0)

DEFAULT_TEMPERATURE_COLUMN = "avg_temp"
DEFAULT_CYCLE_COLUMN = "cycle"


@dataclass(frozen=True)
class ArrheniusFit:
    """A fitted Arrhenius power-law model, with its physical reading."""

    ln_a: float
    activation_energy_j_per_mol: float
    exponent_z: float
    r2_log_space: float
    n_used: int
    n_dropped_nonpositive: int
    temperature_column: str
    smearing_factor: float = 1.0

    @property
    def activation_energy_kj_per_mol(self) -> float:
        return self.activation_energy_j_per_mol / 1000.0

    @property
    def plausible(self) -> bool:
        low, high = PLAUSIBLE_EA_RANGE
        return bool(low <= self.activation_energy_j_per_mol <= high)

    @property
    def admissible(self) -> bool:
        low, high = ADMISSIBLE_EA_RANGE
        return bool(np.isfinite(self.activation_energy_j_per_mol)
                    and low <= self.activation_energy_j_per_mol <= high)

    @property
    def rate_limiting_reading(self) -> str:
        """What the power-law exponent implies about the mechanism."""
        if not np.isfinite(self.exponent_z):
            return "undetermined"
        if self.exponent_z < 0.35:
            return "sub-diffusive (z < 0.35) — slower than SEI diffusion growth"
        if self.exponent_z < 0.7:
            return "diffusion-limited (z ~ 0.5), consistent with SEI growth"
        if self.exponent_z < 1.3:
            return "reaction-limited (z ~ 1), fade roughly linear in cycles"
        return "super-linear (z > 1.3) — accelerating fade, possibly late-life"

    def render(self) -> str:
        flag = "" if self.plausible else "  <-- OUTSIDE PUBLISHED RANGE"
        return "\n".join([
            f"Arrhenius power-law fit ({self.temperature_column})",
            f"  Ea = {self.activation_energy_kj_per_mol:.1f} kJ/mol{flag}",
            f"  z  = {self.exponent_z:.3f}  ({self.rate_limiting_reading})",
            f"  ln A = {self.ln_a:.3f}",
            f"  R2 (log space) = {self.r2_log_space:.4f}",
            f"  n = {self.n_used} used, {self.n_dropped_nonpositive} dropped "
            f"(non-positive fade cannot be log-transformed)",
        ])


def _design_matrix(
    temperature_c: np.ndarray,
    cycle: np.ndarray,
) -> np.ndarray:
    """[1, 1/T_kelvin, ln(N)] — the linearised Arrhenius power law."""
    kelvin = temperature_c + KELVIN_OFFSET
    return np.column_stack([
        np.ones(len(kelvin)),
        1.0 / kelvin,
        np.log(cycle),
    ])


def fit_arrhenius(
    data: pd.DataFrame,
    target: str = "cumulative_fade",
    temperature_col: str = DEFAULT_TEMPERATURE_COLUMN,
    cycle_col: str = DEFAULT_CYCLE_COLUMN,
    smearing: bool = False,
) -> ArrheniusFit:
    """Fit ln(fade) = ln A - (Ea/R)(1/T) + z ln(N) by ordinary least squares.

    Rows with non-positive fade, non-positive cycle index, or a temperature at
    or below absolute zero are dropped; the count is recorded on the returned
    fit. See the module docstring for why an epsilon is not added instead.
    """
    for column in (target, temperature_col, cycle_col):
        if column not in data.columns:
            raise ValueError(f"fit_arrhenius: missing required column '{column}'")

    frame = data[[target, temperature_col, cycle_col]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    fade = frame[target].to_numpy(dtype=float)
    temperature = frame[temperature_col].to_numpy(dtype=float)
    cycle = frame[cycle_col].to_numpy(dtype=float)

    keep = (fade > 0) & (cycle > 0) & (temperature + KELVIN_OFFSET > 0)
    n_dropped = int((~keep).sum())

    if keep.sum() < 4:
        return ArrheniusFit(
            ln_a=float("nan"), activation_energy_j_per_mol=float("nan"),
            exponent_z=float("nan"), r2_log_space=float("nan"),
            n_used=int(keep.sum()), n_dropped_nonpositive=n_dropped,
            temperature_column=temperature_col,
        )

    design = _design_matrix(temperature[keep], cycle[keep])
    response = np.log(fade[keep])

    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    ln_a, beta_inv_t, z = (float(c) for c in coefficients)

    residuals = response - design @ coefficients
    total = float(np.sum((response - response.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residuals ** 2)) / total if total > 0 else float("nan")

    # Duan's smearing estimate: E[exp(residual)], the multiplicative correction
    # from a median back-transform to a mean one.
    factor = float(np.mean(np.exp(residuals))) if smearing else 1.0

    return ArrheniusFit(
        ln_a=ln_a,
        activation_energy_j_per_mol=-beta_inv_t * GAS_CONSTANT,
        exponent_z=z,
        r2_log_space=r2,
        n_used=int(keep.sum()),
        n_dropped_nonpositive=n_dropped,
        temperature_column=temperature_col,
        smearing_factor=factor,
    )


def predict_arrhenius(
    fit: ArrheniusFit,
    data: pd.DataFrame,
    temperature_col: str | None = None,
    cycle_col: str = DEFAULT_CYCLE_COLUMN,
) -> np.ndarray:
    """Predicted cumulative fade. Returns NaN where the model is undefined.

    A cycle index of zero has no logarithm and a temperature at absolute zero
    has no reciprocal; both yield NaN rather than a clipped stand-in, so a
    caller cannot mistake an undefined prediction for a small one.
    """
    column = temperature_col or fit.temperature_column
    temperature = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
    cycle = pd.to_numeric(data[cycle_col], errors="coerce").to_numpy(dtype=float)

    kelvin = temperature + KELVIN_OFFSET
    valid = np.isfinite(kelvin) & (kelvin > 0) & np.isfinite(cycle) & (cycle > 0)

    out = np.full(len(data), np.nan, dtype=float)
    if not valid.any() or not np.isfinite(fit.ln_a):
        return out

    log_fade = (
        fit.ln_a
        - (fit.activation_energy_j_per_mol / GAS_CONSTANT) / kelvin[valid]
        + fit.exponent_z * np.log(cycle[valid])
    )
    out[valid] = np.exp(log_fade) * fit.smearing_factor
    return out


def build_arrhenius(
    features: Sequence[str] = (),
    target: str = "cumulative_fade",
    temperature_col: str = DEFAULT_TEMPERATURE_COLUMN,
    cycle_col: str = DEFAULT_CYCLE_COLUMN,
) -> FitFn:
    """A `FitFn` wrapping the Arrhenius model, for the benchmark harness.

    `features` is accepted and ignored: the functional form fixes which
    columns are used, and silently honouring an arbitrary feature list would
    misrepresent what was fitted. The temperature and cycle columns are
    parameters instead, which is the honest way to make it configurable.

    NaN predictions are replaced by the training fold's mean fade before
    returning, because the validator requires a finite prediction per test
    row. That substitution is the model declining to extrapolate, and it is
    scored as such — falling back to the training mean is exactly the
    behaviour of the `train_mean` baseline, so a model that cannot evaluate
    on most of a fold scores like the baseline rather than being excused.
    """
    def fit(train: pd.DataFrame):
        fitted = fit_arrhenius(
            train, target=target, temperature_col=temperature_col,
            cycle_col=cycle_col,
        )
        fallback = float(pd.to_numeric(train[target], errors="coerce").mean())

        def predict(test: pd.DataFrame) -> np.ndarray:
            values = predict_arrhenius(
                fitted, test, temperature_col=temperature_col, cycle_col=cycle_col,
            )
            return np.where(np.isfinite(values), values, fallback)

        return predict
    return fit


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Whether an activation energy can be estimated from this data at all.

    Modelled on `adaptive.datasets.assess_suitability`: the question is asked
    and answered *before* fitting, because a fit produced on data that cannot
    support it is not a weak result, it is a meaningless one — and a
    meaningless number in a results table is worse than an absent one.
    """

    identifiable: bool
    reasons: tuple[str, ...]
    n_ambient_levels: int
    within_cohort_temp_spread: float
    between_cohort_temp_spread: float
    early_soh_spread: float

    @property
    def status(self) -> str:
        return "IDENTIFIABLE" if self.identifiable else "NOT_IDENTIFIABLE"

    @property
    def within_between_ratio(self) -> float:
        return (
            self.within_cohort_temp_spread / self.between_cohort_temp_spread
            if self.between_cohort_temp_spread > 0 else float("nan")
        )

    def render(self) -> str:
        lines = [f"Arrhenius identifiability: {self.status}"]
        lines.append(f"  ambient levels: {self.n_ambient_levels}")
        lines.append(
            f"  temperature spread within/between cohorts: "
            f"{self.within_cohort_temp_spread:.2f} / "
            f"{self.between_cohort_temp_spread:.2f} "
            f"(ratio {self.within_between_ratio:.3f})"
        )
        lines.append(f"  early-cycle SOH spread across ambient levels: "
                     f"{self.early_soh_spread:.4f}")
        for reason in self.reasons:
            lines.append(f"  - {reason}")
        return "\n".join(lines)


# A thermal coefficient is confounded with protocol unless a reasonable share
# of the temperature variation occurs *inside* cohorts.
MIN_WITHIN_BETWEEN_RATIO = 0.25
# Early-cycle SOH should not vary much across ambient levels; large spread
# means capacity was not measured at a common temperature.
MAX_EARLY_SOH_SPREAD = 0.05
MIN_AMBIENT_LEVELS = 3


def assess_identifiability(
    data: pd.DataFrame,
    soh_col: str = "soh",
    ambient_col: str = "ambient_temperature_c",
    temperature_col: str = DEFAULT_TEMPERATURE_COLUMN,
    cohort_col: str = "cohort",
    cycle_col: str = DEFAULT_CYCLE_COLUMN,
    early_cycles: int = 20,
) -> IdentifiabilityReport:
    """Can an activation energy be estimated from this frame? Usually not.

    Three checks, each of which independently defeats identification. See the
    module docstring for the NASA result that motivated them.
    """
    reasons: list[str] = []

    levels = (
        int(data[ambient_col].nunique()) if ambient_col in data.columns else 0
    )
    if levels < MIN_AMBIENT_LEVELS:
        reasons.append(
            f"only {levels} distinct ambient temperature level(s); at least "
            f"{MIN_AMBIENT_LEVELS} are needed to fit a slope against 1/T and "
            f"see curvature."
        )

    within = between = float("nan")
    if cohort_col in data.columns and temperature_col in data.columns:
        grouped = data.groupby(cohort_col)[temperature_col]
        within = float(grouped.std(ddof=0).mean())
        between = float(grouped.mean().std(ddof=0))
        ratio = within / between if between > 0 else float("nan")
        if not np.isfinite(ratio) or ratio < MIN_WITHIN_BETWEEN_RATIO:
            reasons.append(
                f"temperature varies {ratio:.3f}x as much within cohorts as "
                f"between them, below the {MIN_WITHIN_BETWEEN_RATIO} minimum. "
                f"Ambient temperature is effectively a label for protocol, so a "
                f"thermal coefficient cannot be separated from a protocol effect."
            )

    spread = float("nan")
    if soh_col in data.columns and ambient_col in data.columns:
        early = data[(data[cycle_col] <= early_cycles) & data[soh_col].notna()]
        if not early.empty:
            means = early.groupby(ambient_col)[soh_col].mean()
            spread = float(means.max() - means.min()) if len(means) > 1 else 0.0
            if spread > MAX_EARLY_SOH_SPREAD:
                reasons.append(
                    f"apparent SOH differs by {spread:.3f} across ambient levels "
                    f"at cycle <= {early_cycles}, before real degradation is "
                    f"possible. Capacity was not measured at a common "
                    f"temperature, so the target carries a reversible thermal "
                    f"offset opposing the effect being fitted "
                    f"(see thermal_confound.py)."
                )

    return IdentifiabilityReport(
        identifiable=not reasons,
        reasons=tuple(reasons),
        n_ambient_levels=levels,
        within_cohort_temp_spread=within,
        between_cohort_temp_spread=between,
        early_soh_spread=spread,
    )


def arrhenius_stability(
    data: pd.DataFrame,
    target: str = "cumulative_fade",
    cohort_col: str = "cohort",
    temperature_col: str = DEFAULT_TEMPERATURE_COLUMN,
) -> pd.DataFrame:
    """Fit Ea separately per cohort, to test whether it is a constant.

    This is the module's central claim reduced to one table. If the activation
    energy is a property of the chemistry, per-cohort fits should agree within
    noise. If they scatter as widely as a linear temperature coefficient does,
    the Arrhenius form is not buying transferability and this module's
    hypothesis is wrong — which is a result worth having either way.
    """
    rows = []
    for cohort, group in data.groupby(cohort_col):
        fitted = fit_arrhenius(group, target=target, temperature_col=temperature_col)
        rows.append({
            "cohort": str(cohort),
            "n_rows": len(group),
            "n_used": fitted.n_used,
            "ea_kj_per_mol": fitted.activation_energy_kj_per_mol,
            "exponent_z": fitted.exponent_z,
            "r2_log_space": fitted.r2_log_space,
            "plausible": fitted.plausible,
            "admissible": fitted.admissible,
        })
    return pd.DataFrame(rows)
