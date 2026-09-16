# Architecture diagram — corrected specification

The BEACON architecture poster is the most widely seen artifact this project
produces. It is also the least validated: it was drawn early, from the plan,
and several boxes still describe what was intended rather than what was built
and measured.

This document is the **authoritative source for a redraw**. It states, box by
box, what the poster currently claims, what the evidence supports, and what the
label should say instead. A box not listed under "Corrections" is accurate and
should be redrawn unchanged.

The corrections are not cosmetic. Three of them are cases where the poster
claims a result the project's own validation gate rejected, which is precisely
the failure mode the project exists to criticise (ADR 0005). Publishing the
current poster alongside `docs/final_report.md` would put the two in direct
contradiction.

**Every figure below that comes from this project's artifacts is pinned by
`tests/test_reported_numbers.py`.** If one goes stale, that suite fails and
names this file and the section.

The one exception is the arithmetic in C3 — lab and pack fade rates, and the
coulomb-counting drift they are compared against. Those are properties of
cells and instrumentation, not outputs of this repository, so there is no
artifact to pin them to. They are stated with their inputs so the arithmetic
can be checked by hand.

---

## Corrections

### C1 — Box 3.4: "Harmful Behaviour Detection" is not what was built

| | |
|---|---|
| **Currently reads** | `3.4 Harmful Behavior Detection` — Unsupervised Clustering (K-Means / DBSCAN), Anomaly Detection (Isolation Forest / LOF). **Detect:** Overcharging, Deep Discharge, High-rate Fast Charging, High Temperature Usage |
| **Should read** | `3.4 Usage Segmentation & Outlier Detection` |
| **Authority** | ADR 0011 |

Two separate errors are folded into this one box.

**The label asserts the result.** Clustering and outlier detection find
statistical structure and statistical outliers. Neither algorithm is given a
fade target; neither knows what degradation is. Nothing about a cluster
boundary makes one side harmful. The project tested the assumption directly
and it did not hold:

| Detector | Flagged | Median fade (flagged) | Median fade (unflagged) | p | Verdict |
|---|---:|---:|---:|---:|---|
| Isolation Forest | 135 / 2682 | -0.005221 | +0.000101 | 1.0000 | NOT_ASSOCIATED |
| Local Outlier Factor | 135 / 2682 | -0.002545 | +0.000000 | 0.8999 | NOT_ASSOCIATED |

Flagged cycles are followed by *less* fade than unflagged ones, within cell.
The two detectors also agree on only 34 of their 135 flags.

**The four listed conditions belong to a different box.** Overcharging, deep
discharge, high-rate fast charging and high-temperature usage are detected by
name, by threshold rules, in `features.behavior_features` — that is box 3.1,
not 3.4. Listing them under the unsupervised box implies the clustering found
them. It did not; it *rediscovered* them, which is corroboration of the rules
and an argument for keeping them, not a new detector.

**Redraw as:**

- Title: `3.4 Usage Segmentation & Outlier Detection`
- Left: `Unsupervised Clustering (K-Means / DBSCAN)` — keep
- Right: `Statistical Outlier Detection (Isolation Forest / LOF)` — keep
- Replace the `Detect:` list with a `Validated:` list:
  - `6 usage segments, silhouette 0.541 vs null 0.153`
  - `Segments reproduce the rule-based flags`
  - `Outliers show NO link to faster fade (p = 1.00 / 0.90)`
- Move the four named conditions into box 3.1 as `Rule-based flags`.

This box should read as a **completed negative result**, not a gap. It is the
most defensible output the box could have produced, and it should be presented
as a finding rather than apologised for or quietly dropped.

---

### C2 — Box 7: `SOH Prediction Error < 3%` is met only under the optimistic split

| | |
|---|---|
| **Currently reads** | `SOH Prediction Error < 3%` |
| **Should read** | `SOH MAE <= 5% leave-one-cell-out; <= 10% leave-one-cohort-out, reported as an interval over cohort draws` |
| **Authority** | ADR 0008, ADR 0010; `docs/roadmap.md` |

Best figures anywhere in the project, on CALCE CS2+CX2 (19 cells, 8 cohorts,
ceiling 0.870):

| Method | LOBO | LOCO |
|---|---:|---:|
| `lstm` | **2.26%** | 15.68% |
| `gpr_matern` | **2.93%** | 10.06% |
| `random_forest` | 3.04% | **8.59%** |

Under leave-one-cell-out, two methods clear 3%. Under leave-one-cohort-out —
the split that actually estimates what happens when the operating protocol
changes — the best is 8.59%, roughly 3x the target.

The target is reachable **precisely where the measurement is optimistic**. A
poster claiming `< 3%` next to a report demonstrating that LOBO overstates
deployable skill is the project contradicting itself in public.

The restatement also carries the interval requirement, because a single LOCO
number cannot be trusted to a decimal place (see C5).

---

### C3 — Box 7: `RUL within +/- 20 cycles` is a lab-cell claim stated as a product claim

| | |
|---|---|
| **Currently reads** | `RUL Estimation Accuracy Within +/- 20 Cycles` |
| **Should read** | `RUL within +/- 20 cycles on lab-accelerated cells; on production packs, expressed as a percentage of remaining life` |
| **Authority** | `docs/roadmap.md`, specification defects |

NASA cells fade at 0.251% SOH/cycle, so +/-18 cycles follows from a 4.5% SOH
error — achievable. A production EV pack fades roughly 19x slower (about 20%
over 1,500 cycles, ~0.013%/cycle), so +/-20 cycles there requires **0.27% SOH
accuracy** — below coulomb-counting drift, which is typically 1-2%.

"+/-20 cycles on NASA" and "+/-20 cycles on a car" are not the same claim, and
the poster sits next to a car.

---

### C4 — Box 3.1: `Usage Patterns (Daily/Weekly Trends)` has no data to run on

| | |
|---|---|
| **Currently reads** | `Usage Patterns (Daily / Weekly Trends)`, drawn as delivered |
| **Should read** | Same label, marked `not applicable to lab-cycled data` |
| **Authority** | `docs/roadmap.md`; `docs/hardware_integration.md` |

Every dataset held here is cycle-indexed. There is no wall-clock axis, so daily
and weekly aggregation has nothing to aggregate over. NASA's timestamps cover
test-rig time; CALCE's Arbin exports carry test time only. This is not a
missing feature — there is no modelling change that fixes it.

The instrumented rig is the route to unblocking it, since a microcontroller
streams at wall-clock intervals. Two things are still required: a physical rig
logging, and a capture long enough for weekly aggregation to mean anything —
days of continuous logging, not a bench session.

**Redraw as:** keep the item, greyed or asterisked, with the footnote
`requires wall-clock telemetry (instrumented rig / fleet feed)`.

---

### C5 — Box 7: add the missing third target

The poster's two performance targets are both accuracy claims. The project's
actual headline result is about **measurement**, and it has no box:

> The LOCO estimate's interquartile range across cohort draws is **0.724 R2**,
> against a median between-method difference of **0.351 R2** — the measurement
> noise is roughly twice the effect it is used to measure.

**Add:** `Validation: LOCO reported as an interval over cohort draws, never a
point estimate` — with the target's noise ceiling reported alongside every R2.

This is the contribution. Leaving it off the poster while keeping two accuracy
targets the project has argued are the wrong frame gets the emphasis exactly
backwards.

---

### C6 — Box 4: the Battery Guardian speech bubble is a mockup

| | |
|---|---|
| **Currently reads** | *"Your battery health is 87%. Frequent fast charging and high temperature usage are increasing degradation risk..."* |
| **Authority** | ADR 0004 |

ADR 0004 exists because the approved visual design was mocked up with
placeholder readings, and the decision recorded there is that the dashboard
renders only computed values. The poster still carries the placeholder.

**Redraw as:** regenerate the bubble from an actual `main.py` run and use that
text verbatim, or label the bubble `illustrative`. Do not leave an invented
reading rendered in the house style of a real one.

---

### C7 — "Value Delivered" column claims outcomes never measured

| Claim | Status |
|---|---|
| Accurate Battery Health Monitoring & Prediction | **Not supported** — nothing has passed the promotion gate |
| Early Risk Detection & Alerts | **Not supported** — the risk score is rule-based triage, labelled as such throughout |
| Personalized Battery Life Optimization | **Not supported** — recommendations are rule-derived, never trialled |
| Extended Battery Lifespan | **Not measurable here** — would require a controlled intervention study |
| Improved Safety, Reliability & Performance | **Not measurable here** |
| Data-driven Decision Making | Supportable |

Five of six are outcome claims that no artifact in this repository evidences.
They are the kind of statement the validation gate was built to stop, and they
are currently the most prominent text on the poster.

**Redraw as** capability statements the project can defend, e.g.:

- `Validation methodology that detects optimistic cross-validation`
- `Refusal gate: no unvalidated model reaches the API or dashboard`
- `Target noise ceilings reported alongside every score`
- `Exact Shapley attribution over the rule scores`
- `Two-transport telemetry ingestion (CAN + serial) behind one coverage gate`
- `Reproducible negative results, tracked as artifacts`

---

## Boxes that are accurate — redraw unchanged

| Box | Note |
|---|---|
| 1. Data Sources | NASA, CALCE and Oxford loaders all exist. Stanford is screened MARGINAL and deliberately not loaded — if it is drawn, mark it as screened-out rather than ingested. |
| 2. Ingestion & Preprocessing | Accurate end to end. |
| 3.1 Behavior Analysis | Accurate except the daily/weekly item (C4). Gains the four rule-based flags from C1. |
| 3.2 ML Models | Accurate. LSTM, XGBoost and GPR are real implementations, not stubs. |
| 3.3 Health & Risk Assessment | Computation is accurate. Nothing is promoted, which is a property of the gate, not of these boxes. |
| 5. Integration Layer | Accurate. FastAPI, JSON, CAN and serial paths all exist. |
| 6. System Foundation | Accurate for modular architecture, digital twin, containers and CI. **Persistence, streaming and observability are not built** and are deliberately gated — if the poster implies they ship, soften to `designed, gated on model promotion`. |

---

## What must not be added to make the diagram look fuller

- **A model wired into the dashboard.** The gate is the product (ADR 0005). A
  dashboard showing a number the gate rejected is the failure this project was
  built to prevent.
- **A fitted severity score replacing the rule score.** The rule score is
  labelled as triage and that labelling is accurate. Swapping in an unvalidated
  fitted model changes what is displayed without changing what is known.
- **Any restatement of 3.4 that implies harm was detected.** The negative
  result is the finding.

---

## See also

- ADR 0004 — the dashboard renders only computed values
- ADR 0005 — the adaptive calibration gate defaults to REJECT
- ADR 0008 — model capacity vs transfer
- ADR 0010 — LOCO variance
- ADR 0011 — unsupervised detection finds structure, not harm
- `docs/roadmap.md` — specification defects, stated in dependency order
- `tests/test_reported_numbers.py` — every figure above, pinned to its artifact
