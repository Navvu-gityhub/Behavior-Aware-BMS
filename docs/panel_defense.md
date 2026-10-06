# BEACON — panel defence: honest answers

Every answer below is checked against what the repository actually contains
(October 2026; updated after the gap-filling work that followed this review). Where something was not done, the answer
says so. Numbers trace to files under `reports/metrics/`.

For a one-page view of every claim and its evidence, see `docs/validation_matrix.md`.

Answers are written to be said out loud. Short where the honest answer is
short.

---

## Fix these on your slides BEFORE the panel

Your slides currently make claims the work does not support. A panel will find
them. Change them first.

| Slide says | The truth | What to say instead |
|---|---|---|
| Engineered features → XGBoost / LSTM / GPR → SOH/RUL | The 12 benchmark models were evaluated for **SOH only** (target `soh`, NASA). None estimates RUL. **None of them is used by the running system.** | "We benchmarked 12 published methods for SOH. None transferred across test protocols, so the shipped system uses two estimators that fit nothing across cells." |
| XGBoost is the best model | Its LOCO 95% interval is [−2.80, 0.86]; it overlaps every other method's. No ranking is supportable. | "XGBoost has the highest point estimate; the intervals overlap, so we do not claim it is best." |
| Table shows LOBO MAE but not LOBO R² | Both were computed (`docs/final_report.md`, benchmark table). | Show the full table: LOBO MAE, LOCO MAE, LOBO R², LOCO R². |
| RUL ±20 cycles, 88% | 88% used a reference capacity taken from the cell's whole life (the future). With the reference a BMS can hold, it is **73%**, and that is conditioned on the *true* remaining life. By the *prediction*: when it says under 25 cycles, the cell lasted at least that long 95% of the time. | "Near end of life it is a safe lower bound: when it predicts under 25 cycles, the cell lasted at least that long 95% of the time." |
| SOH 3.3% median error | Superseded. The field method (voltage, current, time only) gives **1.7% median on 17 of 22 cells**. | Use 1.7% / 17 of 22. |
| Digital twin | It is a state tracker: four states relabelled from the health state, transition detection, and per-battery history. No physics model, no forward simulation, no what-if. | Call it a "battery state tracker" or "digital shadow", or be ready to defend exactly what it is (section P). |
| Hardware validates the system | It validates the **telemetry path only**. The cell was never discharged; no capacity or ageing was measured. | "The rig proves acquisition, the wire protocol and the refusal logic on real silicon. It does not validate health estimation." |
| Behaviour predicts degradation (anywhere, spoken or written) | Ablation: XGBoost on usage features alone scores LOCO R² −0.29; on age alone −0.27; only the combination reaches 0.459, inside overlapping intervals. A straight line on age alone reaches 0.406, so usage adds about 0.05, within the noise. Only temperature showed a consistent link, and only in direction. | "Usage features added no skill we could separate from age. Temperature was the one usage factor linked to faster fade, and we could not measure how much." |
| ~21k lines | Repository is about 48k lines including tests, scripts, frontend and firmware. | Quote whichever you can show; don't guess. |

**Your three strongest answers**, in one line each:
1. *Research:* under leave-one-cohort-out, every one of 12 methods lost skill, and the uncertainty on one method's score was wider than the gap between all of them. So model choice is unresolvable at this data size, while fixing how the data was conditioned cut worst-case error 5×.
2. *Engineering:* the system refuses, with a reason, any number its inputs cannot support. That design came from a real bug: a missing temperature sensor was being scored as "not hot".
3. *Result:* SOH from voltage, current and time only, at 1.7% median error on real lab cells, and 3.2–3.4% from partial discharges alone. It is plotted against every lab capacity test in the README.

---

## A. Project fundamentals

**1. What is your project, in one sentence?**
A software layer that takes battery telemetry from a BMS, a bench rig or a research dataset, estimates state of health and remaining life from it, and refuses any figure the data cannot support.

**2. What does it solve that a conventional BMS does not?**
A conventional BMS protects the cell and estimates state of charge. BEACON adds state-of-health and remaining-life estimates checked against lab data, explains them in plain language, and states when it cannot compute something. Honestly: production BMS firmware often does estimate SOH too; our contribution is the validation and the refusal logic, not the idea of estimating SOH.

**3. BMS, monitoring system, health estimator, or a layer on top?**
A software layer on top of a BMS, doing health estimation and monitoring. It is not a BMS: it does no protection, balancing or control.

**4. Research contribution?**
A measured finding: under leave-one-cohort-out, every method we tested lost skill, and model ranking was not statistically resolvable, while data-conditioning decisions changed error by 5×. Plus a field-realisable SOH method validated on real cells, including real partial cycling, with physically motivated gates that refuse bad readings.

**5. Engineering contribution?**
One shared pipeline for three data sources (CAN, serial rig, datasets); refusals carried as values all the way to the user; a checksummed serial protocol with firmware for three microcontroller families; 964 tests and an 11-job CI.

**6. How is this different from a dashboard of voltage, current and temperature?**
Those three are raw inputs. BEACON segments the discharges, integrates current into charge, estimates internal resistance, computes SOH and RUL, scores them against lab truth, and refuses when a reading is untrustworthy. A dashboard of raw values does none of that.

**7. Inputs?**
Voltage, current and time are required. Temperature and SOC are optional. Rated capacity is optional but unlocks C-rate features and the learned-window SOH. Sources: CAN frames plus a DBC file, a serial rig, or a dataset file.

**8. Outputs?**
Measured SOH, RUL to 90% capacity, a heuristic health index and risk score (labelled heuristic), heat-exposure advice, a battery state (HEALTHY / WARNING / DEGRADED / CRITICAL), refusals with reasons, and the report card.

**9. Measured vs estimated?**
Measured: voltage, current, temperature, time. Derived by integration: charge per discharge. Estimated: SOH (a ratio of two measured window charges, so close to a measurement), RUL (an extrapolation), the health index and risk score (heuristics). The report card labels each line MEASURED or ESTIMATE.

**10. Model-derived vs rule/calculation-derived?**
Everything the running system shows is calculation- or rule-derived. SOH is a ratio of charges; RUL is a straight-line extrapolation of that cell's own fade; the health index and risk score are hand-set rules. The ML models exist only in the benchmark study.

**11. Claims you can prove today?**
- SOH 1.7% median error, 17 of 22 CALCE cells, from voltage, current and time only.
- Unchanged, 3.0% on all 8 cells of a second dataset (Oxford: different chemistry, maker, format, 40 °C); 2.3% with a learned window.
- SOH 3.2–3.4% from real partial cycling (2 cells).
- Resistance from the load step tracks the lab instrument on 19 of 20 cells.
- RUL 73% within ±20 cycles in the last 25 cycles before 90%; near end of life, predictions were rarely exceeded (95% CALCE, 97% Oxford at 80%).
- Readings' consistency predicts SOH error (1.3% vs 5.1%); discharge count does not.
- No benchmark method kept its skill under LOCO.
- The rig decodes 83/83 frames at 1.000 s; injected faults are rejected or refused.

All reproducible from the repo; see `docs/validation_matrix.md`.

**12. Limitations?**
- Single cells only; no packs.
- Two cathode families (LCO, NMC/LCO blend); no LFP.
- One real drive-cycle discharge only; the rest is constant current.
- RUL is only useful near end of life; at 80% it is a conservative bound with low precision.
- The rig has never measured a discharge.
- Partial-discharge SOH rests on 2 cells per band.
- RUL for partial-only logs is refused (four methods tested, none accurate enough).
- API persistence is opt-in (`BEACON_STATE_FILE`); the default is in memory.

**13. Why lithium-ion?**
It is the chemistry used in EVs and the one the public ageing datasets (NASA, CALCE) cover.

**14. Why is health prediction difficult?**
Ageing is slow, path-dependent and driven by several mechanisms at once. Capacity can't be measured directly in the field: it needs a full controlled discharge, which a car rarely does. The signals that matter are small compared with measurement noise and temperature effects.

**15. Why can't voltage alone tell you health?**
Terminal voltage depends on state of charge, current (I × R sag), temperature and recent history, as well as on age. A new and an old cell at rest at the same SOC read almost the same voltage. Health shows up in how much charge flows between two voltages, which needs current and time too.

---

## B. NASA and CALCE

**16. What are the datasets?**
NASA PCoE battery ageing: 18650 cells cycled under 9 different protocols (ambient roughly 4–43 °C, different currents and cutoffs). We used 34 cells, about 7.2 million telemetry rows. CALCE (University of Maryland): CS2 cells (1.1 Ah; Type 1 0.9 Ah) and CX2 cells (1.35 Ah), LCO prismatic, cycled at room temperature under 6 test types per family.

**17. Same chemistry?**
Both are recorded as LCO in our dataset specs: NASA as 18650, CALCE as prismatic. Same chemistry family; different cells.

**18. Same manufacturer?**
No, as far as the published documentation tells us. We do not rely on it.

**19. Same capacity?**
No. NASA cells are about 2 Ah; CALCE CS2 1.1 Ah (0.9 Ah for Type 1); CX2 1.35 Ah.

**20. Same temperature?**
No. NASA varies ambient from about 4 to 43 °C; CALCE is room temperature.

**21. Same C-rate?**
No. Both vary it within their own protocols.

**22. Same charging protocol?**
No.

**23. Same discharge protocol?**
No. CALCE Type 3 even switches rate six times within one cycle.

**24. Same sampling frequency?**
No. CALCE partial-cycling cells log every 30 s; the rig logs every 1 s.

**25. Are the measurements directly comparable?**
Not without care. That's why every SOH is a ratio against the same cell's own early capacity, so it has no units and no cell-size dependence. Temperature is absent from CALCE cycling files entirely, and we leave it absent rather than fill in 23 °C.

**26. If the cells differ, why should a model trained on one work on another?**
It shouldn't be assumed to, and our results show it doesn't: models lose skill even across NASA's own protocols. That is why the shipped estimators fit nothing across cells.

**27. What common physical relationship are you learning?**
In the shipped system, none is learned. The SOH method relies on one physical fact: on the voltage plateau, the charge between two voltages scales with capacity. The benchmark models learned correlations, and those did not transfer.

**28. Physics or dataset correlations?**
The benchmark ML models learned dataset correlations; LOCO showed it. The shipped estimators use a physical ratio, not learned coefficients.

**29. How do you deal with domain shift?**
By not needing to transfer coefficients. Each cell is compared with its own beginning of life. We also measure the shift directly with LOCO, and gate inputs physically: plateau-only windows, references formed early, plausibility bounds.

**30. Did you normalize?**
Yes, two kinds. SOH is capacity divided by that cell's own reference capacity. For the ML benchmark, features are standardised with a scaler fitted on the training fold only.

**31. What and why?**
Capacity: so cells of 0.9, 1.1, 1.35 and 2 Ah are on the same 0–1 scale. Features: so models sensitive to scale (SVR, GPR, MLP, LSTM) train properly. The scaler never sees test data.

**32. Can normalization remove physical differences?**
No. It removes unit and scale differences. Different chemistry, temperature and protocol stay. That's exactly what LOCO exposed.

**33. Did you train on NASA and test on CALCE?**
No. When that study was planned, the CALCE files we had held no ageing trajectory, so it couldn't be run (report §4.8). Leave-one-cohort-out within NASA was used instead. We have not run a NASA→CALCE transfer since. Say so plainly.

**34. Did you train on CALCE and test on NASA?**
No. The NASA raw files aren't in the current checkout, and the shipped estimators need no training anyway.

**35. What happens when you leave a whole cohort out?**
Every method loses skill. XGBoost goes from LOBO R² 0.732 to LOCO 0.459. The LSTM goes from 0.594 to −0.185. The simple age-linear baseline loses least (0.483 → 0.406).

**36. Why is leave-one-battery-out not enough?**
The held-out battery's siblings from the same test protocol stay in training. The model only has to recognise a protocol it has already seen. A real new battery comes with conditions the model has never seen.

**37. Why LOCO?**
To test generalisation to a new test protocol, which is closer to the deployment question.

**38. What's a cohort?**
One experimental protocol: a combination of ambient temperature, discharge current or profile, and cutoff voltage. NASA has 9; CALCE has 6 types per cell family.

**39. What does LOCO tell you that a random split doesn't?**
Whether the model works on conditions it hasn't seen. A random split mixes cycles of the same battery into train and test, so it mostly measures memorisation.

**40. Why is a random split dangerous here?**
Neighbouring cycles of one battery are nearly identical. With some in training and some in test, the model can "predict" by recalling the neighbour. Scores look excellent and mean nothing.

**41. Can samples from one battery be in both train and test?**
Not in our protocol. Splits are by cell (LOBO) or by cohort (LOCO).

**42. If they were, would that be leakage?**
Yes.

**43. Temporal leakage?**
Features use trailing windows, past cycles only. The LSTM window for cycle k contains cycles up to k. RUL estimates use only history up to the estimate's cycle, and a test changes the future and checks that the estimate doesn't move.

**44. Battery-level leakage?**
Splitting by cell or cohort. LSTM early stopping uses held-out training cells, not windows.

**45. Cycle-level leakage?**
No cycle of a test cell is ever in training. Scalers are fitted on training rows only.

**46. Why better under LOBO than LOCO?**
Under LOBO the model has seen the test cell's protocol. Under LOCO it hasn't. The gap is the protocol effect the model had memorised.

**47. Why does XGBoost drop from 0.73 to 0.46?**
Because part of what it learned was protocol-specific: for example, how fade relates to temperature at that protocol's ambient. On an unseen protocol that part is wrong.

**48. What does the drop tell you?**
That about a third of the apparent skill was protocol memorisation, and that the LOBO numbers in most papers overstate real-world performance.

**49. Does 0.459 LOCO R² mean the model is good?**
No. Its 95% interval runs from −2.80 to 0.86, so on some held-out protocols it is worse than predicting the average. And a baseline that only knows the cycle count reaches 0.406.

**50. Does it generalise to a new real-world battery?**
Tested on three datasets. Unchanged, the SOH method gives 1.7% on CALCE and 3.0% on Oxford (40 °C), but 9.2% on NASA, where it fails in the cold (4 °C, 22 °C cohorts) and works at 43 °C (1.7%). Its own confidence label, fixed on CALCE, flags those failures: 4.5% error when it says HIGH/MEDIUM vs 13.2% when it says LOW. Fitted ML models look excellent inside one dataset (0.7%) but degrade to 5.5–6.7% on another, so we don't ship them.

**51. What would you need before claiming that?**
What we did for the SOH method: an independent dataset with a different chemistry, maker, format and temperature, with nothing re-tuned (3.0%, 8 of 8 cells). Still needed: LFP, packs, field data from real vehicles, and more than one drive-cycle discharge.

---

## C. Of what use is this model?

**52. What decision does the output enable?**
For the shipped estimators: when to plan replacement (SOH crossing 90% or 80%), and whether a battery is ageing faster than expected. For the ML benchmark models: none. They are a research result, not a product component.

**53. Action taken from an SOH prediction?**
The report card maps measured SOH to a state and an action. Under 80%: replace. 80–90%: plan replacement. 90–95%: monitor. It's advice to a person; nothing is actuated.

**54. How does RUL help?**
Near end of life, it says roughly how many cycles remain, to schedule a replacement. When it predicts under 25 cycles, the cell lasted at least that long 95% of the time. It needs complete discharges. For a driver who almost never fully discharges, four methods were tested (`reports/metrics/calce_field_rul/`): extrapolating the field SOH trajectory, a self-calibrated version, and sparse capacity checks with a lowered minimum. None was accurate enough to ship. So for partial-only logs the card refuses RUL and says why. That's an open problem, not a solved one.

**55. How is uncertainty communicated?**
As validation-derived error bands printed with every figure on the report card (`health/error_bands.py`, pinned to `reports/metrics/error_bands.csv`). SOH: 90% of readings were within ±3.4 points (±4.9 with a learned window). RUL: the band is indexed by the *predicted* value, the only one a user sees. Example: when the card predicts under 25 cycles, the true life fell 1–82 cycles later in 90% of cases. These are empirical bands from lab cells, not model confidence intervals.

**56. What if it's wrong?**
It's advisory, so a wrong number leads to a wrong recommendation, not an unsafe action. The gates are designed to refuse rather than give a confidently wrong number. For example, on near-empty partials it refuses instead of reporting 7–13% error.

**57. Overestimating or underestimating health — which is more dangerous?**
Overestimating. You'd keep a degraded battery in service. Our RUL errs early, which is the safer direction.

**58. Would you let it control charging?**
No.

**59. Why not?**
It isn't validated for packs, other chemistries or dynamic loads. It runs on a host computer, not safety-rated firmware. And control needs certified safety logic that this project doesn't have.

**60. Diagnostic or closed-loop controller?**
Diagnostic and decision support only.

**61. Why does that matter?**
A wrong diagnostic number costs a bad decision. A wrong control action can cause overheating or fire. The validation bar is completely different.

**62. If BEACON disappeared tomorrow, what remains?**
Everything safety-related: protection, cutoffs, balancing and SOC all live in the BMS (or, on our rig, the protection module). Only the health estimates, explanations and refusals would be lost.

---

## D. Hardware validation

**63. What are you trying to prove with the hardware?**
That the telemetry path works on real silicon: sensors, firmware, the wire protocol, checksums, parsing, the coverage gate and the refusal path.

**64. Validating the ML model?**
No.

**65. Validating telemetry acquisition?**
Yes.

**66. Validating the BMS architecture?**
Partly: the ingestion and refusal design. Not protection or control.

**67. Validating the sensors?**
Only that they report plausible, stable values. No calibration against a reference instrument is recorded.

**68. What does one 18650 cell prove?**
That a real board can stream data the pipeline accepts: 83/83 frames, 1.000 s cadence, no gaps. And that the system correctly refuses a channel the board can't supply.

**69. What does it not prove?**
Anything about SOH, RUL, ageing, accuracy under load, or other cells.

**70. Can one cell validate a model trained on NASA/CALCE?**
No.

**71. Then why is the hardware useful?**
It proves the system runs end to end on real hardware, not only on files. It exercised the refusal logic against a real missing sensor. And it's the platform for the next experiment: a measured discharge.

**72. How do the physical and dataset experiments relate?**
They share one pipeline. The datasets validate the estimators; the rig validates acquisition. Neither validates the other's part.

**73. Model accuracy or telemetry pipeline?**
Telemetry pipeline.

**74. Ground truth for the physical cell?**
None for health. The voltage reading is consistent with the INA219's 4 mV steps, but it hasn't been checked against a calibrated meter on record.

**75. How do you know its capacity or SOH?**
We don't.

**76. Do you know its initial capacity?**
Only the nominal label value. The firmware declares 2.2 Ah. It wasn't measured.

**77. Have you done a complete charge-discharge cycle?**
No.

**78. How many cycles measured?**
Zero. Each capture is a cell at rest, about 82–84 seconds.

**79. Observed degradation?**
No.

**80. Then how can you claim the hardware validates degradation prediction?**
We don't, and the slides shouldn't. Degradation is validated only on CALCE and NASA data.

**81. Can you show ageing from a few minutes of data?**
No.

**82. Degradation timescale?**
Hundreds to thousands of cycles. At about 2–4 hours per cycle, that's months of continuous cycling to see a few percent of fade.

**83. Your hardware timescale?**
About 90 seconds per capture.

**84. Compatible?**
No, by several orders of magnitude.

---

## E. Hardware measurement

**85. What do you measure?**
Voltage and current (INA219), temperature (LM35), time (microcontroller clock). SOC is computed in firmware by coulomb counting from an assumed starting value; it isn't measured.

**86. Voltage?**
INA219 bus-voltage channel, over I²C. Datasheet resolution 4 mV per step.

**87. Current?**
INA219 shunt channel: the voltage across a small shunt resistor, 0.1 Ω on the common breakout board. Datasheet resolution 10 µV, which is 0.1 mA at 0.1 Ω.

**88. Temperature?**
LM35 analogue sensor into the ESP8266's A0 pin, through the NodeMCU's on-board 220k/100k divider. 32 samples averaged; about 0.31 °C per ADC count.

**89. Why INA219?**
Voltage and current on one I²C chip, no hand-built shunt amplifier or divider to calibrate, and a cheap common breakout.

**90. Why LM35?**
Simple and linear (10 mV/°C), needs no calibration constants, and was available. A DS18B20 would have been the more robust digital choice.

**91. INA219 accuracy?**
From the datasheet, roughly ±0.5–1% of reading. Our design doc marks these figures "[verify]". We haven't checked them against a reference. Measured noise: voltage SD 1.9 mV; current at the ±0.4 mA noise floor (no load).

**92. LM35 accuracy and resolution?**
Datasheet about ±0.5 °C near 25 °C. Our ADC resolution is about 0.31 °C per count; observed SD 0.086 °C over 85 s.

**93. Where is the temperature sensor?**
On or near the cell surface on the bench. Its exact mounting isn't documented in the repo; describe it as built.

**94. Does surface temperature represent core temperature?**
No. Under load the core runs hotter than the surface. At rest they're close.

**95. Temperature measurement errors?**
- Surface vs core temperature.
- Poor thermal contact.
- ESP8266 ADC non-linearity and noise.
- Divider tolerance.
- A disconnected pin reading near zero.

The firmware treats readings below 5 °C as "disconnected", not "cold".

**96. How is the INA219 wired?**
High side: the shunt sits in the positive leg between the cell and the load; I²C (SDA/SCL) goes to the ESP8266. This follows `docs/hardware_design.md`.

**97. Role of the shunt?**
Current through it makes a small voltage (I × R) that the INA219 measures and converts to current.

**98. Does the sensor add a voltage drop?**
Yes: I × 0.1 Ω in the current path, 0.1 V at 1 A. It's measured, not hidden.

**99. How do you account for sensor error?**
Mostly by design rather than correction. SOH is a ratio of two readings from the same sensor, so constant gain errors cancel. Out-of-range values are rejected, not clamped. No explicit error budget has been measured.

**100. How do you calibrate the sensors?**
We don't, beyond datasheet scaling. No calibration record exists. That's a gap.

**101. What if a sensor disconnects?**
The firmware detects it (the LM35 below 5 °C; the INA219 not responding on I²C). It drops that channel from its declaration and reports why. The host then refuses the features that need it.

**102. What if a sensor returns NaN?**
The firmware omits the field. The host rejects a record missing a declared channel and counts it. NaN is never treated as a value.

**103. What if current suddenly becomes very high?**
Above about ±3.2 A, a 0.1 Ω shunt saturates the INA219 range and the reading is wrong. The schema's range check catches absurd values but not saturation. Protection is the protection board's job, not BEACON's.

**104. What protection sits between the 18650 and the ESP8266?**
The cell isn't connected to ESP8266 pins at all; only the INA219 (over I²C) and the LM35 are. Cell protection is specified as a TP4056 + DW01 module. Confirm on your actual bench build whether it's fitted. I can't see the bench.

**105. Can the ESP8266 tolerate battery voltage directly?**
Not on its ADC pin without a divider. The design avoids it: the INA219 measures the cell (up to 26 V bus).

**106. What if the battery is connected backwards?**
The INA219 and the protection module can be damaged. The software can't prevent it. A reversed *current sign* (shunt leads swapped) is detected by the host, and it suggests the fix.

**107. Is the hardware a BMS protection circuit?**
No. This prototype is a monitoring and telemetry system, not a protection-and-balancing BMS.

**108. Over-voltage protection?**
Not from BEACON. Only from the protection module, if fitted.

**109. Over-current?**
Same.

**110. Over-temperature?**
No.

**111. Balancing?**
No. It's a single cell.

---

## F. How long do you measure?

**112. Sampling interval?**
1.000 s (`SAMPLE_PERIOD_MS = 1000`). Measured cadence exactly 1.000 s with no gaps.

**113. Why 1 s?**
Fast enough to see a load step and to integrate current accurately for slow discharges. It uses about 7% of the serial channel at the default baud.

**114. Capture length?**
About 82–84 seconds per capture.

**115. Samples per experiment?**
83 and 85 records in the two main captures.

**116. A complete charge cycle?**
No.

**117. A complete discharge?**
No.

**118. Voltage, current and temperature together?**
No. In the committed captures, the INA219 and the LM35 were never working at the same time. Each capture declares only the channels its sensors could supply.

**119. Repeated cycles?**
No.

**120. Under load?**
No load on the rig. But the method has now been tested on one real dynamic discharge: an Artemis urban drive cycle on an Oxford cell (current −5 to +1.6 A). Read in a window that discharge covers, it came within 1.4% of the same cell's steady 1C discharge.

**121. What load?**
None was applied.

**122. C-rate?**
Zero; the cell was at rest.

**123. What happens to voltage when current changes?**
It drops by I × R immediately, then relaxes further over seconds to minutes as polarisation builds. We've measured this on CALCE data, where the step-based resistance is about 0.165 Ω on CS2_35. We haven't measured it on the rig.

**124. Can you show internal resistance from your hardware?**
Not yet. It needs a load step. The software already extracts it from load steps in CALCE data.

**125. Temperature rise under load?**
No.

**126. Capacity fade experimentally?**
Not on hardware. Only in the lab datasets.

**127. Then what exactly is the hardware evidence?**
That a real ESP8266 with real sensors streams validated, checksummed telemetry the pipeline accepts and scores; that timing holds; and that a real missing sensor is refused by name.

---

## G. Hardware → dashboard pipeline

**128. What happens when the battery is connected?**
The firmware boots, checks each sensor, prints a status line per sensor, then a HELLO line declaring its schema, the channels it can supply, its cell ID and its capacity. Then it sends one data line per second.

**129. How does the ESP8266 acquire measurements?**
INA219 over I²C (bus voltage and shunt current). LM35 via `analogRead(A0)`, averaged over 32 samples.

**130. How often?**
Once per second.

**131. How are readings formatted?**
`BEACON1 D t=… v=… i=… tc=… soc=…*CS`: a fixed prefix, key=value pairs, then an XOR checksum in hex.

**132. Why a checksum?**
Serial lines get corrupted (noise, dropped bytes, a reset mid-line). A corrupted number that still parses would be scored as if it were real.

**133. What does it protect against?**
Corrupted or truncated lines being accepted. It's an 8-bit XOR, which catches most single-byte errors. It doesn't catch everything and isn't security.

**134. What happens on checksum failure?**
The line is rejected and counted in the decode statistics. If too few lines are accepted overall, the run is refused.

**135. How are timestamps generated?**
The firmware's elapsed time `t` in seconds since start. The host can add a wall-clock timestamp if the capture start time is given.

**136. How do you detect dropped packets?**
From the cadence: gaps in `t` larger than the declared period. Plus counts of lines read vs accepted.

**137. How do you detect time reversal?**
The host checks that `t` never goes backwards and refuses a capture where it does, reporting the first reversed sample.

**138. Where does serial data go after the ESP8266?**
USB serial to the host PC, read by `SerialPortSource` (pyserial), or saved to a text file and replayed.

**139. What process receives it?**
The Python serial pipeline, either directly or through the FastAPI route `/telemetry/serial/live` (or `/replay`).

**140. How is it converted to the common schema?**
`serial_schema.py` maps wire names (`v`, `i`, `tc`) to unified channels (`voltage_v`, `current_a`, `temperature_c`), checks each against a range, and builds a table.

**141. After ingestion?**
Segment into charge, discharge and rest phases. Integrate current per discharge. Measure SOH from the voltage window. Extrapolate RUL. Then, if the channels allow, behaviour features, risk, the Guardian report and the twin state.

**142. How does the backend know a measurement is valid?**
Checksum, per-field range checks, channel coverage against what the board declared, capture-level checks (SOC scale, current sign, time order), then physical gates in the estimators.

**143. How does data reach FastAPI?**
FastAPI calls the Python pipeline in-process. The pipeline is a library, not a separate service.

**144. Why Express if FastAPI exists?**
It's a gateway for the React client: one origin, CORS, and consistent error mapping (502 when the Python service is down). Honestly, it isn't technically necessary. It was part of the MERN-stack deliverable.

**145. Why React?**
A component-based dashboard with interactive charts. It was also part of the planned stack.

**146. What does the dashboard receive?**
JSON per battery: state, health index, risk, RUL, SOH series, measured SOH where available, `state_basis`, attribution, the Guardian text, heat advice, general guidance, and provenance (simulated or measured).

**147. Does the frontend calculate SOH or RUL?**
No.

**148. The backend?**
Yes, the Python pipeline.

**149. Where does ML inference happen?**
Nowhere in the running system. ML runs only in the benchmark scripts.

**150. Where are raw measurements stored?**
Rig captures are text files (`data/interim/rig_*.txt`). Dataset files are on disk under `data/raw/`. The API keeps nothing on disk.

**151. Processed measurements?**
`main.py` writes CSVs to `data/features/` and `reports/`. Studies write CSVs to `reports/metrics/`. The API holds results in memory.

**152. Model outputs?**
Same: CSV files from scripts, in memory in the API.

**153. Do results persist after a server restart?**
Yes, if enabled. Set `BEACON_STATE_FILE` and the fleet (Guardian rows, twin snapshots, transition history) is written after every run and reloaded at start-up; this is tested. The default stays in-memory. The behaviour-feature timeline is not persisted.

**154. Can you reproduce a dashboard result from raw telemetry?**
Yes. The pipeline is a pure function of its input, so replaying the same capture gives the same output. That's tested.

**155. How do you trace a displayed SOH back to the measurement?**
SOH = window charge now ÷ window charge of the first 5 accepted discharges. The window charges come from integrating current between two voltages on specific discharges. The table behind it lists each discharge's cycle, window charge and gate decisions. There's no per-number audit link in the UI; it's traced through the pipeline output.

**156. How do you know the dashboard isn't showing stale data?**
The twin API now reports `last_update`, `age_seconds` and `stale` (older than 10 minutes) for every battery, so stopped telemetry can't look like "nothing changed". The dashboard also labels simulated runs.

---

## H. One shared pipeline

**157. Why one pipeline?**
So a CAN run and a serial run over the same battery can only differ because the data differs. Separate code paths would make disagreements untraceable.

**158. Data from CAN?**
Frames are decoded with the DBC (cantools), mapped to unified channels, regrouped by timestamp without filling gaps, coverage-checked, then sent into the same scoring function.

**159. From USB serial?**
Parsed and checksummed lines become the same unified table and go into the same function.

**160. From a dataset?**
A dataset loader produces the same table (CALCE needs time made continuous across files first).

**161. Are the three truly equivalent?**
No. CAN interleaves messages; serial gives whole records; datasets have their own sampling rates and missing channels. Rated capacity can be declared on serial but not on CAN.

**162. Where are differences handled?**
In each source adapter, before the shared scoring function. Anything an adapter can't resolve becomes a refusal.

**163. Units?**
Fixed unified units (V, A, °C, s, % SOC). The serial schema declares units per field. SOC given as a 0–1 fraction where percent is declared is detected and refused.

**164. Timestamps?**
Must be monotonic; CALCE's per-file restarts are joined by a function that keeps them in order. Serial reversal is refused.

**165. Missing temperature?**
Behaviour and heat scoring refuse with the reason. SOH and RUL still run, since they don't need temperature.

**166. Missing current?**
Nothing can be computed: segmentation, coulomb counting and SOH all need current. The run refuses.

**167. NaN?**
Never converted to a number. Missing fields are rejected at parse time, and estimators drop NaN rows explicitly.

**168. Why refusals as values, not exceptions?**
So a refusal travels with the result to the dashboard and the report card, together with what *did* succeed. An exception would just stop the run and lose the partial results.

**169. What bug led to that?**
A missing temperature channel was scored as healthy. `NaN > 40` evaluates to False in NumPy, so every "is it hot?" check passed for a battery with no temperature sensor.

**170. Why is NaN-as-normal dangerous?**
Because a comparison with NaN quietly answers "no". "Too hot?" — no. "Too deep?" — no. An unmeasured battery comes out looking perfectly healthy.

---

## I. SOH

**171. What is SOH?**
How much charge the battery can hold now, compared with when it was new.

**172. Mathematically?**
SOH = Q_now / Q_reference.

In our field method:
SOH = (charge delivered between V_high and V_low on this discharge) ÷ (the same quantity, median of the cell's first 5 accepted discharges).

**173. Based on capacity, resistance or power?**
Capacity.

**174. Why capacity?**
It's the standard definition (end of life is quoted as a capacity fraction), and it's what the lab ground truth measures.

**175. How do you measure capacity?**
Integrate current over time during a discharge: Q = ∫|I| dt / 3600, in Ah.

**176. Why a fixed voltage window?**
A car rarely discharges fully. A window can be measured on any discharge that passes through it, and on the plateau its charge scales with capacity.

**177. Why not integrate the whole discharge?**
Partial discharges would read as fade. A discharge that starts at 80% and stops at 40% delivers half the charge without the cell having lost anything.

**178. Partial discharge?**
If it crosses the window, it's a full measurement of the window. If the standard window is never crossed, the system learns a window inside the band this cell actually uses.

**179. How does it handle partial discharges?**
- It learns the band the first 10 discharges cover and picks its flattest part.
- It refuses windows on the end-of-discharge knee.
- The reference must form within 20 equivalent full cycles.

Result: 3.2–3.4% error on real top-of-charge partial cycling (2 cells). Near-empty partials are refused.

**180. Why ~3% (now 1.7%) median error?**
- The method fits nothing, so nothing has to transfer between cells.
- Ohmic correction keeps the window on the same part of the curve as resistance grows.
- Gates remove the cut-short and physically implausible readings.

Remaining error comes mainly from the curve not scaling perfectly with age.

**181. Ground truth?**
The CALCE cycler's measured discharge capacity on full discharges, divided by the cell's early capacity.

**182. How do you know the "new" reference is right?**
It's the median of the first five readings (the first checks within 10 cycles for sparsely tested cells). It's only right if the record starts when the cell was new. For a log that may not, the system reports "fade since logging began" and assigns no state.

**183. Sensitivity to temperature?**
Real and not measured here: CALCE cycling has no temperature channel and ran at room temperature. Capacity and voltage both shift with temperature, so field use would need temperature-matched references.

**184. Sensitivity to discharge rate?**
Large, through I × R. Uncorrected, 1C cells read 6–9% error; with step-resistance correction, about 1–2%.

**185. How does internal resistance affect it?**
Terminal voltage = OCV − I × R. As R grows with age, the fixed terminal-voltage window slides to a different part of the curve, and that slide reads as extra fade.

**186. Why did ohmic compensation help?**
It shifts the window by each discharge's own I × R, so it keeps reading the same part of the curve. Median error went from 4.3% (12 cells) to 1.7% (17 cells).

**187. ΔV = I × R and the window?**
At 1.1 A and about 0.165 Ω, terminal voltage sits about 0.18 V below the OCV. A 3.90–3.60 V terminal window is then really 4.08–3.78 V of OCV. As R rises, the shift changes. Moving the window by each discharge's own I × R holds the OCV range constant.

---

## J. RUL

**188. What is RUL?**
Cycles remaining until SOH reaches the end-of-life threshold.

**189. What threshold?**
90% of early capacity.

**190. Why 0.90?**
The CALCE trajectories end near 81%, so the conventional 80% is crossed by only one cell in 22. A threshold you never observe can't be validated. 90% is crossed by 20 of 22 cells with plenty of data afterwards.

**191. Is 90% universally accepted?**
No. 80% is the automotive convention.

**192. What if the application uses 80%?**
It's tested now, on Oxford, whose cells fade to 62–80%. The unchanged estimator, when it predicted under 500 cycles to 80%, was not exceeded 97% of the time: a safe bound. But it's imprecise (median miss 385 cycles), so we present 80% as a conservative bound, not an accurate prediction.

**193. How do you calculate RUL?**
Take this cell's SOH history up to now, smooth it with an 11-cycle rolling median, fit a straight line to the full history, and solve for where it crosses 0.90. It needs at least 30 points. It refuses a flat or rising trend, and any estimate that reaches more than twice as far ahead as its own history.

**194. Why extrapolate each cell's own trend?**
Nothing has to transfer between cells, and it was measured best among the forms we tried: straight line, square root, quadratic, full history versus a trailing window.

**195. Assumptions?**
That the recent fade rate continues, and that conditions stay similar.

**196. Is degradation linear?**
No. CALCE cells fade fast early and then flatten. That's why our estimates run early, predicting end of life too soon.

**197. What if degradation accelerates?**
After a "knee", the line under-predicts the fade, so the estimate runs late. That's the unsafe direction. Our data showed mostly early errors, but a knee would reverse that.

**198. What if it temporarily slows?**
The estimate drifts later; the median smoothing damps short-term changes.

**199. Why is ±20 cycles only valid near end of life?**
Errors grow with distance to end of life. There are two honest ways to state accuracy. By the *true* distance: within ±20 cycles 73% of the time when truly under 25 cycles away; 44% at 25–50; 0% at 200–400. By the *prediction*, which is what a user sees: when the estimate says under 25 cycles, the cell lasted at least that long 95% of the time, typically 23 cycles longer. So near end of life it works as a safe lower bound: it warns early.

**200. Can you claim 88% (now 73%) for the whole lifetime?**
No.

**201. Why not?**
Measured far from end of life, it's mostly wrong (median miss 144 cycles at 200–400 cycles out).

---

## K. XGBoost vs LSTM

**First, the setup every answer in K, L, U, V and W depends on.** The models were benchmarked on one task: predict SOH for each cycle of a NASA cell from 7 per-cycle features. The data is 2,585 rows, 31 cells and 9 cohorts. Evaluation used leave-one-cell-out (31 folds) and leave-one-cohort-out (9 folds). None of these models predicts RUL, and none is used by the running system.

**202. Why XGBoost?**
The strongest standard method for tabular data, and widely used in battery SOH papers. It was one of 12 methods run through identical validation.

**203. Why LSTM?**
To test whether a genuine sequence model, which sees 10 consecutive cycles, beats per-cycle features. It didn't.

**204. Why GPR?**
Common in battery papers because it gives uncertainty, and it does well on small data.

**205. Why Random Forest?**
A standard tabular baseline, and less prone to overfitting than boosting.

**206. Why ElasticNet?**
A regularised linear baseline. If it matches the complex models, they aren't adding much.

**207. Why multiple models at all?**
The research question was whether any published method generalises across protocols. That needs a range of methods judged on the same validation.

**208. Why can't LSTM do everything XGBoost does?**
In principle it can learn the same mapping. In practice, with 2,360 training windows it overfits to each cell's trajectory shape. XGBoost on tabular features needs less data.

**209. Why not give the LSTM the whole battery history?**
Cells have different lengths, and the longer the sequence, the fewer independent training examples you get. A 10-cycle window keeps all 31–32 cells, and it already gives the LSTM twice the context of the 5-cycle trailing features.

**210. XGBoost can use time-derived features too. What's fundamentally different?**
XGBoost sees one row: the current cycle's features, including the cycle number and trailing averages. The LSTM sees an ordered sequence and can learn patterns in how features change. Here, that extra capacity didn't help.

**211. What does XGBoost see?**
7 features per cycle: average temperature, maximum temperature, average stress score, deep-discharge duration, aggressive-discharge count, average SOC, cycle number.

**212. What does the LSTM see?**
The same 7 features for the 10 consecutive cycles ending at the prediction cycle: a 10×7 matrix.

**213. What temporal sequence is supplied?**
Cycles k−9 to k of one cell, sorted by cycle, predicting SOH at cycle k. Early cycles are left-padded, and that is recorded.

**214. Sequence length?**
10 cycles.

**215. How are sequences built?**
One window per row, per cell. A window never mixes two cells, and never spans train and test.

**216. Why isn't XGBoost a sequence model?**
It treats each row independently; order means nothing to it. Any history has to be engineered into the row's features.

**217. XGBoost's advantage with engineered tabular features?**
It's data-efficient, handles nonlinear interactions, and is robust to scaling.

**218. LSTM's advantage with temporal dependencies?**
It can learn patterns across time without hand-made features, given enough data. We didn't have enough.

**219. Why GPR?**
See 204.

**220. What does GPR give that XGBoost doesn't?**
A predictive mean and variance for each prediction.

**221. Does GPR give uncertainty?**
Yes, from the kernel (Matern ν = 1.5 plus a white-noise term).

**222. How do you interpret that uncertainty?**
We didn't use it in the study, which scored point predictions only. Strictly, it's the model's uncertainty *given its own assumptions*. Under protocol shift those assumptions are wrong, so the intervals would be too narrow. Our conformal-coverage work studies exactly this failure.

**223. Why does the slide say all three do SOH/RUL?**
It shouldn't. They were evaluated for SOH only. Fix the slide.

**224. Are all three validated for both SOH and RUL?**
No. SOH only, and none of them kept its skill under LOCO.

**225. Best model per task?**
No model was selected for deployment. SOH and RUL in the product use non-learned estimators.

**226. One model assumed globally best?**
No. We explicitly withdrew ranking claims.

---

## L. The model comparison table

**227. What does 4.21% LOBO MAE mean?**
XGBoost's SOH predictions were off by 4.21 percentage points on average, on a held-out cell from a protocol it had seen.

**228. 7.62% LOCO MAE?**
Off by 7.62 points on average on cells from a protocol it had never seen.

**229. 0.459 LOCO R²?**
Averaged over held-out cohorts, it explained about 46% more variance than predicting the training mean. The 95% interval is −2.80 to 0.86.

**230. Why is GPR's LOCO MAE 17.06%?**
The GPR trained on a subsample (800 rows, because exact GPR scales as n³), and with a smooth kernel it extrapolates badly outside the conditions it saw. Its LOBO MAE was 4.48%, so it fits seen protocols fine and fails on new ones.

**231. Why is the LSTM's LOCO R² negative?**
It learned cell- and protocol-specific trajectory shapes that are wrong on a new protocol. Its predictions there are worse than predicting the average.

**232. Can R² be negative?**
Yes.

**233. What does negative R² mean physically?**
The model's errors are larger than those of always guessing the training-set average SOH. Physically, it's predicting the wrong fade pattern.

**234. Why is Random Forest worse than XGBoost under LOCO?**
RF scored best under LOBO (0.773) but 0.150 under LOCO. Averaging many deep trees fits seen protocols closely and transfers poorly here. With overlapping intervals, the "worse" isn't statistically established.

**235. Why is ElasticNet competitive?**
The real signal is mostly "older cells have less capacity", which is close to linear. Simple models capture it and overfit less.

**236. Why does the age-only baseline reach 0.406 LOCO?**
SOH falls with cycle count in every cell, so cycle number alone predicts a lot. It also lost the least between LOBO and LOCO (−0.076).

**237. If the baseline is nearly as good, why ML?**
On this data, the behaviour features plus ML add about 0.05 R² over counting cycles, within the noise. That's the finding, not a reason to hide the baseline.

**238. What does the baseline tell you?**
How much of the "skill" is just knowing the battery's age. Any model has to beat it to show it has learned anything about usage.

**239. Why MAE?**
It's in the same units as SOH (percentage points), easy to say out loud, and not dominated by a few large errors.

**240. Why not MSE?**
Squared units are hard to interpret. It was computed implicitly through R².

**241. Why not RMSE?**
We computed it (LOCO RMSE for XGBoost is 0.122, i.e. 12.2 points) but didn't headline it. It penalises big misses more; quote it if asked.

**242. Why R²?**
It compares against a baseline (the training mean), so it shows whether the model beats doing nothing. MAE alone can't.

**243. Why not F1 or F2?**
They're classification metrics. SOH prediction is regression.

**244. Why not accuracy?**
Same reason. Accuracy is for discrete classes.

**245. Why not precision or recall?**
Classification metrics again.

**246. What metric for healthy / degraded / failing classification?**
Per-class precision and recall, plus a confusion matrix. Recall on "failing" matters most.

**247. Which metric if underestimating degradation is dangerous?**
For classification, recall on the degraded class. For regression, signed error (bias) and the rate of optimistic errors, not just MAE. We report RUL bias for this reason.

**248. Why doesn't your LOCO R² table show LOBO R²?**
The slide omitted it. The study computed it: XGBoost LOBO R² 0.732 (CI 0.579–0.874).

**249. Was LOBO R² calculated?**
Yes, for all 12 methods.

**250. Why isn't it shown?**
Slide space, not concealment. Show the full table.

**251. If not calculated, why not?**
N/A — it was calculated.

**252. Are you selectively showing metrics?**
The slide was. The report isn't: it shows both splits, both metrics, and the intervals. Correct the slide.

---

## M. The big research result

**253. Why does performance collapse under LOCO?**
Models learn relationships that hold within a protocol (e.g. fade vs temperature at one ambient) and change between protocols. Temperature is entangled with protocol in NASA: the 4 °C cells *are* their own cohorts.

**254. What does it say about cross-cohort generalisation?**
With 31 cells in 9 cohorts, none of the published methods we tested generalised reliably to an unseen protocol.

**255. Does that mean your ML models are useless?**
For deployment on unseen conditions, yes on this data. As evidence, no: they show what doesn't work, and why the shipped estimators avoid learning across cells.

**256. If model selection is uncertain, why five (twelve) models?**
Because the question was whether *any* would generalise. You only learn that by testing several under the same rules.

**257. How did you test whether model differences are meaningful?**
Bootstrap 95% intervals on each method's median score, resampling folds (not rows), 2,000 resamples. Overlapping intervals mean the difference isn't established.

**258. What do the bootstrap intervals tell you?**
LOCO intervals are wide (XGBoost −2.80 to 0.86) and all overlap. Model ranking isn't resolvable at 9 cohorts.

**259. Why do the intervals overlap zero?**
On some held-out cohorts every method is worse than predicting the mean. With 9 folds, those cohorts pull the lower bound below zero.

**260. What does it mean that ranking intervals overlap?**
You can't say which method is better.

**261. Why did you withdraw ranking claims?**
Six earlier claims ("X beats Y") rested on point estimates. Once intervals were computed, none was supported. One headline rested on a single unscreened cell.

**262. Why is this a useful result?**
Most battery ML papers report leave-one-cell-out scores and pick a winner. We show that this evaluation overstates performance, and that winners can't be told apart. That saves others from deploying a model that won't transfer.

**263. Why "conditioning matters more than model choice"?**
The spread across 14 methods was 2.06 R², smaller than the 3.66-wide interval on one method. Meanwhile two data-conditioning fixes cut worst-case SOH error 5×.

**264. What were the two fixes?**
1. Reject cut-short discharges that were being read as fade: a cycle must deliver at least 50% of the reference discharge.
2. Compensate the voltage window for I × R sag.

**265. How did they change the error?**
Fix 1 took the worst cell from 52% to about 10% error. Fix 2 roughly halved median error on high-rate cells (6.8% to 3.5%). In the later field study, step-resistance correction took median error from 4.3% to 1.7%.

**266. What was the 52% result?**
CS2_9's late cycles delivered 0.03 Ah against 1.13 Ah early. The estimator read that as a 97% capacity loss; the cell had lost 8%. Those cycles were truncated discharges, not fade.

**267. Why did it become ~10%?**
The truncated cycles are now detected (too little total charge) and excluded, not read as fade.

**268. What does that teach about battery ML pipelines?**
How each cycle is cut and labelled can create fake degradation larger than any model's error. Check the data before fitting anything.

---

## N. Battery science

**269. Calendar ageing?**
Capacity loss while the battery sits, mainly from slow side reactions (SEI growth), even with no use.

**270. Cycle ageing?**
Loss from charging and discharging: SEI growth, particle cracking, loss of active material, lithium plating.

**271. What accelerates calendar ageing?**
High temperature and high state of charge during storage.

**272. Cycle ageing?**
High or low temperature, high C-rates (especially fast charging), deep cycles, high charge voltage.

**273. Temperature?**
Heat speeds up side reactions (SEI). Cold makes lithium plating likely during charging. Our data: hotter use went with faster fade in 7 of 7 NASA cells, but the size of the effect didn't transfer between protocols.

**274. High SOC?**
High electrode potential speeds electrolyte oxidation and SEI growth, especially in storage.

**275. Deep discharge?**
More mechanical stress per cycle. Very low voltage can dissolve the copper current collector.

**276. Charging rate?**
High rates raise the risk of lithium plating and heating.

**277. Discharge rate?**
More heat and mechanical stress; less damaging than fast charging.

**278. C-rate?**
Current divided by capacity. 1C fully discharges the cell in about one hour; 0.5C in two hours.

**279. Why does temperature matter for SOH estimation?**
Capacity and voltage both change with temperature, reversibly. A cold cell reads lower capacity without having aged. CALCE cycling has no temperature channel, so this effect isn't corrected in our results.

**280. Internal resistance?**
Opposition to current inside the cell: electrolyte, electrodes, contacts, interfaces. It causes the voltage drop under load.

**281. Why does it rise with age?**
SEI thickens, contacts degrade, electrolyte is consumed, particles crack.

**282. How does resistance affect terminal voltage?**
V_terminal = OCV − I × R during discharge. More resistance means lower voltage under the same load.

**283. Polarisation?**
The extra voltage loss beyond the instant ohmic drop, from charge transfer and diffusion. It builds over seconds to minutes under load. Our step resistance (0.165 Ω vs the cycler's 0.099 Ω) includes some of it.

**284. Capacity fade?**
Loss of the charge a cell can store.

**285. Power fade?**
Loss of the power it can deliver, mainly from rising resistance.

**286. Lithium plating?**
Metallic lithium depositing on the anode instead of entering it, during cold or fast charging. It's irreversible loss, and a safety risk.

**287. SEI growth?**
The solid-electrolyte interphase is a film on the anode that consumes lithium as it grows. It's the main slow ageing mechanism.

**288. Which mechanisms can your telemetry observe?**
Capacity fade (window charge) and resistance growth (the voltage drop at each load step). Resistance is now reported on the report card. It tracks the lab cycler's own resistance on 19 of 20 cells (median life-long correlation 0.86; growth 1.26× vs 1.29×), per `reports/metrics/calce_resistance/`.

**289. Which can't you distinguish?**
SEI vs plating vs active-material loss. Separating them needs incremental capacity analysis on low-rate data, or post-mortem analysis. We tried ICA features; they didn't help prediction.

**290. Cause or effect?**
Only the observable effect. We don't claim mechanisms.

---

## O. CALCE-specific

**291. Why is CALCE Type 3 hard to score?**
Its discharge switches between rates about six times within one cycle. No single constant-current part is a full discharge, so capacity-based SOH isn't defined the usual way.

**292. What happens to current in Type 3?**
It steps between different levels within one discharge.

**293. Why can't every discharge segment count as a capacity measurement?**
A segment covers only part of the charge range. Its charge reflects where it starts and stops, not capacity.

**294. What happens if you compute capacity from an incomplete discharge?**
You get a small number that looks like heavy fade.

**295. Could that create artificial degradation?**
Yes. It did: the 52% error on CS2_9.

**296. How did you detect it?**
The voltage-window ratio claimed 97% capacity loss while the cycler's own capacity showed 8%. The ICA area (which integrates to charge) fell 97% too, which was impossible. Inspecting those cycles showed about 100 samples delivering 0.03 Ah.

**297. What did you do with the affected data?**
A gate now rejects cycles delivering under 50% of the reference discharge. Type 3 cells are still scored, but with their limitation stated: they are the worst 3 cells (6.4–10.3%).

**298. Why "provably unscoreable", not just "bad data"?**
The data isn't wrong. The protocol simply never produces a constant-current full discharge, so the quantity is undefined. That's a property of the test design, not an error. (Under the later field method the Type 3 cells *are* scored, at the worst errors.)

**299. What if published papers use the same flawed interpretation?**
They would report fake degradation, or inflated errors, on those cells. We can't say whether specific papers did; we didn't audit them.

---

## P. Digital twin

**300. Where is the twin?**
In `src/bms/digital_twin/twin.py` and `telemetry/twin_integration.py`. It holds a per-battery snapshot (state, health index, RUL, failure likelihood), detects state transitions, and keeps a bounded history per battery.

**301. What's the physical asset?**
A battery: a dataset cell, a simulated cell, or the rig cell.

**302. Its digital counterpart?**
The snapshot plus history for that battery ID.

**303. What state variables?**
Twin state (NORMAL / MODERATE_RISK / HIGH_RISK / FAILURE_IMMINENT), health index, RUL, replacement policy, failure likelihood, timestamp.

**304. How often is it updated?**
Once per pipeline run on new telemetry. It isn't streaming continuously.

**305. What's the link between the physical system and the twin?**
The telemetry pipeline: serial or CAN in, scoring, then the twin update.

**306. Where does it run?**
In the Python process, in memory.

**307. Is it just a dashboard representation?**
More than that, less than a true twin. It keeps state and history and detects transitions. But it has no model of its own: its state is a relabel of the health state.

**308. If so, why call it a digital twin?**
The honest answer: the name came from the original plan, and it overstates what was built. Call it a "battery state tracker" or "digital shadow".

**309. How is it different from a database record?**
It adds transition detection and history semantics. A database row with a history table could do the same.

**310. Does it keep state?**
Yes, in memory, per battery, bounded.

**311. Does it update from real-time telemetry?**
From each live or replayed run, yes.

**312. Can it predict future battery state?**
Only through the RUL figure it carries. It doesn't simulate the future.

**313. Can you change a parameter and see the predicted consequence?**
No. There's no what-if or simulation capability.

**314. Does it stay synchronised with the physical battery?**
Only as often as runs happen. There's no heartbeat.

**315. What if communication is interrupted?**
It keeps the last state and is now flagged stale (`stale: true`, with its age) once no update has arrived for 10 minutes.

**316. What protocol?**
None of its own: serial or CAN into the pipeline, then HTTP/JSON to the dashboard.

**317. "A digital twin needs a different communication protocol"?**
That premise is wrong. A twin is defined by a model synchronised with the physical asset, not by any protocol.

**318. What evidence does the twin exist?**
The code, its tests (`tests/test_digital_twin.py`), and the API route `/telemetry/twin/{battery_id}`, which returns snapshots and transitions.

**319. Can you show battery → telemetry → state update live?**
With a replayed capture, yes. With the live rig the pipeline runs, but a resting cell never changes state.

**320. Disconnect the battery and show the twin going stale?**
Yes. After 10 minutes without telemetry, `/telemetry/twin/{id}` returns `stale: true` and the age in seconds.

**321. Reconnect and resync?**
The next run updates it.

**322. What's synchronised?**
State, health index, RUL, replacement policy.

**323. Update latency?**
Not measured.

**324. What's genuinely twin and what's visualisation?**
Genuinely stateful: snapshot, history, transitions. Everything else is presentation. Nothing is a physics or simulation twin.

---

## Q. Data storage and logging

**325. Where is raw hardware data logged?**
Text files of the raw wire lines, e.g. `data/interim/rig_stage_b_voltage_verified.txt`.

**326. CSV?**
Pipeline outputs and study results, yes.

**327. Database?**
No.

**328. JSON?**
API responses, and `results_showcase.json`.

**329. In memory?**
The API's fleet store and twin history.

**330. Research datasets?**
`data/raw/` (gitignored, not redistributable). Derived caches in `data/interim/` (gitignored).

**331. Processed features?**
`data/features/` from `main.py`; CSVs in `reports/metrics/` from studies.

**332. Predictions?**
CSV files from scripts; memory in the API.

**333. How are model versions tracked?**
Through git commits. The benchmark results record the method, configuration and seeds. There's no model registry.

**334. Can you reproduce an old prediction?**
Yes: check out the commit and rerun. The pipeline and studies are deterministic, with fixed seeds.

**335. How do you know which model produced a prediction?**
Outputs carry provenance fields, e.g. `rul_method`, `rul_validated`, `state_basis`, `soh_reference`, `window_mode`.

**336. Timestamps?**
Yes, `test_time_s` per sample, and run timestamps on twin snapshots.

**337. Source information?**
Yes: source name, transport, declared channels, cell ID, measurement unit.

**338. Provenance?**
Yes: simulated vs measured; which estimator; which reference.

**339. What if the server crashes?**
The API's state is lost. Raw captures on disk survive and can be replayed.

**340. What if the network drops?**
Live serial capture is local, not networked. Dashboard requests fail with an error (502 from the gateway).

**341. Can you replay history?**
Yes. Serial and CAN logs both have replay functions, and replay is identical to live by construction.

**342. Same input, same output?**
Yes. The pipeline is a pure function of its input; this is tested.

---

## R. BMS engineering

**343. Core functions of a conventional BMS?**
Measuring cell voltages, current and temperature; protection (over/under-voltage, over-current, over-temperature); SOC and SOH estimation; cell balancing; thermal management; contactor control; communication.

**344. Which does BEACON implement?**
Measurement (on the rig), SOH estimation, RUL estimation, communication (serial/CAN ingestion), diagnostics and reporting.

**345. Which not?**
Protection, balancing, thermal management, contactor control, charge/discharge control, and SOC estimation (it takes SOC from the source).

**346. Balancing?**
No.

**347. Charging control?**
No.

**348. Discharging control?**
No.

**349. Contactors?**
No.

**350. Safety cutoffs?**
No.

**351. SOC?**
No. The rig's SOC is the firmware's coulomb count from an assumed start; CAN SOC comes from the vehicle's BMS.

**352. SOH?**
Yes.

**353. RUL?**
Yes.

**354. SOC vs SOH?**
SOC: how full the battery is now, as a percentage of its current capacity. SOH: how big that capacity is compared with new.

**355. SOH vs RUL?**
SOH is the current condition. RUL is how long until SOH reaches a threshold.

**356. Why can't you just derive RUL from SOH?**
You need the *rate* of change of SOH, which depends on future use. One SOH value gives no rate; a history gives a trend, which is what we extrapolate.

**357. Coulomb counting?**
Integrating current over time to track charge in or out.

**358. Its limitations?**
Sensor bias builds up, it needs a known starting point, and it drifts without recalibration.

**359. How does current-sensor bias affect SOC?**
A constant bias adds linearly with time. 1 mA of bias is 0.024 Ah per day, about 1% of a 2.2 Ah cell per day.

**360. What if the current has an offset?**
Coulomb-counted charge is biased. Our window charge integrates only over each discharge, so an offset shows up as a constant percentage error on both the reference and later readings, and partly cancels in the ratio.

---

## S. CAN and serial

**361. Why support CAN?**
It's how real vehicle BMSs report data.

**362. Why USB serial?**
It's how a bench rig or development board reports data.

**363. What is CAN actually giving you?**
In our testing: frames decoded from a published DBC example (a Renault Twizy BMS definition) and replayed logs. It has never been tested on a live vehicle bus.

**364. A CAN frame?**
A short message (up to 8 data bytes in classic CAN) with an identifier, broadcast on a shared bus.

**365. A CAN ID?**
The frame's identifier: it says what the message is and sets its priority.

**366. A DBC file?**
A text definition mapping each CAN ID and its bit fields to named signals, with scaling, offset and units.

**367. How does a DBC map raw bytes to engineering values?**
For each signal: take its bits, read them as an integer, then value = raw × factor + offset.

**368. What if the definition is wrong?**
Values come out plausible but wrong. We check that the DBC supplies the required channels (refusing if, say, temperature is missing), but we can't detect a wrong scale factor.

**369. How do you detect corrupted serial data?**
Checksum, field range checks, record completeness, monotonic time.

**370. Why a checksum if serial already works?**
UART has no end-to-end integrity check in practice. Noise, buffer overruns and resets corrupt lines.

**371. What does 83/83 frames prove?**
That every line of that capture arrived intact and parsed.

**372. What does zero dropped samples prove?**
That timing and transport held for those 82 seconds.

**373. Sensor accuracy?**
No.

**374. Health-prediction accuracy?**
No.

**375. Long-term reliability?**
No.

---

## T. Feature engineering

**376. What are the engineered features?**
Per row: high-temperature flag, deep-discharge flag, high-SOC flag, fast-charge flag, aggressive-discharge flag, a stress score, rolling means and standard deviations. Per cycle: average and maximum temperature, average stress, deep-discharge duration, aggressive-discharge count, average SOC, cycle number.

**377. Why engineer them?**
To turn usage behaviour into numbers a model can use. That was the original "behaviour-aware" hypothesis.

**378. Physically meaningful?**
Temperature, C-rate events, SOC extremes and cycle count are physical. Stress score is a made-up composite.

**379. Purely statistical?**
Rolling standard deviations; the stress score.

**380. Rolling mean?**
The average over the last N samples (N = 50 rows here).

**381. Rolling standard deviation?**
The spread over the last N samples.

**382. Temperature exposure?**
Time or fraction spent above a threshold, and average temperature.

**383. Deep-discharge duration?**
Number of samples with SOC below 20%.

**384. Aggressive discharge?**
Discharge current above 1C.

**385. How was "aggressive" defined?**
C-rate greater than 1, a hand-picked heuristic. An earlier fixed 2 A threshold never fired on real data (max observed 1.54 A) and was replaced.

**386. "High temperature"?**
Above 40 °C (row flag). The Guardian's advice uses an average above 35 °C.

**387. How were the thresholds chosen?**
By hand, from common guidance. Not fitted. An audit found 61% of the mean risk score came from terms that never changed across the real data, because the thresholds sat outside its range.

**388. Are they chemistry-dependent?**
They should be. These aren't tuned per chemistry.

**389. Can feature engineering introduce bias?**
Yes. Fixed thresholds outside the data's range made features constant and the risk score uninformative.

**390. Could a rolling feature leak future information?**
It could, if it were centred. Ours are trailing (past only).

**391. How did you ensure that?**
Trailing windows by construction. The LSTM window ends at the prediction cycle. The RUL estimator is tested by changing the future and checking the estimate doesn't move.

---

## U. LSTM

**392. What is an LSTM?**
A recurrent neural network with gated memory cells that can carry information across many time steps.

**393. Why "recurrent"?**
It processes a sequence one step at a time, feeding its own state back in.

**394. What problem does the LSTM cell solve?**
Learning long-range dependencies, which plain RNNs struggle with.

**395. Vanishing gradients?**
When training through many steps, gradients shrink exponentially, so early steps barely learn.

**396. Input, forget and output gates?**
- Input: how much new information to write to memory.
- Forget: how much old memory to keep.
- Output: how much memory to expose as output.

**397. Your input shape?**
(batch, 10 cycles, 7 features).

**398. Sequence length?**
10.

**399. Layers?**
1 LSTM layer, hidden size 32, then a linear output layer.

**400. Loss?**
Mean squared error.

**401. Optimiser?**
Adam.

**402. Learning rate?**
0.001.

**403. How did you prevent overfitting?**
A small network (about 5,300 parameters), early stopping (patience 8, at most 60 epochs) on 20% of the training *cells* held out, and a scaler fitted on training windows only.

**404. How much training data?**
2,360 windows in total, from 31–32 cells. A LOCO fold trains on about 8 of 9 cohorts.

**405. Enough for an LSTM?**
No, in practice. That's part of the result.

**406. Why did the LSTM do worse than XGBoost?**
It learned cell-specific trajectory shapes that didn't carry over to new protocols. With limited data, XGBoost's simpler mapping held up better.

**407. Is LSTM a worse algorithm?**
No. It wasn't demonstrably better on this data, these features and this validation.

---

## V. XGBoost

**408. Gradient boosting?**
Building an ensemble of small trees one at a time, each fitted to correct the current ensemble's errors.

**409. Why "gradient" boosting?**
Each new tree fits the gradient of the loss with respect to the current predictions; for squared error, that's the residuals.

**410. Decision tree?**
A model that splits data by feature thresholds into leaves, each predicting a constant.

**411. What does each successive tree do?**
Predicts what's left of the error, scaled by the learning rate.

**412. Why does XGBoost work well on tabular data?**
It handles nonlinearity and interactions, ignores irrelevant features, is regularised, and needs little preprocessing.

**413. Hyperparameters tuned?**
None. Fixed: 300 trees, learning rate 0.05, max depth 4, subsample 0.8, L2 1.0.

**414. Learning rate?**
How much each new tree's correction counts.

**415. Tree depth?**
The maximum number of splits from root to leaf. It limits how many feature interactions a tree can capture.

**416. Number of estimators?**
How many trees.

**417. How did you avoid overfitting?**
Shallow trees, small learning rate, row subsampling, L2 regularisation, and fixed, untuned settings.

**418. How did you select hyperparameters without contaminating the test cohort?**
We didn't tune at all. Fixed defaults can't leak test information. The one exception is ElasticNet, whose penalty is tuned by 5-fold CV *inside the training fold only*.

---

## W. GPR

**419. Why GPR?**
See 204.

**420. What is a Gaussian process?**
A distribution over functions: any finite set of function values is jointly Gaussian, with correlations set by a kernel.

**421. How is GPR different from ordinary regression?**
It's non-parametric and gives a full predictive distribution, not just a point estimate.

**422. Why is it useful for battery health?**
Smooth degradation curves, small data, and built-in uncertainty.

**423. What does the kernel represent?**
How similar outputs are for similar inputs. Ours is Matern ν = 1.5 (less smooth than RBF) plus white noise for measurement noise.

**424. What uncertainty does GPR give?**
Predictive variance at each input.

**425. Epistemic, aleatoric or both?**
Both: the white-noise term models aleatoric noise; uncertainty growing away from the training data is epistemic. Both are valid only if the kernel assumptions hold.

**426. Why did GPR do poorly under LOCO?**
It extrapolates toward its prior mean in unfamiliar input regions. It also trained on an 800-row subsample.

**427. Does poor GPR performance make uncertainty estimates useless?**
Not useless, but under protocol shift they aren't trustworthy without checking coverage, which is exactly what our conformal study found.

---

## X. Statistics and validation

**428. MAE mathematically?**
MAE = (1/n) Σ |ŷᵢ − yᵢ|.

**429. What does 7.62% MAE mean for a battery?**
On a cell at, say, 85% health, the model might say 77% or 93%. That spans the difference between "plan replacement" and "fine".

**430. R² mathematically?**
R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)², where ȳ is the training-fold mean in our study.

**431. R² = 0?**
No better than predicting the mean.

**432. R² < 0?**
Worse than predicting the mean.

**433. Baseline for R²?**
The training-fold mean SOH. `train_mean` scores exactly 0, which checks the metric.

**434. Low MAE but poor R² — how?**
If the test cell's SOH barely varies, small errors can still be large relative to its variance. Within one cohort, variance is small, so R² punishes hard.

**435. Why can R² mislead for degradation data?**
It depends on the test set's spread. Pooling cells of different ages inflates R², because the model only has to tell old from new. That's why we also report MAE and per-fold results.

**436. Why bootstrap intervals?**
To show how uncertain each method's median score is with only 9 cohorts.

**437. How many resamples?**
2,000 (seed 20260925).

**438. What exactly did you bootstrap?**
Folds (held-out cells or cohorts), not rows. Rows within a cell aren't independent.

**439. Why do the intervals overlap zero?**
On some cohorts, models score below zero, and with 9 folds those pull the lower bound negative.

**440. What does significance mean here?**
Whether a difference in median scores is distinguishable from fold-to-fold variation. Overlapping intervals mean it isn't.

**441. Null hypothesis?**
That two methods have the same median score across folds. We used intervals, not formal tests.

**442. Formal hypothesis tests?**
For specific claims, e.g. temperature vs fade (p < 0.0001, cohort-controlled) and Spearman correlations. Not for model rankings; intervals were used instead.

**443. Statistical vs practical significance?**
Statistical: is the effect distinguishable from noise? Practical: is it big enough to matter? Temperature's effect on fade was statistically significant but small (in-sample R² 0.015).

---

## Y. Generalisation

**444. Why trust it on a new battery?**
Trust the fitted ML models? Don't. The shipped SOH estimator doesn't need training data. It needs the new battery's own early discharges and a plateau window, and it refuses without them.

**445. What distribution does the rig cell come from?**
An HONGLI ICR-18650 (LCO-type, "ICR"). Not from NASA or CALCE.

**446. Same chemistry?**
LCO family, per its ICR designation. Not verified beyond the label.

**447. Same manufacturer?**
No.

**448. Same age?**
Unknown.

**449. Same conditions?**
No.

**450. Evidence for generalisation?**
For the shipped SOH method: it was validated on 17 cells across 7 CALCE protocols (CS2 and CX2 families) with no fitted cross-cell parameters. Nothing on the rig cell yet.

**451. Out-of-distribution detection strategy?**
Physical gates rather than statistical OOD:
- coverage of declared channels
- discharge completeness
- reference formed early
- plateau steepness
- SOH plausibility bounds (0.5–1.1)
- RUL horizon bound

For fitted models, a cohort-matching check refuses inputs outside every known cohort's envelope.

**452. What if a new battery behaves unlike anything seen?**
The fitted models aren't in the product. The estimators depend only on that battery's own data. If its curve has no plateau in the used band, or its references are late, they refuse.

**453. Does BEACON refuse?**
Yes. That's the core design.

**454. If not, why not?**
N/A.

---

## Z. Safety and failure modes

**455. Voltage sensor fails?**
Records without voltage are rejected. SOH needs voltage, so it refuses with "telemetry has no voltage channel".

**456. Current sensor fails?**
No segmentation, no capacity, no SOH. The run refuses.

**457. Temperature sensor fails?**
The firmware stops declaring it. Behaviour and heat scoring refuse; SOH and RUL continue.

**458. Model produces NaN?**
Estimators return NaN with a refusal reason, never a substituted number.

**459. Impossible SOH?**
Below 0.5 it's withheld. If over 20% of a cell's readings are below 0.5, the whole cell is refused. Above 1.10 it's withheld.

**460. Can SOH exceed 100%?**
Yes, slightly: noise, temperature and reversible recovery can push readings to 101–105%. Up to 110% is accepted; above is withheld.

**461. Can RUL go negative?**
No. Once past the threshold it reports 0, with a note.

**462. Bounds enforced?**
- SOH in 0.5–1.1.
- RUL ≥ 0, and reaching no more than 2× the history length ahead.
- Field ranges on every wire value.
- Plausible rated capacity (0.01–500 Ah).

**463. Input outside the training distribution?**
See 451–452.

**464. Does it ever refuse?**
Yes, often, with reasons.

**465. Why is refusing better than a possibly misleading number?**
A plausible wrong number gets acted on. A refusal tells you what's missing. With batteries, false reassurance is the dangerous direction.

**466. How do you tell "healthy" from "not enough data"?**
They're different outputs. "Healthy" needs a measured SOH ≥ 95% against a beginning-of-life reference. With too little data the card says "Not available:" plus the reason, and assigns no state. The NaN bug was exactly the failure to make this distinction.

---

## AA. Dashboard attack

**467. Point to this number. Where did it come from?**
Know the default first: `python main.py` builds the dashboard from a **simulated** fleet (20 simulated batteries), and the page labels it as simulated. Measured numbers appear only for serial, CAN or dataset runs. Chain for each number:

| Number | Chain | Kind |
|---|---|---|
| Voltage, current, temperature | sensor or dataset → parser → unified schema → UI | raw |
| SOH (measured) | discharges segmented from current → charge integrated across a voltage window → ÷ first-5-discharge reference → median of last 5 | calculated, validated on CALCE |
| SOH series on the chart | per-cycle capacity ÷ initial capacity | calculated |
| RUL | SOH history → 11-cycle median → straight-line fit → crossing at 0.90; `rul_validated=true`. Otherwise the heuristic, labelled UNVALIDATED | calculated or heuristic |
| Health index | hand-weighted penalty terms | heuristic |
| Risk score / level | hand-weighted thresholds | heuristic |
| Twin state | relabel of battery state | heuristic or measured, per `state_basis` |
| Attribution bars | exact Shapley split of the heuristic score | calculated, explains the heuristic only |

**468. Which numbers are raw?**
Voltage, current, temperature, SOC (as reported by the source).

**469. Calculated?**
Measured SOH, charge per discharge, RUL by fade extrapolation, the attribution values.

**470. ML predictions?**
None.

**471. Heuristics?**
Health index, risk score and level, stress score, the heuristic RUL fallback, the state when `state_basis = heuristic_index`.

**472. Which have ground truth?**
SOH and RUL methods, through the CALCE studies. Not on the rig or the simulated fleet.

**473. Which have uncertainty?**
None are model confidence intervals. SOH and RUL carry validation-derived error bands on the report card; the heuristic scores carry none, and are labelled heuristic on the dashboard ("What each number is").

**474. Experimental?**
The learned-window (partial-discharge) SOH: validated on 2 cells per band.

**475. Validated?**
Field SOH (17 cells) and fade-extrapolation RUL (near end of life).

**476. Merely visualisation?**
Sparklines, usage breakdown, and the twin badge colour.

---

## AB. "Show me live"

Rehearse this. Use a recorded capture if the board is unreliable on the day, and say that you are.

**477. Raw sensor value?**
Serial monitor at the firmware's baud; or open `data/interim/rig_stage_b_voltage_verified.txt`.

**478. Transmitted frame?**
A `BEACON1 D t=… v=… i=… soc=…*CS` line.

**479. Checksum?**
The two hex digits after `*`: the XOR of every byte of the body. Show `xor_checksum` in `serial_schema.py`.

**480. Backend receiving it?**
`python scripts/run_serial_demo.py --replay data/interim/rig_stage_b_voltage_verified.txt`. It prints frames read and decoded, stages completed, and refusals.

**481. Normalised telemetry object?**
The `telemetry` table in the result: `test_time_s`, `voltage_v`, `current_a`, `soc`.

**482. Feature extraction?**
Shown only when temperature and rated capacity are both present. Otherwise you'll see the refusal, which is correct.

**483. Health calculation?**
`python scripts/health_report.py --calce data/raw/calce/CS2/Type2/CS2_35.zip --upto-cycle 90`. It ends with the lab check.

**484. API response?**
Start `uvicorn src.bms.api.app:app`, then POST `/telemetry/serial/replay` and GET `/telemetry/latest/{id}`, or open `/docs`.

**485. Dashboard updating?**
`docker compose up`, or run the API, gateway and client separately.

**486–487. Now disconnect the sensor. What does it do?**
The firmware sends a status line ("temperature channel unavailable … check wiring") and stops declaring the channel. The host refuses the dependent features by name. Nothing becomes "healthy".

**488–489. Send an invalid frame.**
The checksum fails, so the line is rejected and counted. Enough rejections and the run is refused.

**490–491. Give it NaN.**
The field is omitted or fails to parse, so the record is rejected. NaN never enters scoring.

**492–493. Impossible temperature?**
Outside −40 to 150 °C it fails the field range check and is rejected, not clamped. Voltage is now also checked against what the rig *declared*: for `unit=cell`, anything outside 0–5 V is rejected per record. A live injection of 999 V found this gap; the 0–1000 V field range alone had let it through, because packs are a legal unit. `python scripts/panel_demo.py` step c) shows it.

---

## AC. Digital twin / architecture claims

**494. Why do you need a digital twin?**
Honestly, you don't need the name. What's useful is per-battery state history and transition alerts ("this battery moved to DEGRADED").

**495. What does the twin hold that the database doesn't?**
There is no database. The twin holds the only history in the API.

**496. What does it hold that the dashboard doesn't?**
Transition history, and failure likelihood (a monotonic transform of the health index, not a probability).

**497. Predictive or descriptive?**
Descriptive.

**498. Synchronised continuously?**
No. Per run.

**499. Where is it stored?**
In memory.

**500. State representation?**
A snapshot dataclass (state, health index, RUL, policy, likelihood, time) and a bounded deque per battery.

**501. What happens when telemetry stops?**
It keeps the last state and reports itself stale after 10 minutes.

**502. How does it recover?**
The next run updates it. After a restart it's empty until runs happen.

**503. Can you rebuild it from historical telemetry?**
Yes, by replaying the logs in order.

**504. What makes it a twin rather than a digital model?**
Nothing strong. It's synchronised state without a model of the asset. Call it a state tracker.

---

## AD. Software engineering

**505. Why FastAPI?**
Python, so it calls the analysis library directly. Typed request/response models, automatic OpenAPI docs.

**506. Why Express?**
A gateway for the React client: one origin, CORS, and error mapping (502 when Python is down). It was part of the planned MERN deliverable, and it isn't strictly necessary.

**507. Why React?**
Interactive component-based dashboard; part of the planned stack.

**508. Why not FastAPI only?**
That would work. Express adds a layer, not a capability. Be honest about this.

**509. Why containerise?**
One command (`docker compose up`) runs the API, gateway and client with fixed versions. CI builds the image.

**510. What does CI test?**
- lint (ruff + mypy)
- the full Python suite on 3.10 and 3.12
- the suite without the optional CAN libraries
- the validation suite reproducing the benchmark study
- firmware compile for ESP32, ESP8266 and Arduino Uno
- the gateway tests
- the client build
- the Docker build

11 jobs, all passing on the last verified run (commit `a4e47d3`).

**511. Test coverage?**
89% of source lines, measured with pytest-cov over 948 tests.

**512. What does a unit test cover?**
One function's behaviour on a known input, e.g. a synthetic cell with known capacity and resistance: does the estimator recover them?

**513. Integration test?**
A whole path, e.g. a replayed serial capture through parse → score → result, or API endpoint tests.

**514. What if the backend is unavailable?**
The gateway returns 502 with a message. The client shows the error.

**515. How do errors reach the UI?**
HTTP status plus a JSON error from the gateway. Refusals aren't errors: they come back as data fields.

**516. Why are refusals values, not exceptions?**
See 168.

**517. Why ADRs?**
To record what was decided and why, so a reversal shows its reasoning. 14 exist.

**518. A decision you changed because of experimental evidence?**
Several:
- The SOH window approach replaced the fitted health model after LOCO showed no transfer.
- The 2 A fast-charge threshold became C-rate-based after it never fired on real data.
- A headline claim was withdrawn after one cell turned out to be carrying it.
- The RUL claim went from 88% to 73% after we found the reference used future data.

---

## AE. Research claims

**519. Original hypothesis?**
That user behaviour (temperature, fast charging, deep discharge, SOC extremes) measurably drives degradation, so a behaviour-based score could predict it and guide users.

**520. Did results support it?**
Only partly. Temperature was a real, correctly signed signal (7 of 7 NASA cells, within cohort), but its size didn't transfer between protocols. The current-based signals flipped sign between cohorts. The behaviour risk score didn't track measured fade (ρ = −0.27, p = 0.12, n = 33).

**521. Most surprising result?**
That the target first used, per-cycle capacity loss, was about 96% measurement noise: its maximum attainable R² was 0.044.

**522. What contradicted your expectations?**
The LSTM, given more context, transferred worst. A physically motivated Arrhenius model fitted a *negative* activation energy. And RUL errs early, not late as textbooks predict.

**523. What did you remove as not defensible?**
- The unvalidated ML stress-score model.
- Fitted models in the scoring path.
- The heuristic health index as the source of battery state, where measured SOH exists.
- Model rankings.

**524. What claims did you withdraw?**
- Six model-ranking claims.
- "LOBO selection picks the worst model" (it rested on one cell).
- 88% RUL (now 73%).
- The earlier 3.3% SOH (superseded by the field method).
- An early misdiagnosis that CS2_9 failed from loss of active material (it was truncated discharges).

**525. Why are negative results valuable?**
They stop others repeating the same mistake, and they show where the real difficulty is.

**526. Strongest result?**
SOH from voltage, current and time only: 1.7% median error on 17 real cells, tracked against every lab test (README charts). Plus the LOCO finding.

**527. Weakest?**
Partial-discharge SOH: real, but only 2 cells per band. And the hardware, which has never measured a discharge.

**528. Biggest limitation?**
Single cells of one chemistry, at constant current. Nothing on packs, other chemistries or real drive cycles.

**529. With six more months, what first?**
A bench discharge series on the rig: measure capacity and resistance repeatedly on one cell, then compare the field SOH against a direct capacity measurement on our own hardware.

**530. Which experiment would most improve confidence?**
Same as 529, then a second chemistry (LFP), where the flat voltage plateau is a real test of the window method.

---

## AF. Why should I believe you?

**531. How do I know the model isn't overfitting?**
The shipped estimators have no fitted cross-cell parameters to overfit. The ML models *do* overfit, and that's what LOCO showed and what we report.

**532. How do I know features don't leak the future?**
Trailing windows only. Splits by cell and cohort. Scalers fitted on training only. And a test that rewrites the future and checks the RUL estimate doesn't change.

**533. How do I know the hardware isn't producing plausible-looking numbers?**
Voltage readings are exact multiples of the INA219's 4 mV step, the cadence is exactly 1.000 s, and a real sensor fault was reported and refused. But without a reference meter, plausible is all we can show.

**534. How do I know the dashboard isn't hard-coded?**
Run the pipeline on any capture and the numbers change. The pipeline is tested to be a pure function of its input. The default dashboard runs on simulated data and says so.

**535. How do I know your SOH isn't a transformation of the ground truth?**
The estimator never reads the truth. It reads voltage, current and time. The cycler's capacity column is dropped before scoring (`health_report.py` keeps only those channels). Truth is joined afterwards, only to score.

**536. How do I know RUL isn't circular?**
Each estimate uses only history up to its own cycle. The test that tampers with the future checks this, and it's scored against the later observed crossing.

**537. How do I know you didn't pick the model that gave the nicest result?**
- No model was picked.
- The SOH window, arms and scoring were fixed before each study ran, and that's recorded in the script docstrings.
- Post-hoc choices are labelled "exploratory".
- Sensitivity windows are reported, not chosen between.

**538. How do I know NASA/CALCE differences don't dominate?**
We never pooled them. Each was analysed separately.

**539. How do I know it works outside the dataset?**
You don't, and we don't claim it. The method needs no cross-cell training, which is why it's expected to transfer better, but that expectation hasn't been tested on other chemistries or on field data.

**540. What evidence would make you reject your own model?**
For the SOH method:
- a bench discharge series on new cells where window SOH disagrees with direct capacity by more than about 5 points
- or a second chemistry where it fails without the gates refusing.

For RUL: near-end-of-life accuracy collapsing on a new dataset.

---

## AG. Questions from your two slides

**541. Can all three models produce both SOH and RUL?**
No. They were only evaluated for SOH. Fix the slide.

**542. Why does the diagram imply they're interchangeable?**
It's a drawing mistake.

**543. Why isn't LSTM listed for degradation modelling?**
It was evaluated on the same SOH target as XGBoost. The table labels are inconsistent; fix them.

**544. What are the LSTM's "time-series features"?**
The same 7 per-cycle features, over 10 consecutive cycles.

**545. GPR's "battery health features"?**
The same 7 features as XGBoost. The label is misleading.

**546. Why does the baseline use only age?**
To measure how much of the skill comes from knowing age alone.

**547. Why is the baseline strong?**
Ageing is dominated by cycle count.

**548. Why is the LSTM's LOCO R² negative?**
See 231.

**549. Why is GPR's LOCO MAE so high?**
See 230.

**550. Why does XGBoost have the best LOCO R²?**
It has the highest point estimate, likely because shallow boosted trees with a small learning rate are fairly robust. But its interval overlaps every other method's.

**551. Is it statistically better?**
No.

**552. With overlapping intervals, can you call it the best?**
No.

**553. Then why is XGBoost shown prominently?**
It shouldn't be shown as the winner. Present it as "highest point estimate, not distinguishable from the others".

**554. Why LOBO MAE but not LOBO R²?**
Slide omission. Both exist.

**555. Why is LOCO more important than LOBO?**
Deployment means new conditions. LOCO tests that; LOBO doesn't.

**556. What changes between LOBO and LOCO?**
Whether cells from the test cell's protocol are in training.

**557. Does the model see the same cohort in training?**
Under LOBO yes; under LOCO no.

**558. What if an entire data source is held out?**
Not run. See 33.

**559. Can you show the fold composition?**
Yes. Per-fold results are in `reports/metrics/benchmark_*`, and the cohort list is in the NASA dataset spec. Have one example open.

**560. Batteries per fold?**
LOBO: 1 test cell, 30 training. LOCO: one cohort out of 9, roughly 3–4 cells each.

**561. Samples per fold?**
LOBO median: 2,518 training rows, 67 test rows. LOCO varies by cohort.

**562. Are all batteries equally represented?**
No. Cells have different cycle counts.

**563. Could a large battery dominate MAE?**
Within a fold, yes, since MAE is over rows. Across folds we take the median per fold, so one long cell can't dominate the summary.

**564. Are your intervals per sample or per battery?**
Per fold (cell or cohort), never per sample. Samples within a cell aren't independent, so a per-sample interval would be falsely narrow.

---

## AH. Really nasty questions

**565. Why isn't voltage alone enough for SOH?**
See 15.

**566. Why can two batteries at the same SOC show different terminal voltages?**
Different current (I × R), temperature, resistance (age), recent history (polarisation, hysteresis), and different cell designs.

**567. Why does temperature affect voltage?**
Open-circuit voltage has a temperature coefficient (entropy), and resistance falls as temperature rises, so the drop under load changes.

**568. Why does current affect terminal voltage?**
I × R ohmic drop plus polarisation.

**569. OCV vs terminal voltage?**
OCV is the voltage at rest after relaxing: the thermodynamic value for that SOC. Terminal voltage is what you measure under load: OCV minus the losses.

**570. Why does internal resistance increase with ageing?**
See 281.

**571. Why does capacity decrease?**
Lithium is lost to side reactions (SEI, plating) and active material is lost (particle cracking, loss of contact).

**572. Why isn't degradation linear forever?**
Mechanisms change over life: fast early SEI formation, then slower growth, then sometimes a knee when plating or electrode damage takes over.

**573. Capacity fade vs resistance growth?**
Fade: less charge stored. Resistance growth: more voltage lost under load (power fade). They're related but not the same.

**574. Which does your model predict?**
It measures capacity fade (SOH) and reports resistance growth (power fade) measured from the load step. That resistance tracked the cycler's own on 19 of 20 CALCE cells.

**575. Can you identify the electrochemical mechanism?**
No.

**576. Then why call the output "degradation"?**
Because capacity loss *is* degradation, observed. We don't claim to know its cause.

**577. Can you tell calendar ageing from cycle ageing?**
No.

**578. How do you stop yourself claiming that?**
The outputs don't mention mechanisms. The report card states the evidence limits.

**579. Brand-new battery from another maker tomorrow — will it work?**
The SOH method was tested on exactly this case: a different manufacturer (Kokam), format (pouch) and chemistry (NMC/LCO), at 40 °C, with nothing re-tuned. It measured all 8 cells at 3.0% median error. It still needs that cell's own first discharges as its reference, and LFP remains untested.

**580. If yes, what evidence?**
The Oxford external validation (`reports/metrics/oxford/`): 8 of 8 cells at 3.0%, no parameter changed from the CALCE setup.

**581. If no, what's the practical value?**
It refuses rather than misleads, and it becomes useful once that cell has enough of its own history.

**582. Minimum data before an SOH estimate?**
It's decided per output, not by one number. Capacity health needs voltage, current and time, then at least 6 discharges across the window: 5 form the as-new reference and 1 is measured against it. That 6 is a hard minimum by construction. Confidence then follows the readings' **consistency**, not their count. Measured on CALCE (`calce_sufficiency/`): when the readings agree to within 1 point (standard error, after removing the ageing trend), median error was 1.3%; when they didn't, 5.1%. The number of discharges did *not* predict accuracy, because error grows with age. So BEACON reports LOW confidence and a wide band until the readings are consistent, however many discharges it has. RUL needs 30 complete discharges, and is refused for partial-only logs (Q54).

**583. What happens in the first few cycles?**
SOH: "not available — only N discharges crossed the window; 5 are needed". RUL: "fewer than 30 cycles of history".

**584. Genuine ageing vs different operating conditions?**
- Partly by design: rate is corrected (I × R), and cut-short discharges are excluded.
- Temperature isn't corrected.
- A cell measured cold will read lower SOH, and the system can't currently tell that from ageing.

**585. Your model says 92%, the capacity measurement says 85%. Which do you trust?**
The capacity measurement.

**586. Why?**
SOH is *defined* as capacity relative to new. A direct capacity test is the definition; any estimate is an approximation of it.

---

## AI. ML questions that destroy weak defences

**587. Trainable parameters in the LSTM?**
About 5,300: LSTM layer 4 × 32 × (7 + 32) + 8 × 32 = 5,248, plus the output layer's 33.

**588. Why that architecture?**
Deliberately small, for 2,360 windows. A big network would just memorise cells, and the LOBO/LOCO gap would then say more about model size than about protocol shift.

**589. Training set size?**
NASA SOH benchmark: 2,585 rows from 31 cells in total. LOBO trains on about 2,518 rows per fold.

**590. Validation set size?**
LSTM early stopping: 20% of the training cells. Other models have no separate validation set, since hyperparameters are fixed. ElasticNet uses internal 5-fold CV on training data.

**591. Test set size?**
One cell (LOBO, median 67 rows) or one cohort (LOCO).

**592. How were hyperparameters chosen?**
Fixed, conventional defaults (documented in `benchmarks/classical.py`), deliberately not tuned.

**593. Did you tune on the test set?**
No.

**594. Did you tune per fold?**
No, except ElasticNet's penalty, chosen by CV inside each training fold.

**595. What if you shuffled the time series?**
Run (`scripts/run_ablation_study.py`): with each cell's capacity shuffled in time, every LOCO interval lies at or below zero (XGBoost −1.44 to −0.01; age-linear −0.35 to −0.00). The harness does not create skill where there is none. XGBoost keeps LOBO R² 0.146 on the shuffled data: each cell's average capacity survives the shuffle (per-cell memorisation).

**596. What if you removed the engineered features?**
Run: XGBoost on behaviour features without cycle number scores LOBO 0.576, LOCO −0.293. On cycle number alone: LOBO 0.047, LOCO −0.266. Only the combination reaches 0.459. Both halves alone fall below zero under LOCO (point estimate).

**597. XGBoost on raw voltage, current and temperature only?**
Not possible at cycle level: the NASA benchmark table holds per-cycle aggregates, not raw samples. The nearest test is in 596: behaviour aggregates without age, LOCO −0.293.

**598. How much does feature engineering add?**
About 0.05 LOCO R² over a linear age baseline (0.459 vs 0.406), within the intervals. Ablations in 596 show neither features nor age alone transfer. Curve features (ΔQ(V), ICA) improved 1 of 4 methods, none beyond fold noise.

**599. Which feature matters most?**
Cycle number (age). Among behaviour features, temperature was the only one with a consistent, significant within-cohort relationship to fade.

**600. How did you determine importance?**
Regression coefficients with cohort fixed effects, and the age-only baseline comparison. A SHAP analysis was gated on out-of-sample skill, failed that gate, and so isn't used to explain degradation.

**601. Does importance mean causation?**
No.

**602. Can SHAP say why a battery physically degraded?**
No. SHAP explains the model's output in terms of its inputs. If the model is wrong or correlational, SHAP explains the error faithfully.

**603. Correlation vs causation?**
Correlation: two things move together. Causation: changing one changes the other. NASA's temperature correlation is confounded with protocol, so it can't establish cause on its own. That's exactly why its size didn't transfer.

---

*Generated against commit `9674e93`. If a figure here disagrees with a file
under `reports/metrics/`, the file is right.*
