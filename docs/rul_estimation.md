
# Remaining Useful Life Estimation

Method:
Equivalent Aging Factor (EAF)

Inputs:
- Health Index
- Temperature
- Deep discharge
- Fast charging

Formula:

EAF =
0.40*HI +
0.25*T +
0.20*DD +
0.15*FC

Estimated Cycle Life:
TotalCycles = 1000/EAF

Remaining Useful Life:
RUL = TotalCycles × RemainingHealth

Replacement Policy:
<100 cycles        : REPLACE
100-300 cycles     : PLAN_SERVICE
300-600 cycles     : MONITOR
>600 cycles        : NORMAL

Advantages:
- Explainable
- Embedded implementation
- Industrially deployable
- Digital Twin compatible
