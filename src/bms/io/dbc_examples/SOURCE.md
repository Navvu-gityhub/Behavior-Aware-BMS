# Source and provenance

`twizy_bms_1.dbc` is transcribed verbatim from the Open Vehicles (OVMS)
project's public documentation:

  https://docs.openvehicles.com/en/latest/components/vehicle_dbc/docs/dbc-primer.html

OVMS is an open-source telematics/monitoring project supporting 30+
production EVs (Tesla, Nissan Leaf, BMW i3, Hyundai/Kia models, VW e-Golf,
Renault Zoe, and others) via reverse-engineered CAN decoding. This
particular message (CAN ID 0x155 / 341, "BMS_1") is documented there as
the Renault Twizy's primary BMS status frame: charge current limit,
momentary battery current, and state of charge.

OVMS's own docs note the Twizy's actual production driver doesn't use the
generic DBC engine internally (it has a hand-written C++ decoder instead)
-- this DBC file is presented there specifically as a worked *example* of
how you'd express that same decoding in DBC form. It is real, documented,
correct vehicle protocol data, not a hypothetical.

## What this does and doesn't prove

Decoding this one message type, correctly, against OVMS's own stated
expected output, demonstrates the CAN/DBC adapter pattern works end to
end against real vehicle data (see `src/bms/io/load_can_dbc.py` and
`tests/test_can_dbc.py`).

It does **not** demonstrate broad multi-OEM support. Each vehicle's real
DBC/decoding logic is reverse-engineered individually (this is exactly
why IEEE DataPort's public EV CAN dataset ships raw, undecoded messages --
manufacturer DBC files are proprietary and not publicly available). A
production integration would need either an OEM data-sharing relationship
or the kind of vehicle-by-vehicle reverse engineering OVMS's contributors
have done for the vehicles it supports.

It also only covers current and SoC -- this one message doesn't expose
voltage or temperature, which the unified schema (`src/bms/preprocessing/
schema.py`) requires. See `docs/can_dbc_adapter.md` for what that means
in practice and how the pipeline honestly handles it (it doesn't silently
proceed with fabricated values -- see the test suite).
