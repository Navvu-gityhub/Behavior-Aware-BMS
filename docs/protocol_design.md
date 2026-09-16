# Protocol design and analysis

BEACON is a telemetry system, and telemetry is a communication problem. This
document analyses the system as one: the three protocol stacks it spans, the
design of the custom application-layer protocol at its centre, its error
detection capability, and its measured throughput and channel utilisation.

Companion documents: `hardware_design.md` (the instrumentation), `architecture.md`
(the software system), `hardware_integration.md` (the bring-up procedure).

> Every throughput figure below is **measured**, not estimated — produced by
> encoding real frames through `src/bms/telemetry/serial_schema.py` and counting
> bytes on the wire. The error-detection probabilities are analytical, under the
> stated error model.

---

## 1. Three stacks, one schema

The system spans three distinct communication domains, and its central design
decision is that all three converge on one representation before any analysis
occurs.

| | **CAN** | **BEACON serial** | **HTTP/REST** |
|---|---|---|---|
| Domain | In-vehicle network | Bench instrumentation | Service tier |
| Physical | Differential pair, twisted | USB CDC over UART | Ethernet / loopback |
| Access | CSMA/CR, non-destructive arbitration by identifier | Point-to-point simplex | Switched, full duplex |
| Framing | 11/29-bit ID + ≤8 B payload | Sentinel + newline delimited | TCP stream + HTTP headers |
| Error control | 15-bit CRC, bit stuffing, automatic retransmission | 8-bit LRC, no retransmission | TCP checksum + ARQ |
| Schema authority | DBC file | `HELLO` handshake | OpenAPI |
| Direction | Broadcast, multi-master | Simplex, device → host | Request/response |

The convergence point is the **unified telemetry frame** — six columns
(`cell_id`, `test_time_s`, `voltage_v`, `current_a`, `temperature_c`, `soc`) —
and one scoring function, `score_telemetry_frame`, consumes it regardless of
which stack delivered it.

That is the architectural claim worth defending: three protocols with entirely
different error models, framing and access disciplines, reduced to one
representation early enough that everything downstream is transport-agnostic. A
disagreement between a CAN run and a serial run over the same cell is therefore
a difference in the *data*, never in the code path.

---

## 2. The BEACON serial protocol

### 2.1 Layer decomposition

The protocol is deliberately thin. Mapping it onto the OSI model shows which
layers are present, which are absent, and — more usefully — *why* the absent
ones were not needed.

| Layer | Provision | Notes |
|---|---|---|
| 7 Application | Telemetry semantics: five channels, units, sign conventions | `FIELDS` in `serial_schema.py` is the authority |
| 6 Presentation | Two codecs: JSON and compact `key=value` | Auto-detected by first character; both decode to one record |
| 5 Session | `HELLO` schema announcement | Declares fields *before* streaming, so coverage is checked before decoding |
| 4 Transport | **None** | No segmentation, no ARQ, no flow control — §2.5 |
| 3 Network | **None** | Point-to-point; no addressing needed |
| 2 Data link | Sentinel framing, newline delimiting, 8-bit LRC | §2.2, §2.3 |
| 1 Physical | UART 115200 8N1 over USB CDC | §3 |

### 2.2 Frame format

```
BEACON1 D {"t":12.5,"v":3.91,"i":-1.48,"tc":27.4,"soc":82.1}*06\n
└──┬───┘ │ └──────────────────── body ──────────────────────┘└┬┘└┬┘
sentinel kind                                            checksum  delimiter
```

| Field | Size | Function |
|---|---|---|
| Sentinel `BEACON1` | 7 B | Frame synchronisation |
| Separator | 1 B | — |
| Kind `HELLO`/`D`/`S` | 1–5 B | Record type |
| Separator | 1 B | — |
| Body | variable | Payload |
| Checksum `*HH` | 3 B | Error detection (optional) |
| Delimiter `\n` | 1 B | Frame boundary |

**Framing is by sentinel, not by length.** The alternative — a length prefix —
was rejected because the channel carries uninvited traffic: every
microcontroller prints a ROM boot banner, bootloader chatter, and whatever
`Serial.print` debugging the developer left behind. A length-prefixed parser
resynchronises after such noise only by luck. A sentinel parser discards
anything not beginning with `BEACON1` as noise, counts it, and continues. This
makes the protocol **self-synchronising**: it recovers at the next newline after
any corruption, with a maximum resynchronisation loss of one frame.

### 2.3 Error detection: what the LRC does and does not catch

The checksum is an 8-bit XOR over every byte of the body — a longitudinal
redundancy check, rendered as two hex characters, NMEA-0183 style.

**Detected:**
- All single-bit errors.
- Any **odd** number of bit errors falling in the same bit position across bytes.
- All errors confined to a single byte that change an odd number of bits.

**Not detected — and these are the ones that matter:**

1. **An even number of bit errors in the same bit position.** Bit 3 flipped in
   two different bytes cancels exactly. This is the classic LRC weakness.
2. **Byte transposition — entirely.** XOR is commutative, so *any* reordering of
   the body's bytes produces an identical checksum. A UART re-synchronising
   mid-frame can produce exactly this.
3. **Corruption of the sentinel or the kind field.** The checksum covers **only
   the body** (`_frame()` computes `xor_checksum(body)`), so an error in
   `BEACON1` or in the record-type character is outside its scope. Such a frame
   is caught by the parser's type dispatch — an unknown kind raises — but not by
   the checksum, and a `D` corrupted into an `S` would be silently reclassified
   as a status record rather than rejected.

Under a random-corruption model where a damaged frame's checksum is uniformly
distributed, the undetected-error probability is

```
P(undetected) = 2⁻⁸ = 0.0039  →  0.39% of corrupted frames pass
```

### 2.4 Recommendation: CRC-8 is strictly better at identical cost

**CRC-8 occupies the same 8 bits and dominates LRC on every axis.** It detects
all single- and double-bit errors, all odd numbers of bit errors, and — the
decisive property — **all burst errors up to 8 bits**, which is the error
pattern a noisy UART actually produces. It is also not commutative, so it
detects the byte transposition of §2.3(2) that LRC cannot see at all.

| Scheme | Overhead | P(undetected) | Bursts ≤ 8 bits | Transposition |
|---|---|---|---|---|
| XOR / LRC-8 (current) | 2 B (hex) | 0.39% | Partial | **Never detected** |
| CRC-8 | 2 B (hex) | 0.39% | **All** | Detected |
| CRC-16 | 4 B (hex) | 0.0015% | All ≤ 16 bits | Detected |

The current choice is defensible on the grounds it was made on — three lines of
C on any microcontroller, and legible in a serial monitor during bring-up — and
LRC's real-world detection rate on single-byte corruption is high. But CRC-8 is
a table-driven loop of comparable size, and the cost of the change is one
function on each side. **This is the protocol's clearest improvement, and it is
free in bandwidth.**

CRC-16 would be the choice if the link were long or electrically noisy. Over a
30 cm USB lead it is over-provisioned, and 2 extra bytes per frame at 1 Hz is
irrelevant either way (§3).

### 2.5 Why there is no ARQ, and why that is correct

The protocol has no acknowledgements, no retransmission, no flow control. It is
simplex. This is a deliberate choice against the TCP-style default, and the
justification is a property of telemetry rather than of this implementation:

**A lost sample is preferable to a delayed one.** Retransmission would deliver
sample *n* after sample *n+1*, and the pipeline refuses on non-monotonic time
because coulomb counting integrates over it — so out-of-order delivery is worse
than loss. Buffering to restore order would introduce unbounded latency in a
system whose whole purpose is to observe a physical process in real time.

**Loss is tolerable because the signal is oversampled.** Cell temperature and
capacity move on timescales of minutes; sampling is at 1 Hz. Losing an
occasional sample costs nothing measurable.

**Loss is not tolerable in bulk, and that is handled at the application layer.**
Rather than per-frame retransmission, integrity is enforced *aggregate*:
`SerialDecodeStats` counts every discarded line with its reason, and
`run_serial_pipeline` refuses to score a capture whose accepted fraction falls
below 50%. The reasoning is that a capture losing more than half its frames does
not have noise in it — it has a wrong baud rate, a half-connected cable, or
mismatched firmware. Retransmission would mask that as slow-but-working; the
threshold surfaces it as a fault.

This is the protocol's most interesting design decision: **it substitutes an
end-to-end statistical integrity check for a per-frame reliability mechanism**,
which is appropriate precisely because the application tolerates sparse loss and
cannot tolerate reordering.

---

## 3. Throughput and channel utilisation

Measured at 115200 baud, 8N1 (10 bits per byte including start and stop).

### 3.1 Frame sizes, measured

| Frame | Size (incl. `\n`) |
|---|---:|
| `HELLO` (full, with capacity) | 195 B |
| Data, JSON codec — mean over a session | 72.4 B |
| Data, JSON codec — min / max | 68 / 74 B |
| Data, compact codec | 52 B |
| Status | 25 B |

### 3.2 Derived link performance

| Metric | JSON | Compact |
|---|---:|---:|
| Bits on the wire per frame | 724 | 520 |
| Frame transmission time | 6.28 ms | 4.51 ms |
| **Maximum sustainable rate** | **159 frames/s** | **222 frames/s** |
| Utilisation at 1 Hz (the design point) | 0.63% | 0.45% |
| Utilisation at 10 Hz | 6.3% | 4.5% |

**The link is over-provisioned by more than two orders of magnitude at the
design sampling rate.** At 1 Hz the channel is 0.63% utilised, so sampling could
rise to roughly 150 Hz before the UART becomes the constraint. The binding
limit on sample rate is not bandwidth but the DS18B20's 750 ms conversion time
(`hardware_design.md` §5.2) — a *sensor* constraint, not a *channel* constraint.

This is why the verbose codec is the default. Its 39% bandwidth premium buys
human-legible frames in a serial monitor during bring-up, and it is spending a
resource the system has in vast surplus.

### 3.3 Encoding efficiency

For the JSON codec:

```
framing + checksum overhead : 10 B of 63 B  →  15.9%
five float channels as text : ~30 B
same five as IEEE-754 binary:  20 B
```

A binary encoding would cut the frame roughly in half. It was rejected for the
same reason as the verbose codec: legibility during bring-up is worth more than
bandwidth that is not scarce. **Efficiency is only worth optimising where the
resource is constrained**, and here it is not.

---

## 4. The CAN path

The vehicle-facing stack is a standard automotive network, and the project
implements the decoding half of it.

| Property | Value |
|---|---|
| Access | CSMA/CR — non-destructive bitwise arbitration on the identifier |
| Priority | Lower identifier wins; arbitration loser retries without collision loss |
| Payload | ≤ 8 B per classic frame |
| Error detection | 15-bit CRC, bit stuffing, form and acknowledgement checks |
| Reliability | Automatic retransmission on error, with fault confinement |
| Schema | DBC file mapping signals to bit offsets, scale and offset |

The contrast with the serial path is instructive and worth an explicit paragraph
in any report: **CAN provides in the physical and data-link layers exactly what
BEACON's serial protocol declines to provide** — retransmission, strong CRC,
multi-master arbitration — because CAN carries control traffic where a lost
frame can be a safety event, whereas the bench rig carries observations where a
lost sample is a rounding error.

The schema problem is the same in both, solved differently: a DBC is a static
file the host loads, while `HELLO` is a runtime announcement the device sends.
The serial approach is more flexible (a rig can change its field set without
redistributing a file) and weaker (nothing verifies the declaration against an
external authority). `sources.check_signal_coverage` and the serial coverage
gate are the same function applied to the two schema sources.

---

## 5. The service tier

| | |
|---|---|
| API | FastAPI, `:8000`, JSON over HTTP/1.1 |
| Gateway | Express, `:5000`, aggregation and shaping |
| Client | React/Vite, `:5173` |
| Health | `/healthz`, polled by the container `HEALTHCHECK` on a 30 s interval |
| Dependency | Gateway waits on `service_healthy`, not merely `service_started` |

The `depends_on: condition: service_healthy` ordering is a genuine distributed
systems point: container start order does not imply readiness, and a gateway
that begins proxying before the API can serve produces connection-refused errors
that look like network faults rather than a race.

**Not measured.** No latency, throughput or concurrency figures exist for this
tier — there is no load test. Any claim about its performance would be
unsupported, and §6 lists this as the principal gap.

---

## 6. Gaps

Stated rather than glossed, in descending order of how much they would
strengthen an evaluation:

1. **No measured link error rate.** The 0.39% undetected-error figure of §2.3 is
   analytical, under a uniform random-corruption model. No bit-error rate has
   been measured on a physical UART, because no physical rig has been connected.
   The emulator's `--corrupt-every` exercises the *rejection counters*, not the
   channel.
2. **No service-tier performance measurement.** No latency distribution, no
   throughput ceiling, no behaviour under concurrent clients.
3. **CRC-8 not implemented** (§2.4) — a strict improvement at zero bandwidth
   cost, currently a recommendation rather than a change.
4. **The checksum does not cover the sentinel or kind field** (§2.3(3)).
   Extending it to cover the full line after the sentinel would close this at no
   additional overhead.
5. **No formal verification of the parser** against a fuzzed input corpus.
   Property-based testing over arbitrary byte strings would be a natural
   addition, and the parser's contract — never raise, always classify — is
   exactly the kind of invariant that suits it.

---

## 7. Reproducing the figures

Frame sizes and derived link performance in §3 come from encoding real frames
through the protocol implementation:

```python
from src.bms.telemetry import encode_record, encode_hello, DEFAULT_BAUDRATE

line = encode_record({"t": 12.5, "v": 3.91, "i": -1.48,
                      "tc": 27.4, "soc": 82.1}, compact=False)
frame_bytes = len(line.encode()) + 1        # + newline
bits = frame_bytes * 10                      # 8N1
seconds = bits / DEFAULT_BAUDRATE
```

Session means are taken over the 244 data records of a two-cycle emulated run at
a 120 s sample period.
