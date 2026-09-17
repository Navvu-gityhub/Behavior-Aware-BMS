# Security audit

Audited 2026-09-16 against commit `d678722` plus the working tree. This records
what was checked, what was found, what was fixed and — equally — what this
project deliberately does not defend against.

**Threat model.** BEACON is a local bench and analysis tool. The API binds
`127.0.0.1` by default, has no authentication, no session, no cookie and no
user accounts, and stores its fleet in memory. It is not built to be exposed to
an untrusted network, and nothing below should be read as claiming it is
hardened for that. The audit's question was narrower and answerable: *can a
caller who reaches this service make it do something it should not?*

---

## Findings

### 1. Arbitrary file read via request-supplied paths — **FIXED**

**Severity: the only real finding.**

Three endpoints take a filesystem path from the request body and open it:

| Endpoint | Field |
|---|---|
| `POST /telemetry/replay` | `log_path` |
| `POST /telemetry/serial/replay` | `capture_path` |
| `GET /telemetry/coverage` and the replay endpoints | `dbc_path` |

There was no containment. A caller could pass `~/.ssh/id_rsa`, `/etc/passwd` or
a `.env` two directories up. Contents are not echoed back wholesale, but the
serial path surfaces per-line **rejection reasons** in its decode statistics,
and those can carry fragments of whatever was read. Separately, the difference
between a 404 and a parse error answered *"does this file exist?"* for any path
the process could reach.

**Fix:** `src/bms/api/paths.py`. The resolved path must sit under an allowed
root; otherwise a 400 is returned **before** the existence check, so an
out-of-bounds path cannot be used to probe the filesystem.

A root allowlist, not a `..` blocklist — deliberately. Blocklists on path
strings lose to symlinks, UNC paths, percent-encoding and Windows 8.3 short
names. Resolving first and asking *where did this end up* is decided after any
such trick has already been applied.

Defaults admit the repository's `data/`, `reports/`, `tests/` and
`src/bms/io/dbc_examples/`, plus the system temp directory — because recording
a capture to a temp file and replaying it is the normal bench workflow. Override
for any real deployment:

```bash
BEACON_ALLOWED_DATA_ROOTS=/srv/beacon/data:/srv/beacon/captures
```

The variable **replaces** the defaults rather than extending them, so setting it
on a host cannot accidentally leave the development convenience in place.
Covered by `tests/test_api_path_containment.py` (13 tests).

### 2. CORS was unconditionally permissive — **MADE CONFIGURABLE**

`app.use(cors())` allowed every origin. Now `cors({ origin: config.corsOrigin })`
with `CORS_ORIGIN`, defaulting to `*`.

The default stays permissive because the default deployment is a laptop where
the Vite dev server runs on a different port. **This is not a security boundary
here** and is not presented as one: the gateway exposes no authenticated
endpoint and sets no cookie, so a permissive origin grants a caller nothing they
could not obtain by calling the gateway directly. It is worth tightening on a
shared host; it would be theatre to call it a fix.

### 3. No secrets, and no LLM integration — **VERIFIED ABSENT**

`git grep` across all `.py`, `.js`, `.jsx`, `.yml`, `.toml`, `Dockerfile` and
`docker-compose.yml` for `api_key`, `secret`, `bearer`, `openai`, `anthropic`,
`gemini`, `generativeai`, `langchain`: **zero matches.**

The commit history contains *"Day 20: add battery digital twin and Gemini expert
agent"*, so this was checked rather than assumed. **No LLM component survives in
the tree.** The only surviving trace is a planning line in a notebook markdown
cell (`notebooks/09_digital_twin.ipynb`: "Integrate Gemini Expert Agent").
`src/bms/guardian/` is deterministic rule-based reporting — `generate_guardian_reports`
makes no network call and imports no HTTP client.

Accordingly, **no LLM-specific hardening was added.** Adding prompt-injection
defences to a system with no prompt would be security theatre.

### 4. Request validation — **ADEQUATE, unchanged**

FastAPI + Pydantic validate every request body. Bounds already present on the
fields that matter:

| Field | Bound |
|---|---|
| `duration_s` (CAN live, serial live) | `> 0`, `≤ 300` |
| `n_cycles` (emulator) | `1 … 50` |
| `sample_period_s` | `> 0`, `≤ 3600` |
| `min_accepted_fraction` | `0 … 1` |
| `corrupt_every` | `≥ 0` |

The duration caps matter most: an unbounded capture inside a request handler
never returns. No endpoint accepts a raw list of telemetry lines, so there is no
unbounded-payload ingest vector. `express.json()` retains its 100 kB default on
the gateway.

Two fields are unbounded above — `baudrate` (`> 0`) and `corrupt_every`. Both
only affect a local serial device, and neither is worth a cap that would have to
be guessed.

### 5. Serial parser under adversarial input — **ALREADY SAFE**

The wire parser was written for a lossy transport, which turns out to be most of
what adversarial-input handling needs:

- lines without the sentinel are ignored as noise, not errors — a board's boot
  banner is normal traffic;
- a malformed line raises `LineDecodeError`, is **counted**, and parsing
  continues; nothing aborts the capture;
- field values are validated against declared ranges, and **rejected rather than
  clamped** — a clamped value looks measured and is not;
- NaN and infinity are rejected explicitly, because a NumPy comparison against
  NaN is `False` and would read as "not hot";
- decoding uses `errors="replace"`, so a corrupt byte becomes one rejected line
  rather than an exception;
- below a configurable accepted fraction the whole capture is **refused** rather
  than scored from whatever survived.

Exercised by `tests/test_serial_telemetry.py` and
`tests/test_hardware_readiness.py`, including truncated lines, corrupted
checksums, non-finite values, out-of-range fields and unknown record types.

### 6. Notebook metadata contains the author's Google account — **NOTED**

`notebooks/*.ipynb` carry Colab execution metadata including a display name and
numeric `userId`. Not a credential and not exploitable; noted because the
repository is public and the author may not realise it is there. Stripping it
means re-saving the notebooks with cleared outputs — a decision for the author,
not an audit finding.

---

## Not defended against, by design

- **No authentication or authorisation.** Anyone who reaches the API can use
  every endpoint. Do not expose it without putting something in front.
- **No rate limiting.** A caller can start as many bounded captures as they like.
- **No transport security.** Plain HTTP; terminate TLS upstream if needed.
- **The in-memory fleet store is global.** No tenancy, no isolation.

These are consequences of the tool's scope, not oversights, and the correct fix
for all four is an authenticating reverse proxy rather than code in this
repository.

---

## CI

CI depends on no API key, no physical hardware and no external service beyond
package registries and the Arduino board-core index. The `firmware` job
downloads board cores; everything else runs offline after `pip install`. The
validation job restores any tracked artifact a stage rewrites, so a CI run
cannot quietly commit a changed result.
