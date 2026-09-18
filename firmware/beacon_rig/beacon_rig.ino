/*
 * BEACON serial telemetry rig - reference firmware
 * ================================================
 *
 * Emits the BEACON serial telemetry schema (beacon.telemetry.v1) over USB
 * serial, so a microcontroller rig can drive the same BEACON analysis pipeline
 * that CAN telemetry drives.
 *
 * This sketch is deliberately BOARD-AGNOSTIC and SENSOR-AGNOSTIC. It compiles
 * on ESP32, ESP8266, and AVR (Uno/Nano/Mega) as written, and it does not name
 * any particular current sensor, thermistor or ADC. The four read functions at
 * the bottom are the only place hardware appears, and they ship as clearly
 * marked stubs that emit a synthetic profile. Replace them one at a time as
 * sensors arrive; everything above them stays untouched.
 *
 *
 * SAFETY - READ BEFORE CONNECTING A CELL
 * --------------------------------------
 * The microcontroller is a MONITOR. It is never the protection device.
 *
 *   1. Put a real protection board (TP4056+DW01, or a dedicated BMS module) in
 *      series with the cell. Over-discharge, over-charge and short-circuit
 *      protection must exist in hardware, independently of this firmware. If
 *      this sketch crashes, the cell must still be safe.
 *   2. ESP32 and ESP8266 GPIO are 3.3 V and NOT 5 V tolerant. Never connect
 *      pack voltage to a pin directly. Use a divider sized so the pin sees at
 *      most 3.0 V at the pack's maximum voltage, and check it with a multimeter
 *      before connecting the pin.
 *   3. Measure current through a purpose-built sensor - an INA219/INA226 (I2C,
 *      high-side, <=26 V) or an ACS712 (hall-effect, galvanically isolated).
 *      Do not build a shunt-plus-divider by hand for a first rig.
 *   4. Lithium cells fail dangerously when shorted, over-discharged or
 *      over-charged. Use one protected cell, not a series pack, for a first
 *      build. Charge on a bench, attended, on a non-flammable surface.
 *   5. Keep WiFi/BLE off for USB-serial operation (the default below). Their
 *      transmit current spikes can brown out a weakly-supplied board, which
 *      shows up as a mid-line reset - see the note on time resets below.
 *
 *
 * THE TWO CONVENTIONS THAT ACTUALLY BREAK INTEGRATIONS
 * ----------------------------------------------------
 * Both produce perfectly valid-looking output that means the wrong thing, and
 * neither is catchable by a range check. The host refuses on both, but it is
 * far cheaper to get them right here.
 *
 *   CURRENT SIGN: negative is DISCHARGE, positive is CHARGE.
 *     A rig with the opposite convention streams flawlessly and yields no
 *     discharge phases, so no capacity, so no state of health - while looking
 *     like it is working. If the host reports "sign convention is inverted",
 *     swap the shunt leads or negate the reading in readCurrentA().
 *
 *   SOC SCALE: PERCENT, 0-100. Not a 0-1 fraction.
 *     The host's feature layer compares SOC against 20.0 and 90.0. A 0-1 rig
 *     would flag deep discharge on every sample. The host detects this and
 *     refuses; fix it here by multiplying by 100, because the unit belongs to
 *     the device that measured it.
 *
 *
 * WIRE FORMAT
 * -----------
 * Every line begins with the sentinel "BEACON1", which lets the host ignore
 * boot banners, bootloader chatter and any Serial.print debugging you leave in.
 * You do not need to suppress those.
 *
 *   BEACON1 HELLO {"schema":"beacon.telemetry.v1","fields":[...],...}*4A
 *   BEACON1 S <free text status>*3C
 *   BEACON1 D {"t":12.5,"v":3.91,"i":-1.48,"tc":27.4,"soc":82.1}*7F
 *
 * The trailing *HH is an XOR of every byte of the record body - the same scheme
 * NMEA uses. It is optional; set EMIT_CHECKSUM to 0 on a very small board. It
 * catches the truncation and single-byte corruption that USB serial actually
 * produces, and costs three lines of C.
 *
 * A compact form is also accepted, for boards without room for JSON:
 *
 *   BEACON1 D t=12.5 v=3.91 i=-1.48 tc=27.4 soc=82.1*2B
 *
 * Set EMIT_COMPACT to 1 to use it. Both forms decode to the same record; the
 * host's test suite asserts that rather than assuming it.
 *
 *
 * CONNECTING THIS TO THE HOST
 * ---------------------------
 *   1. Flash this sketch. Open a serial monitor at SERIAL_BAUD and confirm you
 *      see BEACON1 lines. Nothing else needs to be true yet.
 *   2. Close the serial monitor. Only one program can hold the port.
 *   3. On the host:
 *
 *        from src.bms.telemetry import SerialPortSource, run_serial_pipeline
 *        source = SerialPortSource(
 *            name="bench_rig", port="COM5", baudrate=115200, duration_s=120,
 *        )
 *        print(run_serial_pipeline(source).render())
 *
 *      There is no default port on the host, by design: auto-picking the first
 *      enumerated device silently reads the wrong one on a laptop with a
 *      Bluetooth serial port. `available_ports()` lists candidates.
 *
 * See docs/hardware_integration.md for the full field table and wiring notes.
 */

/* ===================================================================== */
/* Configuration - the only section you should need to edit.             */
/* ===================================================================== */

/* Must match the host. 115200 is the project default, and what
 * src/bms/telemetry/serial_source.py and the docs assume.
 *
 * Overridable at build time, so a board whose USB bridge will not sustain the
 * default needs no edit to this file:
 *
 *   arduino-cli compile --fqbn esp8266:esp8266:nodemcuv2
 *     --build-property compiler.cpp.extra_flags=-DSERIAL_BAUD=9600
 *     firmware/beacon_rig
 *
 * Pass the matching --baudrate to the host. Keeping the repository default at
 * 115200 matters: baking one board's bridge fault into the shared default
 * would leave the firmware and the host disagreeing for everyone else.
 *
 * 9600 costs nothing at this sampling rate: one JSON record is ~72 bytes, or
 * 724 bits on the wire at 8N1, so 9600 carries ~13 records per second against
 * a SAMPLE_PERIOD_MS of 1000 - about 7% channel utilisation. The binding limit
 * on sample rate is the sensor, not the link. */
#ifndef SERIAL_BAUD
#define SERIAL_BAUD      115200
#endif

/* Real sensors, or the synthetic profile?
 *
 * 0 keeps the shipped behaviour: a synthetic profile so the sketch is
 * verifiable on a bare board with nothing attached. 1 reads an INA219 over I2C
 * and an LM35 on the ADC, and REFUSES rather than substituting a value when
 * either is absent. Override at build time:
 *
 *   --build-property compiler.cpp.extra_flags=-DUSE_REAL_SENSORS=1
 *
 * The default stays 0 so a clean checkout still flashes and emits on a board
 * with no sensors wired, which is what the bring-up procedure above assumes. */
#ifndef USE_REAL_SENSORS
#define USE_REAL_SENSORS 0
#endif

/* Pins for the real-sensor build. I2C is bit-banged by the ESP8266 core, so
 * these are free choices; D1/D2 avoid every strapping pin. */
#define PIN_I2C_SDA      4    /* D2 */
#define PIN_I2C_SCL      5    /* D1 */
#define PIN_LM35         A0

/* LM35 scaling for a NodeMCU v3: the board divides A0 by 220k/100k, so the pin
 * reads 0-3.2 V across 1024 counts (3.125 mV/count), and the LM35 gives
 * 10 mV/C. One count is therefore 0.3125 C. MEASURE YOUR OWN DIVIDER - a bare
 * ESP-12 has no divider at all and needs 0.0977 here. */
#define LM35_C_PER_COUNT 0.3125f
#define LM35_SAMPLES     32

/* Below this, treat the LM35 as disconnected rather than cold.
 *
 * A0 is held near ground by the divider's lower leg, so an unplugged sensor
 * reads about 0 C - inside the schema's -40..150 C range, and therefore
 * indistinguishable from a real measurement to the host. This floor is what
 * makes that failure detectable. It is a bench-rig assumption: a cell in a
 * room is never at 4 C. Do not use it where the rig might genuinely be cold. */
#define LM35_MIN_PLAUSIBLE_C 5.0f

#define SAMPLE_PERIOD_MS 1000

/* Identity reported to the host. One rig, one id. If you run two rigs into one
 * fleet view, give them different ids or their telemetry will be pooled. */
#define CELL_ID          "RIG_01"

#define EMIT_CHECKSUM    1   /* 1 = append *HH; 0 = omit (still conforming). */
#define EMIT_COMPACT     0   /* 0 = JSON records; 1 = compact key=value form. */

/* Nominal capacity of the cell under measurement, in amp-hours. SET THIS TO
 * YOUR CELL'S RATING - it is not a formality.
 *
 * It is declared to the host in the HELLO line and becomes the denominator of
 * every C-rate the host computes, which is how `aggressive_discharge_event` and
 * `fast_charge_flag` are defined. A 3.4 Ah 18650 left at the 2.0 below reads
 * 1.7 C while drawing 1 C, so both flags fire on every row of the capture and
 * the resulting stress score, health index and RUL are confidently wrong.
 *
 * The host refuses to score a capture from a rig that declares no capacity at
 * all, so this cannot be forgotten silently - but it CAN be wrong silently if
 * you leave a value here that does not match the cell you connected.
 *
 * A 3400 mAh cell is 3.4f, not 3400.0f.
 *
 * It is also used by the fallback coulomb-counting SOC estimate below. */
#define NOMINAL_CAPACITY_AH 2.2f   /* HONGLI ICR-18650-2200mAh, printed rating */

/* Protocol constants. Do not change these without changing the host: the host
 * refuses a schema id it does not implement rather than guessing that the
 * fields still mean what they used to. */
#define BEACON_SENTINEL  "BEACON1"
#define BEACON_SCHEMA    "beacon.telemetry.v1"
#define BEACON_FIRMWARE  "beacon-rig-reference/1.0"

#if defined(ESP32)
  #define BEACON_DEVICE "esp32"
#elif defined(ESP8266)
  #define BEACON_DEVICE "esp8266"
#elif defined(ARDUINO_AVR_UNO) || defined(ARDUINO_AVR_NANO) || defined(ARDUINO_AVR_MEGA2560)
  #define BEACON_DEVICE "avr"
#else
  #define BEACON_DEVICE "unknown"
#endif

/* ===================================================================== */
/* Wire formatting                                                       */
/* ===================================================================== */

static unsigned long g_startMillis = 0;

/* Sensor hardware. Declared here rather than inside the hardware layer only
 * because Arduino's generated prototypes need the types in scope. */
#if USE_REAL_SENSORS
#include <Wire.h>
#include <Adafruit_INA219.h>
static Adafruit_INA219 g_ina219;
static bool g_inaReady = false;
static float g_lm35Counts = 0.0f;   /* last raw A0 average, for the refusal text */
static bool  g_haveTemp   = false;  /* probed once at startup - see initSensors */
#endif

/* Forward declarations. The Arduino IDE generates these automatically, but
 * stating them keeps the file valid C++ for any other toolchain (PlatformIO,
 * arduino-cli with a plain compiler, a host-side compile check). */
void  initSensors();
float readVoltageV();
float readCurrentA();
float readTemperatureC();
float readSocPercent(float currentA, float dtSeconds);
float syntheticVoltageV();
float syntheticCurrentA();
float syntheticTemperatureC();

/* XOR of every byte, as two uppercase hex digits. */
static void appendChecksum(String &line, const String &body) {
  uint8_t checksum = 0;
  for (unsigned int i = 0; i < body.length(); i++) {
    checksum ^= (uint8_t)body[i];
  }
  char hex[4];
  snprintf(hex, sizeof(hex), "*%02X", checksum);
  line += hex;
}

static void emit(const char *kind, const String &body) {
  String line = String(BEACON_SENTINEL) + " " + kind + " " + body;
#if EMIT_CHECKSUM
  appendChecksum(line, body);
#endif
  Serial.println(line);
}

/* Format a float with enough precision for the host's coulomb counting without
 * wasting bytes. Four decimals resolves a milliamp on a 1 A rig. */
static String num(float value, int decimals = 4) {
  return String(value, decimals);
}

static void emitHello() {
  String body = "{";
  body += "\"schema\":\"" BEACON_SCHEMA "\",";
  /* Declare the sensors this rig ACTUALLY HAS, probed at startup - not the
   * ones it was designed around. The host holds a rig to its own declaration
   * and rejects records that omit a channel HELLO promised, which is right:
   * a channel that vanishes mid-capture is a fault and should be loud. So the
   * honest move is to promise less, not to deliver a fabricated value. */
#if USE_REAL_SENSORS
  body += "\"fields\":[\"t\"";
  if (g_inaReady) body += ",\"v\",\"i\",\"soc\"";
  if (g_haveTemp) body += ",\"tc\"";
  body += "],";
#else
  body += "\"fields\":[\"t\",\"v\",\"i\",\"tc\",\"soc\"],";
#endif
  body += "\"cell_id\":\"" CELL_ID "\",";
  body += "\"device\":\"" BEACON_DEVICE "\",";
  body += "\"firmware\":\"" BEACON_FIRMWARE "\",";
  body += "\"period_ms\":";
  body += String(SAMPLE_PERIOD_MS);
  /* The host divides current by this to get C-rate. Without it the host
   * refuses to score the capture rather than assuming a cell size. */
  body += ",\"capacity_ah\":";
  body += num(NOMINAL_CAPACITY_AH, 3);
  body += "}";
  emit("HELLO", body);
}

static void emitStatus(const char *text) {
  emit("S", String(text));
}

/* A channel that could not be measured is OMITTED, not zeroed and not sent as
 * NAN. The host requires only `t` on a data record - see TIME_FIELD and the
 * note on REQUIRED_WIRE_FIELDS in serial_schema.py - and its coverage gate
 * decides separately which metrics the delivered channels can support. So a
 * dead thermometer costs you the temperature features and nothing else; the
 * voltage and current beside it are still real measurements and still scored.
 *
 * Refusing the whole record instead would discard good data to report a bad
 * sensor, which is a different mistake from the one this project guards
 * against, and just as wrong. */
static void emitSample(float t, float v, float i, float tc, float soc) {
  bool haveV  = !isnan(v);
  bool haveI  = !isnan(i);
  bool haveTc = !isnan(tc);
  bool haveSoc = !isnan(soc) && haveI;   /* coulomb count is meaningless without current */

#if EMIT_COMPACT
  String body = "t=" + num(t, 3);
  if (haveV)   body += " v=" + num(v);
  if (haveI)   body += " i=" + num(i);
  if (haveTc)  body += " tc=" + num(tc, 3);
  if (haveSoc) body += " soc=" + num(soc, 2);
#else
  String body = "{\"t\":" + num(t, 3);
  if (haveV)   body += ",\"v\":" + num(v);
  if (haveI)   body += ",\"i\":" + num(i);
  if (haveTc)  body += ",\"tc\":" + num(tc, 3);
  if (haveSoc) body += ",\"soc\":" + num(soc, 2);
  body += "}";
#endif
  emit("D", body);
}

/* ===================================================================== */
/* Sketch                                                                */
/* ===================================================================== */

void setup() {
  Serial.begin(SERIAL_BAUD);
  /* Wait briefly for a USB-CDC port to enumerate. Bounded, because a board
   * powered from a wall adapter has no host to wait for and must still run. */
  unsigned long waitStart = millis();
  while (!Serial && (millis() - waitStart) < 3000) {
    delay(10);
  }

  /* Radios stay off for USB-serial operation. Their transmit current spikes can
   * brown out a weakly-supplied board, and a brown-out restarts millis() at
   * zero. The host refuses a capture whose time goes backwards rather than
   * sorting it, because sorting would interleave two unrelated sessions and
   * cancel charge against itself. Nothing here enables WiFi or BLE; if you add
   * them, expect resets and give the board a supply that can hold them. */

  g_startMillis = millis();

  initSensors();
  emitHello();
  emitStatus("rig started");
}

void loop() {
  static unsigned long nextSampleAt = 0;
  unsigned long now = millis();

  if ((long)(now - nextSampleAt) < 0) {
    return;
  }
  nextSampleAt = now + SAMPLE_PERIOD_MS;

  float elapsedS = (now - g_startMillis) / 1000.0f;

  float voltageV     = readVoltageV();
  float currentA     = readCurrentA();
  float temperatureC = readTemperatureC();
  float socPercent   = readSocPercent(currentA, SAMPLE_PERIOD_MS / 1000.0f);

  /* Announce a channel appearing or disappearing, once per transition. Saying
   * it every second would bury the data records it is meant to annotate. */
#if USE_REAL_SENSORS
  static int lastTempOk = -1;   /* -1 = nothing reported yet */
  int tempOk = isnan(temperatureC) ? 0 : 1;
  if (tempOk != lastTempOk) {
    if (tempOk) {
      emitStatus("temperature channel available");
    } else {
      String why = "temperature channel unavailable: LM35 raw="
                 + String(g_lm35Counts, 1) + " counts ("
                 + String(g_lm35Counts * LM35_C_PER_COUNT, 1) + " C), below "
                 + String(LM35_MIN_PLAUSIBLE_C, 1) + " C floor - check wiring";
      emitStatus(why.c_str());
    }
    lastTempOk = tempOk;
  }
#endif

  emitSample(elapsedS, voltageV, currentA, temperatureC, socPercent);
}

/* ===================================================================== */
/* Hardware layer - THE ONLY PART THAT TOUCHES SENSORS                   */
/* ===================================================================== */
/*
 * These ship as a synthetic profile so the sketch is verifiable on a bare board
 * with nothing attached: flash it, open a serial monitor, confirm BEACON1 lines
 * appear, and the host pipeline will score them. Replace them one at a time.
 *
 * No pin numbers, sensor part numbers or I2C addresses appear anywhere above
 * this line, so a replacement here is the whole integration.
 */

void initSensors() {
#if USE_REAL_SENSORS
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  g_inaReady = g_ina219.begin();
  if (!g_inaReady) {
    /* Say so and keep running. The readers below then return NAN and the
     * sample is refused, which is the honest outcome: the host sees a gap
     * with a stated reason rather than a plausible constant. */
    emitStatus("INA219 not responding on I2C 0x40 - voltage and current refused");
  } else {
    emitStatus("INA219 ready at 0x40");
  }
  pinMode(PIN_LM35, INPUT);

  /* One probe, at startup, to decide what to declare. Averaged the same way
   * the reader averages so the decision and the measurement agree. */
  long probe = 0;
  for (int i = 0; i < LM35_SAMPLES; i++) { probe += analogRead(PIN_LM35); delay(1); }
  g_lm35Counts = probe / (float)LM35_SAMPLES;
  g_haveTemp = (g_lm35Counts * LM35_C_PER_COUNT) >= LM35_MIN_PLAUSIBLE_C;
  if (!g_haveTemp) {
    String why = "no temperature sensor: A0 reads " + String(g_lm35Counts, 1)
               + " counts (" + String(g_lm35Counts * LM35_C_PER_COUNT, 1)
               + " C) - channel not declared";
    emitStatus(why.c_str());
  } else {
    emitStatus("LM35 ready on A0");
  }
#endif
  /* Sensor bring-up goes here. For example, with an INA219:
   *
   *   #include <Adafruit_INA219.h>
   *   Adafruit_INA219 ina219;          // declare at file scope
   *   ina219.begin();                  // here
   *
   * and with a DS18B20:
   *
   *   #include <DallasTemperature.h>
   *   sensors.begin();
   *
   * If a sensor fails to initialise, say so and keep running:
   *
   *   emitStatus("INA219 not responding on I2C");
   *
   * Status lines are reported by the host and never scored, so an honest
   * report costs nothing. Do NOT substitute a default reading for a sensor
   * that is not answering: the host treats an absent channel as a refusal and
   * a plausible-looking constant as a measurement.
   */
}

float readVoltageV() {
  /* Real implementation, divider on an analog pin:
   *
   *   const float DIVIDER_RATIO = 2.0f;   // measure yours, do not assume
   *   const float ADC_REF_V     = 3.3f;
   *   const float ADC_COUNTS    = 4095.0f;
   *   return analogRead(PIN) / ADC_COUNTS * ADC_REF_V * DIVIDER_RATIO;
   *
   * Or, with an INA219: return ina219.getBusVoltage_V();
   */
#if USE_REAL_SENSORS
  if (!g_inaReady) return NAN;          /* refuse, never substitute */
  return g_ina219.getBusVoltage_V();
#else
  return syntheticVoltageV();
#endif
}

float readCurrentA() {
  /* NEGATIVE FOR DISCHARGE. See the header note - this is the one that silently
   * costs an afternoon.
   *
   *   return -ina219.getCurrent_mA() / 1000.0f;   // sign depends on wiring
   *
   * Verify with a known load before trusting it: put the cell under load and
   * confirm the printed value is negative.
   */
#if USE_REAL_SENSORS
  if (!g_inaReady) return NAN;
  /* Negated so discharge reads negative with the cell on VIN+ and the load on
   * VIN-. Swap the sign here, not the leads, if your wiring is the other way. */
  return -g_ina219.getCurrent_mA() / 1000.0f;
#else
  return syntheticCurrentA();
#endif
}

float readTemperatureC() {
  /* Cell surface temperature in degrees Celsius.
   *
   *   sensors.requestTemperatures();
   *   return sensors.getTempCByIndex(0);
   *
   * A disconnected DS18B20 returns -127.0, and a floating thermistor input
   * reads full scale. Both fall outside the schema's declared range, so the
   * host rejects those records rather than averaging them into a health index.
   * That is intended: do not clamp them here.
   */
#if USE_REAL_SENSORS
  /* Averaged because the ESP8266 ADC is noisy at this scale: the LM35 uses
   * only the bottom fifth of the range, so a few counts of noise is a degree.
   * 32 samples cuts it by about 5.7x. */
  long sum = 0;
  for (int i = 0; i < LM35_SAMPLES; i++) { sum += analogRead(PIN_LM35); delay(1); }
  g_lm35Counts = sum / (float)LM35_SAMPLES;
  float celsius = g_lm35Counts * LM35_C_PER_COUNT;
  if (celsius < LM35_MIN_PLAUSIBLE_C) return NAN;   /* disconnected, not cold */
  return celsius;
#else
  return syntheticTemperatureC();
#endif
}

float readSocPercent(float currentA, float dtSeconds) {
  /* PERCENT, 0-100.
   *
   * If your BMS reports SOC, return it here. Otherwise this coulomb-counting
   * estimate is a reasonable stand-in for a bench rig, with a caveat worth
   * understanding: it integrates current, so its error accumulates and it has
   * no way to re-reference itself. Over a long capture it drifts. The host's
   * capacity measurement does NOT depend on this value - capacity is integrated
   * from current directly - so drift here degrades the behaviour flags, not the
   * state-of-health figure.
   */
  static float accumulatedAh = NOMINAL_CAPACITY_AH;
  accumulatedAh += currentA * dtSeconds / 3600.0f;
  if (accumulatedAh < 0.0f)                  accumulatedAh = 0.0f;
  if (accumulatedAh > NOMINAL_CAPACITY_AH)   accumulatedAh = NOMINAL_CAPACITY_AH;
  return 100.0f * accumulatedAh / NOMINAL_CAPACITY_AH;
}

/* --------------------------------------------------------------------- */
/* Synthetic profile - delete once every reader above is real.            */
/* --------------------------------------------------------------------- */

static bool  g_discharging = true;
static float g_phaseStart  = 0.0f;

static float phaseElapsedS() {
  return (millis() - g_startMillis) / 1000.0f - g_phaseStart;
}

static void advancePhaseIfDue(float phaseLengthS) {
  if (phaseElapsedS() >= phaseLengthS) {
    g_discharging = !g_discharging;
    g_phaseStart = (millis() - g_startMillis) / 1000.0f;
  }
}

float syntheticCurrentA() {
  advancePhaseIfDue(120.0f);
  return g_discharging ? -1.5f : 0.75f;
}

float syntheticVoltageV() {
  float fraction = phaseElapsedS() / 120.0f;
  if (fraction > 1.0f) fraction = 1.0f;
  return g_discharging ? (4.15f - 1.15f * fraction)
                       : (3.00f + 1.15f * fraction);
}

float syntheticTemperatureC() {
  return 25.0f + (g_discharging ? 6.0f : 3.0f) * (phaseElapsedS() / 120.0f);
}
