#include <Wire.h>
#include <Adafruit_INA219.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <math.h>

#include "model.h"
#include <tflm_esp32.h>
#include <eloquent_tinyml.h>

// ─── Pin Definitions ─────────────────────────────────────────────────────────
#define OLED_MOSI   23
#define OLED_CLK    18
#define OLED_DC      2
#define OLED_RESET  15
#define GATE_PIN     5
#define BOOT_PIN     0

// ─── Display ──────────────────────────────────────────────────────────────────
#define SCREEN_W 128
#define SCREEN_H  64
Adafruit_SSD1306 display(SCREEN_W, SCREEN_H,
    OLED_MOSI, OLED_CLK, OLED_DC, OLED_RESET, -1);

// ─── Sensors ──────────────────────────────────────────────────────────────────
Adafruit_ADS1115 ads;
Adafruit_INA219  ina219;

// ─── TFLite ───────────────────────────────────────────────────────────────────
#define N_INPUTS         6
#define N_OUTPUTS        1
#define TF_NUM_OPS       8
#define ARENA_SIZE      (96 * 1024)
Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> ml;

// ─── MinMax scaler params (match your Python training scaler) ─────────────────
const float SCALER_MIN[6] = { 16.873f, 0.725111f, 0.970588f, 16.873f, 2.6104f, 17.526f };
const float SCALER_MAX[6] = { 29.071f, 0.786388f, 1.335163f, 29.071f, 2.831f, 35.493f };

// ─── Timing ───────────────────────────────────────────────────────────────────
#define CYCLE_INTERVAL_MS   10000UL
#define STARTUP_IGNORE_MS    4000UL
#define DEBOUNCE_MS            50UL
#define DOUBLE_TAP_MS         500UL

// ─── Load pulse durations ─────────────────────────────────────────────────────
#define PRE_OCV_SETTLE_MS  1000UL   // settle before OCV read
#define PULSE_10MS           10UL   // short pulse for IR(10ms)
#define PULSE_1S            990UL   // remainder to reach 1 s total load time

// ─── State ────────────────────────────────────────────────────────────────────
bool oledReady     = false;
bool sensorsReady  = false;
bool mlReady       = false;
bool paused        = false;

unsigned long lastCycleTime   = 0;
unsigned long appStartTime    = 0;
unsigned long lastDebounce    = 0;
unsigned long firstTapTime    = 0;
int  lastBootReading          = HIGH;
bool waitingSecondTap         = false;

// ─── Helpers ──────────────────────────────────────────────────────────────────
float clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

float minMaxScale(float val, int idx) {
    float d = SCALER_MAX[idx] - SCALER_MIN[idx];
    if (d <= 0.0f) return 0.0f;
    return clamp((val - SCALER_MIN[idx]) / d, 0.0f, 1.0f);
}

// ─── Init helpers ─────────────────────────────────────────────────────────────
bool initOLED() {
    if (!display.begin(SSD1306_SWITCHCAPVCC)) return false;
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("Battery SoH Estimator");
    display.println("Initializing...");
    display.display();
    delay(800);
    return true;
}

bool initSensors() {
    Wire.begin();
    bool inaOk = ina219.begin();
    bool adsOk = ads.begin(0x48);
    if (adsOk) {
        ads.setGain(GAIN_ONE);
        ads.setDataRate(RATE_ADS1115_860SPS);
    }
    return inaOk && adsOk;
}

bool initModel() {
    ml.setNumInputs(N_INPUTS);
    ml.setNumOutputs(N_OUTPUTS);
    ml.resolver.AddFullyConnected();
    ml.resolver.AddQuantize();
    ml.resolver.AddDequantize();
    ml.resolver.AddRelu();
    if (!ml.begin(model_tflite).isOk()) {
        Serial.print("[ML] Init failed: ");
        Serial.println(ml.exception.toString());
        return false;
    }
    return true;
}

// ─── OLED screens ─────────────────────────────────────────────────────────────
void oledShowReadings(float soh, float v_ocv, float ir_10ms) {
    if (!oledReady) return;
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(14, 0);
    display.println("Battery SoH Meter");
    display.drawLine(0, 9, 127, 9, SSD1306_WHITE);
    display.setTextSize(2);
    display.setCursor(8, 14);
    display.print("SoH:");
    display.print(soh, 1);
    display.print("%");
    display.setTextSize(1);
    display.setCursor(0, 38);
    display.print("V_ocv: ");
    display.print(v_ocv, 4);
    display.print(" V");
    display.setCursor(0, 50);
    display.print("IR:    ");
    display.print(ir_10ms, 2);
    display.print(" mOhm");
    display.display();
}

void oledShowPaused() {
    if (!oledReady) return;
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(14, 0);
    display.println("Battery SoH Meter");
    display.drawLine(0, 9, 127, 9, SSD1306_WHITE);
    display.setTextSize(2);
    display.setCursor(22, 24);
    display.println("Paused");
    display.display();
}

// ─── Button handler ──────────────────────────────────────────────────────────
// single tap  → pause / resume
// double tap  → reserved (no-op, can extend later)
void handleBootButton() {
    if (millis() - appStartTime < STARTUP_IGNORE_MS) {
        lastBootReading = digitalRead(BOOT_PIN);
        waitingSecondTap = false;
        return;
    }

    int reading = digitalRead(BOOT_PIN);
    if (reading != lastBootReading) lastDebounce = millis();

    if ((millis() - lastDebounce) > DEBOUNCE_MS) {
        static int stableState = HIGH;
        if (reading != stableState) {
            stableState = reading;
            if (stableState == LOW) {
                if (waitingSecondTap && (millis() - firstTapTime <= DOUBLE_TAP_MS)) {
                    waitingSecondTap = false;
                    // double tap — reserved for future feature
                    Serial.println("[BTN] Double tap detected.");
                } else {
                    waitingSecondTap = true;
                    firstTapTime = millis();
                }
            }
        }
    }

    if (waitingSecondTap && (millis() - firstTapTime > DOUBLE_TAP_MS)) {
        waitingSecondTap = false;
        paused = !paused;
        Serial.print("[BTN] State: ");
        Serial.println(paused ? "paused" : "running");
        if (paused) oledShowPaused();
        else lastCycleTime = millis();
    }

    lastBootReading = reading;
}

// ─── Core measurement + inference cycle ──────────────────────────────────────
void runCycle() {
    Serial.println("\n========= NEW MEASUREMENT CYCLE =========");

    // 1. Settle then read OCV (no load)
    digitalWrite(GATE_PIN, LOW);
    delay(PRE_OCV_SETTLE_MS);

    int16_t raw = ads.readADC_SingleEnded(0);
    float v_ocv = ads.computeVolts(raw);

    // 2. Apply load, read current & voltage at 10 ms
    digitalWrite(GATE_PIN, HIGH);
    delay(PULSE_10MS);

    raw = ads.readADC_SingleEnded(0);
    float v_under_10ms = ads.computeVolts(raw);
    float i_10ms = ina219.getCurrent_mA() / 1000.0f;  // A

    // 3. Continue load to 1 s, read again
    delay(PULSE_1S);

    raw = ads.readADC_SingleEnded(0);
    float v_under_1s = ads.computeVolts(raw);
    float i_1s = ina219.getCurrent_mA() / 1000.0f;    // A

    // 4. Remove load
    digitalWrite(GATE_PIN, LOW);

    // 5. Compute IR values (mOhm = dV/dI * 1000)
    float delta_v_10ms = v_ocv - v_under_10ms;
    float delta_v_1s   = v_ocv - v_under_1s;
    float ir_10ms = (i_10ms > 0.001f) ? (delta_v_10ms / i_10ms) * 1000.0f : 0.0f;
    float ir_1s   = (i_1s   > 0.001f) ? (delta_v_1s   / i_1s  ) * 1000.0f : 0.0f;

    // Guard: if IR values are physically implausible, warn and skip inference
    if (ir_10ms <= 0.0f || ir_10ms > 200.0f || ir_1s <= 0.0f || ir_1s > 200.0f) {
        Serial.println("[WARN] IR values out of range — check connections.");
        Serial.print("  ir_10ms: "); Serial.println(ir_10ms);
        Serial.print("  ir_1s  : "); Serial.println(ir_1s);
        return;
    }

    // 6. Build raw feature vector (must match training order)
    float raw_f[6] = {
        ir_10ms,                                             // f0
        v_ocv / 3.6f,                                        // f1
        (ir_10ms > 0.0f) ? (ir_1s / ir_10ms) : 0.0f,        // f2
        ir_10ms,                                             // f3 (copy — matches training)
        v_ocv,                                               // f4
        ir_1s                                                // f5
    };

    // 7. MinMax scale
    float ml_in[6];
    for (int i = 0; i < 6; i++) ml_in[i] = minMaxScale(raw_f[i], i);

    // 8. Run inference
    float predicted_soh = 0.0f;
    if (mlReady) {
        if (ml.predict(ml_in).isOk()) {
            predicted_soh = ml.output(0) * 100.0f;  // model outputs 0‑1, scale to %
            predicted_soh = clamp(predicted_soh, 0.0f, 100.0f);
        } else {
            Serial.print("[ML] Inference error: ");
            Serial.println(ml.exception.toString());
            return;
        }
    } else {
        Serial.println("[ML] Model not loaded — skipping inference.");
        return;
    }

    // 9. Serial log
    Serial.println("--- SENSOR DATA ---");
    Serial.print("V_ocv        : "); Serial.print(v_ocv, 4);        Serial.println(" V");
    Serial.print("V_under_10ms : "); Serial.print(v_under_10ms, 4); Serial.println(" V");
    Serial.print("V_under_1s   : "); Serial.print(v_under_1s, 4);   Serial.println(" V");
    Serial.print("I_10ms       : "); Serial.print(i_10ms, 4);        Serial.println(" A");
    Serial.print("I_1s         : "); Serial.print(i_1s, 4);          Serial.println(" A");
    Serial.print("IR_10ms      : "); Serial.print(ir_10ms, 3);       Serial.println(" mOhm");
    Serial.print("IR_1s        : "); Serial.print(ir_1s, 3);         Serial.println(" mOhm");
    Serial.println("--- ML INPUTS (scaled) ---");
    for (int i = 0; i < 6; i++) {
        Serial.print("x["); Serial.print(i); Serial.print("] = ");
        Serial.println(ml_in[i], 6);
    }
    Serial.println("-------------------");
    Serial.print(">>> PREDICTED SOH: ");
    Serial.print(predicted_soh, 2);
    Serial.println(" % <<<");
    Serial.println("==========================================\n");

    // 10. OLED
    oledShowReadings(predicted_soh, v_ocv, ir_10ms);
}

// ─── Setup ────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(1500);
    appStartTime = millis();

    pinMode(GATE_PIN, OUTPUT);
    digitalWrite(GATE_PIN, LOW);
    pinMode(BOOT_PIN, INPUT_PULLUP);

    Serial.println("\n==========================================");
    Serial.println("   BATTERY SoH ESTIMATOR — LIVE MODE     ");
    Serial.println("==========================================");

    oledReady    = initOLED();
    sensorsReady = initSensors();
    mlReady      = initModel();

    Serial.println("------------------------------------------");
    Serial.print("OLED    : "); Serial.println(oledReady    ? "OK" : "FAIL");
    Serial.print("Sensors : "); Serial.println(sensorsReady ? "OK" : "FAIL");
    Serial.print("Model   : "); Serial.println(mlReady      ? "OK" : "FAIL");
    Serial.println("------------------------------------------");
    Serial.println("Controls:");
    Serial.println("  BOOT single tap  — pause / resume");
    Serial.println("  'T' via Serial   — manual trigger");
    Serial.println("==========================================\n");

    lastCycleTime = millis();
}

// ─── Loop ─────────────────────────────────────────────────────────────────────
void loop() {
    handleBootButton();

    if (Serial.available() > 0) {
        char c = Serial.read();
        while (Serial.available() > 0) Serial.read();
        if ((c == 'T' || c == 't') && !paused) {
            runCycle();
            lastCycleTime = millis();
        }
    }

    if (paused) { delay(10); return; }

    if (millis() - lastCycleTime >= CYCLE_INTERVAL_MS) {
        runCycle();
        lastCycleTime = millis();
    }
}