#include <Wire.h>
#include "MAX30105.h"

MAX30105 sensor;

// ============================================================
// ESP32 I2C
// ============================================================

#define SDA_PIN 21
#define SCL_PIN 22

// ============================================================
// AUTO LED BRIGHTNESS
// ============================================================

byte irPower  = 40;
byte redPower = 40;

unsigned long lastAdjustTime = 0;

// ============================================================
// FINGER DETECTION
// ============================================================

#define FINGER_THRESHOLD 20000

// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR:MAX30102_NOT_FOUND");
    while (1) {
      delay(100);
    }
  }

  /*
     Configuration:

     LED mode      = 2 -> RED + IR
     Sample rate   = 400 SPS
     Sample average= 8

     Effective output ≈ 50 samples/sec
     Pulse width    = 411 us
     ADC range      = 4096
  */

  sensor.setup(
    0,       // LED brightness initially controlled below
    8,       // sample average
    2,       // RED + IR
    400,     // sample rate
    411,     // pulse width
    4096     // ADC range
  );

  sensor.setPulseAmplitudeIR(irPower);
  sensor.setPulseAmplitudeRed(redPower);

  Serial.println("MAX30102_READY");

  // ----------------------------------------------------------
  // Wait for finger
  // ----------------------------------------------------------

  Serial.println("WAITING_FOR_FINGER");

  while (sensor.getIR() < FINGER_THRESHOLD) {
    delay(50);
  }

  Serial.println("FINGER_DETECTED");

  delay(500);

  // CSV header
  Serial.println("timestamp,IR,RED");
}

// ============================================================
// LOOP
// ============================================================

void loop() {

  // ----------------------------------------------------------
  // Read RAW sensor values
  // ----------------------------------------------------------

  uint32_t irRaw  = sensor.getIR();
  uint32_t redRaw = sensor.getRed();

  // ----------------------------------------------------------
  // Finger removed
  // ----------------------------------------------------------

  if (irRaw < FINGER_THRESHOLD) {

    Serial.println("FINGER_REMOVED");

    while (sensor.getIR() < FINGER_THRESHOLD) {
      delay(50);
    }

    Serial.println("FINGER_DETECTED");

    return;
  }

  // ----------------------------------------------------------
  // AUTO LED CONTROL
  // ----------------------------------------------------------

  if (millis() - lastAdjustTime > 50) {

    bool adjusted = false;

    // IR LED

    if (irRaw < 60000 && irPower < 250) {

      irPower += 2;
      adjusted = true;

    }
    else if (irRaw > 220000 && irPower > 5) {

      irPower = (irPower > 10) ? irPower - 5 : 5;
      adjusted = true;

    }
    else if (irRaw > 180000 && irPower > 5) {

      irPower -= 2;
      adjusted = true;
    }

    // RED LED

    if (redRaw < 60000 && redPower < 250) {

      redPower += 2;
      adjusted = true;

    }
    else if (redRaw > 220000 && redPower > 5) {

      redPower = (redPower > 10) ? redPower - 5 : 5;
      adjusted = true;

    }
    else if (redRaw > 180000 && redPower > 5) {

      redPower -= 2;
      adjusted = true;
    }

    if (adjusted) {

      sensor.setPulseAmplitudeIR(irPower);
      sensor.setPulseAmplitudeRed(redPower);
    }

    lastAdjustTime = millis();
  }

  // ----------------------------------------------------------
  // SEND RAW DATA TO PYTHON
  // ----------------------------------------------------------

  Serial.print(millis());
  Serial.print(",");
  Serial.print(irRaw);
  Serial.print(",");
  Serial.println(redRaw);
}