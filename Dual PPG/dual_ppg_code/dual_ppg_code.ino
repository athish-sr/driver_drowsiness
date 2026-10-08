#include <Wire.h>
#include "MAX30105.h"

// ============================================================
// TWO MAX30102 SENSORS
// ============================================================

MAX30105 sensor1;
MAX30105 sensor2;

// ============================================================
// I2C BUS 0 - SENSOR 1
// ============================================================

TwoWire I2C_1 = TwoWire(0);

#define SDA1 21
#define SCL1 22

// ============================================================
// I2C BUS 1 - SENSOR 2
// ============================================================

TwoWire I2C_2 = TwoWire(1);

#define SDA2 25
#define SCL2 26

// ============================================================
// LED POWER
// ============================================================

byte irPower1  = 40;
byte redPower1 = 40;

byte irPower2  = 40;
byte redPower2 = 40;

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

  // ==========================================================
  // INITIALIZE I2C BUS 0
  // ==========================================================

  I2C_1.begin(SDA1, SCL1, 800000);

  // ==========================================================
  // INITIALIZE I2C BUS 1
  // ==========================================================

  I2C_2.begin(SDA2, SCL2, 800000);

  // ==========================================================
  // INITIALIZE SENSOR 1
  // ==========================================================

  if (!sensor1.begin(I2C_1, I2C_SPEED_FAST)) {

    Serial.println("ERROR:MAX30102_SENSOR1_NOT_FOUND");

    while (1) {
      delay(100);
    }
  }

  // ==========================================================
  // INITIALIZE SENSOR 2
  // ==========================================================

  if (!sensor2.begin(I2C_2, I2C_SPEED_FAST)) {

    Serial.println("ERROR:MAX30102_SENSOR2_NOT_FOUND");

    while (1) {
      delay(100);
    }
  }

  // ==========================================================
  // SENSOR 1 CONFIGURATION
  // ==========================================================

  sensor1.setup(
    0,
    8,
    2,
    400,
    411,
    4096
  );

  sensor1.setPulseAmplitudeIR(irPower1);
  sensor1.setPulseAmplitudeRed(redPower1);

  // ==========================================================
  // SENSOR 2 CONFIGURATION
  // ==========================================================

  sensor2.setup(
    0,
    8,
    2,
    400,
    411,
    4096
  );

  sensor2.setPulseAmplitudeIR(irPower2);
  sensor2.setPulseAmplitudeRed(redPower2);

  // ==========================================================
  // STATUS
  // ==========================================================

  Serial.println("MAX30102_SENSOR1_READY");
  Serial.println("MAX30102_SENSOR2_READY");

  Serial.println("WAITING_FOR_FINGER");

  // ==========================================================
  // WAIT FOR BOTH FINGERS
  // ==========================================================

  while (
    sensor1.getIR() < FINGER_THRESHOLD ||
    sensor2.getIR() < FINGER_THRESHOLD
  ) {

    delay(50);
  }

  Serial.println("BOTH_FINGERS_DETECTED");

  delay(500);

  // ==========================================================
  // CSV HEADER
  // ==========================================================

  Serial.println(
    "timestamp,IR1,RED1,IR2,RED2"
  );
}

// ============================================================
// LOOP
// ============================================================

void loop() {

  // ==========================================================
  // READ SENSOR 1
  // ==========================================================

  uint32_t irRaw1  = sensor1.getIR();
  uint32_t redRaw1 = sensor1.getRed();

  // ==========================================================
  // READ SENSOR 2
  // ==========================================================

  uint32_t irRaw2  = sensor2.getIR();
  uint32_t redRaw2 = sensor2.getRed();

  // ==========================================================
  // FINGER STATUS SENSOR 1
  // ==========================================================

  bool finger1 = irRaw1 >= FINGER_THRESHOLD;

  // ==========================================================
  // FINGER STATUS SENSOR 2
  // ==========================================================

  bool finger2 = irRaw2 >= FINGER_THRESHOLD;

  // ==========================================================
  // AUTO LED CONTROL
  // SENSOR 1
  // ==========================================================

  if (millis() - lastAdjustTime > 50) {

    bool adjusted = false;

    // --------------------------------------------------------
    // SENSOR 1 - IR LED
    // --------------------------------------------------------

    if (finger1) {

      if (irRaw1 < 60000 && irPower1 < 250) {

        irPower1 += 2;
        adjusted = true;
      }

      else if (irRaw1 > 220000 && irPower1 > 5) {

        irPower1 = (irPower1 > 10)
                   ? irPower1 - 5
                   : 5;

        adjusted = true;
      }

      else if (irRaw1 > 180000 && irPower1 > 5) {

        irPower1 -= 2;
        adjusted = true;
      }

      // ------------------------------------------------------
      // SENSOR 1 - RED LED
      // ------------------------------------------------------

      if (redRaw1 < 60000 && redPower1 < 250) {

        redPower1 += 2;
        adjusted = true;
      }

      else if (redRaw1 > 220000 && redPower1 > 5) {

        redPower1 = (redPower1 > 10)
                    ? redPower1 - 5
                    : 5;

        adjusted = true;
      }

      else if (redRaw1 > 180000 && redPower1 > 5) {

        redPower1 -= 2;
        adjusted = true;
      }
    }

    // ========================================================
    // AUTO LED CONTROL
    // SENSOR 2
    // ========================================================

    if (finger2) {

      // ------------------------------------------------------
      // SENSOR 2 - IR LED
      // ------------------------------------------------------

      if (irRaw2 < 60000 && irPower2 < 250) {

        irPower2 += 2;
        adjusted = true;
      }

      else if (irRaw2 > 220000 && irPower2 > 5) {

        irPower2 = (irPower2 > 10)
                   ? irPower2 - 5
                   : 5;

        adjusted = true;
      }

      else if (irRaw2 > 180000 && irPower2 > 5) {

        irPower2 -= 2;
        adjusted = true;
      }

      // ------------------------------------------------------
      // SENSOR 2 - RED LED
      // ------------------------------------------------------

      if (redRaw2 < 60000 && redPower2 < 250) {

        redPower2 += 2;
        adjusted = true;
      }

      else if (redRaw2 > 220000 && redPower2 > 5) {

        redPower2 = (redPower2 > 10)
                    ? redPower2 - 5
                    : 5;

        adjusted = true;
      }

      else if (redRaw2 > 180000 && redPower2 > 5) {

        redPower2 -= 2;
        adjusted = true;
      }
    }

    // ========================================================
    // APPLY LED POWER
    // ========================================================

    if (adjusted) {

      sensor1.setPulseAmplitudeIR(irPower1);
      sensor1.setPulseAmplitudeRed(redPower1);

      sensor2.setPulseAmplitudeIR(irPower2);
      sensor2.setPulseAmplitudeRed(redPower2);
    }

    lastAdjustTime = millis();
  }

  // ==========================================================
  // SEND DATA TO PYTHON
  // ==========================================================

  Serial.print(millis());
  Serial.print(",");

  Serial.print(irRaw1);
  Serial.print(",");

  Serial.print(redRaw1);
  Serial.print(",");

  Serial.print(irRaw2);
  Serial.print(",");

  Serial.println(redRaw2);
}