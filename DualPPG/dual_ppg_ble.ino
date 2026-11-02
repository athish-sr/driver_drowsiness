#include <Wire.h>
#include "MAX30105.h"
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ============================================================
// TWO MAX30102 SENSORS  -  25 Hz output = one sample every 40 ms
// ============================================================

MAX30105 sensor1;
MAX30105 sensor2;

TwoWire I2C_1 = TwoWire(0);
TwoWire I2C_2 = TwoWire(1);

#define SDA1 21
#define SCL1 22
#define SDA2 25
#define SCL2 26
#define FINGER_THRESHOLD 20000

// Alert outputs (active-HIGH: pin HIGH = buzzer sounds / LED lights)
#define BUZZER_PIN 14
#define ALERT_LED_PIN 13

// Output rate = SAMPLE_RATE / SAMPLE_AVERAGE = 200 / 8 = 25 Hz = 40 ms
#define SAMPLE_AVERAGE 8
#define SAMPLE_RATE    200
#define PULSE_WIDTH    411
#define ADC_RANGE      4096

// ============================================================
// SENSOR HEALTH WATCHDOG (restarts ESP32 if a sensor dies)
// ============================================================

#define SENSOR_I2C_ADDR     0x57     // MAX30102 / MAX30105 fixed address
#define SENSOR_TIMEOUT_MS   2000     // no new sample for this long = dead
#define PROBE_INTERVAL_MS   500      // how often to ping each sensor on I2C
#define PROBE_FAIL_LIMIT    3        // consecutive failed pings = dead

unsigned long lastData1 = 0, lastData2 = 0, lastProbe = 0;
uint8_t probeFail1 = 0, probeFail2 = 0;

bool sensorPresent(TwoWire &bus) {
  bus.beginTransmission(SENSOR_I2C_ADDR);
  return bus.endTransmission() == 0;      // 0 = ACK received
}

void restartDevice() {
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(ALERT_LED_PIN, LOW);
  delay(100);
  ESP.restart();
}

// ============================================================
// BLUETOOTH LOW ENERGY CONFIGURATION
// ============================================================

static const char *DEVICE_NAME = "DualPPG-ESP32";
static const char *SERVICE_UUID = "7b2e0001-6f4a-4b9a-9c2d-3e5f6a7b8c01";
static const char *PPG_CHARACTERISTIC_UUID = "7b2e0002-6f4a-4b9a-9c2d-3e5f6a7b8c01";
static const char *ALERT_CHARACTERISTIC_UUID = "7b2e0003-6f4a-4b9a-9c2d-3e5f6a7b8c01";
static const char *STATUS_CHARACTERISTIC_UUID = "7b2e0004-6f4a-4b9a-9c2d-3e5f6a7b8c01";

BLECharacteristic *ppgCharacteristic;
BLECharacteristic *alertCharacteristic;
BLECharacteristic *statusCharacteristic;

volatile bool deviceConnected = false;
volatile bool alertCommandPending = false;
volatile uint8_t pendingBuzzerState = 0;
volatile uint8_t pendingLedState = 0;
bool alertBuzzerOn = false;
bool alertLedOn = false;

// ============================================================
// SAMPLE QUEUE (sampling loop -> BLE task)
// ============================================================

struct PpgSample {            // 20 bytes, same packet format as before
  uint32_t t;
  uint32_t ir1, red1, ir2, red2;
};

QueueHandle_t sampleQueue;
#define QUEUE_LEN 100         // 100 x 40 ms = 4 s of buffering

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server, esp_ble_gatts_cb_param_t *param) override {
    deviceConnected = true;
    // Ask for a short connection interval (7.5-15 ms, units of 1.25 ms)
    // so each 40 ms notification goes out promptly.
    server->updateConnParams(param->connect.remote_bda, 0x06, 0x0C, 0, 400);
  }

  void onDisconnect(BLEServer *server) override {
    deviceConnected = false;
    BLEDevice::startAdvertising();
  }
};

class AlertCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *characteristic) override {
    String value = characteristic->getValue();
    if (
      value.length() == 2 &&
      static_cast<uint8_t>(value.charAt(0)) <= 1 &&
      static_cast<uint8_t>(value.charAt(1)) <= 1
    ) {
      pendingBuzzerState = static_cast<uint8_t>(value.charAt(0));
      pendingLedState = static_cast<uint8_t>(value.charAt(1));
      alertCommandPending = true;
    }
  }
};

// ============================================================
// SENSOR LED POWER CONTROL
// ============================================================

byte irPower1 = 40;
byte redPower1 = 40;
byte irPower2 = 40;
byte redPower2 = 40;
unsigned long lastAdjustTime = 0;

// Latest values from each sensor (kept between loop passes)
uint32_t irRaw1 = 0, redRaw1 = 0, irRaw2 = 0, redRaw2 = 0;

void adjustLed(bool finger, uint32_t irRaw, uint32_t redRaw,
               byte &irPower, byte &redPower, bool &adjusted) {
  if (!finger) return;

  if (irRaw < 60000 && irPower < 250) { irPower += 2; adjusted = true; }
  else if (irRaw > 220000 && irPower > 5) { irPower = (irPower > 10) ? irPower - 5 : 5; adjusted = true; }
  else if (irRaw > 180000 && irPower > 5) { irPower -= 2; adjusted = true; }

  if (redRaw < 60000 && redPower < 250) { redPower += 2; adjusted = true; }
  else if (redRaw > 220000 && redPower > 5) { redPower = (redPower > 10) ? redPower - 5 : 5; adjusted = true; }
  else if (redRaw > 180000 && redPower > 5) { redPower -= 2; adjusted = true; }
}

// ============================================================
// ALERT STATUS (called only from the BLE task)
// ============================================================

void publishAlertStatus() {
  if (!deviceConnected || statusCharacteristic == nullptr) return;

  uint8_t status[2] = {
    static_cast<uint8_t>(alertBuzzerOn),
    static_cast<uint8_t>(alertLedOn)
  };
  statusCharacteristic->setValue(status, sizeof(status));
  statusCharacteristic->notify();
}

void handleAlertCommand() {
  if (!alertCommandPending) return;

  alertBuzzerOn = pendingBuzzerState == 1;
  alertLedOn = pendingLedState == 1;
  alertCommandPending = false;

  digitalWrite(BUZZER_PIN, alertBuzzerOn ? HIGH : LOW);
  digitalWrite(ALERT_LED_PIN, alertLedOn ? HIGH : LOW);
  publishAlertStatus();
}

// ============================================================
// BLE TASK (core 0) - sends queued samples so notify() can never
// delay the 40 ms sampling in loop()
// ============================================================

void bleTask(void *param) {
  PpgSample s;

  for (;;) {
    handleAlertCommand();

    if (deviceConnected) {
      // Send a few queued samples per pass, paced so the BLE
      // stack's buffers are not overrun.
      for (int i = 0; i < 4; i++) {
        if (xQueueReceive(sampleQueue, &s, 0) != pdTRUE) break;
        ppgCharacteristic->setValue(reinterpret_cast<uint8_t *>(&s), sizeof(s));
        ppgCharacteristic->notify();
        vTaskDelay(pdMS_TO_TICKS(2));
      }
    } else {
      // Nobody connected: discard so stale data is not sent later
      xQueueReset(sampleQueue);
    }

    vTaskDelay(pdMS_TO_TICKS(2));
  }
}

void startBluetooth() {
  BLEDevice::init(DEVICE_NAME);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService *service = server->createService(SERVICE_UUID);

  ppgCharacteristic = service->createCharacteristic(
    PPG_CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  ppgCharacteristic->addDescriptor(new BLE2902());

  alertCharacteristic = service->createCharacteristic(
    ALERT_CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_WRITE
  );
  alertCharacteristic->setCallbacks(new AlertCallbacks());

  statusCharacteristic = service->createCharacteristic(
    STATUS_CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
  );
  statusCharacteristic->addDescriptor(new BLE2902());
  uint8_t initialStatus[2] = {0, 0};
  statusCharacteristic->setValue(initialStatus, sizeof(initialStatus));

  service->start();

  BLEAdvertising *advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setScanResponse(true);
  BLEDevice::startAdvertising();
}

// ============================================================
// SETUP
// ============================================================

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(ALERT_LED_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(ALERT_LED_PIN, LOW);

  I2C_1.begin(SDA1, SCL1, 400000);
  I2C_2.begin(SDA2, SCL2, 400000);

  // If either sensor is missing at boot, wait a moment and restart
  if (!sensor1.begin(I2C_1, I2C_SPEED_FAST) ||
      !sensor2.begin(I2C_2, I2C_SPEED_FAST)) {
    delay(1000);
    restartDevice();
  }

  // setup(brightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange)
  sensor1.setup(0, SAMPLE_AVERAGE, 2, SAMPLE_RATE, PULSE_WIDTH, ADC_RANGE);
  sensor2.setup(0, SAMPLE_AVERAGE, 2, SAMPLE_RATE, PULSE_WIDTH, ADC_RANGE);

  sensor1.setPulseAmplitudeIR(irPower1);
  sensor1.setPulseAmplitudeRed(redPower1);
  sensor2.setPulseAmplitudeIR(irPower2);
  sensor2.setPulseAmplitudeRed(redPower2);

  sensor1.clearFIFO();
  sensor2.clearFIFO();

  sampleQueue = xQueueCreate(QUEUE_LEN, sizeof(PpgSample));

  startBluetooth();

  // BLE sending on core 0; Arduino loop() (sampling) runs on core 1
  xTaskCreatePinnedToCore(bleTask, "ble", 4096, NULL, 1, NULL, 0);

  // Start the watchdog timers only after everything is initialised
  lastData1 = lastData2 = lastProbe = millis();
}

// ============================================================
// LOOP - one sample pair per hardware FIFO sample (40 ms)
// ============================================================

void loop() {
  // check() returns the number of new samples it pulled from the FIFO
  if (sensor1.check() > 0) lastData1 = millis();
  if (sensor2.check() > 0) lastData2 = millis();

  // ---- Watchdog 1: no new data from a sensor for too long ----
  unsigned long now = millis();
  if (now - lastData1 > SENSOR_TIMEOUT_MS || now - lastData2 > SENSOR_TIMEOUT_MS) {
    restartDevice();
  }

  // ---- Watchdog 2: sensor no longer answers on I2C ----
  if (now - lastProbe >= PROBE_INTERVAL_MS) {
    lastProbe = now;

    probeFail1 = sensorPresent(I2C_1) ? 0 : probeFail1 + 1;
    probeFail2 = sensorPresent(I2C_2) ? 0 : probeFail2 + 1;

    if (probeFail1 >= PROBE_FAIL_LIMIT || probeFail2 >= PROBE_FAIL_LIMIT) {
      restartDevice();
    }
  }

  // Keep the two FIFOs aligned and latency low if their clocks drift
  while (sensor1.available() > 2) sensor1.nextSample();
  while (sensor2.available() > 2) sensor2.nextSample();

  while (sensor1.available() && sensor2.available()) {
    irRaw1  = sensor1.getFIFOIR();
    redRaw1 = sensor1.getFIFORed();
    irRaw2  = sensor2.getFIFOIR();
    redRaw2 = sensor2.getFIFORed();
    sensor1.nextSample();
    sensor2.nextSample();

    PpgSample s = { millis(), irRaw1, redRaw1, irRaw2, redRaw2 };
    xQueueSend(sampleQueue, &s, 0);   // never blocks
  }

  // Auto LED control every 50 ms
  if (millis() - lastAdjustTime > 50) {
    bool adjusted = false;

    adjustLed(irRaw1 >= FINGER_THRESHOLD, irRaw1, redRaw1, irPower1, redPower1, adjusted);
    adjustLed(irRaw2 >= FINGER_THRESHOLD, irRaw2, redRaw2, irPower2, redPower2, adjusted);

    if (adjusted) {
      sensor1.setPulseAmplitudeIR(irPower1);
      sensor1.setPulseAmplitudeRed(redPower1);
      sensor2.setPulseAmplitudeIR(irPower2);
      sensor2.setPulseAmplitudeRed(redPower2);
    }
    lastAdjustTime = millis();
  }

  delay(1);
}
