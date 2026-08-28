#include <SPI.h>
#include "protocentralAds1292r.h"
#include "ecgRespirationAlgo.h"

// ============================================================
// ESP8266 PIN CONFIGURATION
// ============================================================

// ADS1292R control pins
#define ADS1292_DRDY_PIN     D1    // GPIO5
#define ADS1292_CS_PIN       D8    // GPIO15
#define ADS1292_START_PIN    D2    // GPIO4
#define ADS1292_PWDN_PIN     D0    // GPIO16

// Hardware SPI pins
// D5 = GPIO14 = SCK
// D6 = GPIO12 = MISO
// D7 = GPIO13 = MOSI


// ============================================================
// GLOBAL VARIABLES
// ============================================================

volatile uint8_t globalHeartRate = 0;
volatile uint8_t globalRespirationRate = 0;

int16_t ecgWaveBuff = 0;
int16_t ecgFilterout = 0;

int16_t resWaveBuff = 0;
int16_t respFilterout = 0;


// ============================================================
// ADS1292R OBJECTS
// ============================================================

ads1292r ADS1292R;

ecg_respiration_algorithm ECG_RESPIRATION_ALGORITHM;


// ============================================================
// SETUP
// ============================================================

void setup()
{
    // Give ESP8266 time to boot
    delay(2000);

    // --------------------------------------------------------
    // SERIAL
    // --------------------------------------------------------

    Serial.begin(115200);

    delay(500);

    Serial.println();
    Serial.println("======================================");
    Serial.println(" ADS1292R + ESP8266 ECG MONITOR");
    Serial.println("======================================");

    Serial.println();
    Serial.println("SPI Pins:");
    Serial.println("SCK  = D5 / GPIO14");
    Serial.println("MISO = D6 / GPIO12");
    Serial.println("MOSI = D7 / GPIO13");

    Serial.println();
    Serial.println("Control Pins:");
    Serial.println("CS    = D8 / GPIO15");
    Serial.println("DRDY  = D1 / GPIO5");
    Serial.println("START = D2 / GPIO4");
    Serial.println("PWDN  = D0 / GPIO16");

    Serial.println("--------------------------------------");


    // --------------------------------------------------------
    // SPI
    // --------------------------------------------------------

    SPI.begin();

    delay(100);


    // --------------------------------------------------------
    // PIN CONFIGURATION
    // --------------------------------------------------------

    pinMode(ADS1292_DRDY_PIN, INPUT);

    pinMode(ADS1292_CS_PIN, OUTPUT);

    pinMode(ADS1292_START_PIN, OUTPUT);

    pinMode(ADS1292_PWDN_PIN, OUTPUT);


    // --------------------------------------------------------
    // INITIAL PIN STATES
    // --------------------------------------------------------

    digitalWrite(ADS1292_CS_PIN, HIGH);

    digitalWrite(ADS1292_START_PIN, LOW);

    digitalWrite(ADS1292_PWDN_PIN, HIGH);

    delay(100);


    // --------------------------------------------------------
    // ADS1292R INITIALIZATION
    // --------------------------------------------------------

    Serial.println("Initializing ADS1292R...");

    ADS1292R.ads1292Init(
        ADS1292_CS_PIN,
        ADS1292_PWDN_PIN,
        ADS1292_START_PIN
    );

    delay(500);

    Serial.println("ADS1292R initialized.");

    Serial.println();
    Serial.println("Starting ECG acquisition...");
    Serial.println("--------------------------------------");

    // CSV HEADER
    Serial.println("RawECG,FilteredECG,BPM");

    Serial.println("--------------------------------------");
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
    ads1292OutputValues ecgRespirationValues;


    // --------------------------------------------------------
    // GET ADS1292R SAMPLE
    // --------------------------------------------------------

    boolean ret =
        ADS1292R.getAds1292EcgAndRespirationSamples(
            ADS1292_DRDY_PIN,
            ADS1292_CS_PIN,
            &ecgRespirationValues
        );


    // --------------------------------------------------------
    // NEW SAMPLE AVAILABLE
    // --------------------------------------------------------

    if (ret == true)
    {
        // ----------------------------------------------------
        // RAW ECG
        // ----------------------------------------------------

        ecgWaveBuff =
            (int16_t)(
                ecgRespirationValues.sDaqVals[1] >> 8
            );


        // ----------------------------------------------------
        // RESPIRATION RAW
        // ----------------------------------------------------

        resWaveBuff =
            (int16_t)(
                ecgRespirationValues.sresultTempResp >> 8
            );


        // ----------------------------------------------------
        // LEAD-OFF CHECK
        // ----------------------------------------------------

        if (ecgRespirationValues.leadoffDetected == false)
        {
            // ------------------------------------------------
            // ECG FILTER
            // ------------------------------------------------

            ECG_RESPIRATION_ALGORITHM.ECG_ProcessCurrSample(
                &ecgWaveBuff,
                &ecgFilterout
            );


            // ------------------------------------------------
            // QRS / HEART RATE
            // ------------------------------------------------

            ECG_RESPIRATION_ALGORITHM.QRS_Algorithm_Interface(
                ecgFilterout,
                &globalHeartRate
            );


            // ------------------------------------------------
            // SEND DATA
            //
            // Format:
            //
            // RawECG,FilteredECG,BPM
            //
            // Example:
            //
            // 1250,43,76
            // ------------------------------------------------

            Serial.print(ecgWaveBuff);
            Serial.print(",");

            Serial.print(ecgFilterout);
            Serial.print(",");

            Serial.println(globalHeartRate);
        }
        else
        {
            // ------------------------------------------------
            // LEADS DISCONNECTED
            // ------------------------------------------------

            ecgFilterout = 0;

            respFilterout = 0;

            globalHeartRate = 0;

            globalRespirationRate = 0;


            // Send zero values

            Serial.println("0,0,0");
        }
    }
}