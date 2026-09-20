const int sensorPins[] = {4, 5, 6, 7, 15};
const int sensorCount = sizeof(sensorPins) / sizeof(sensorPins[0]);

void setup() {
    Serial.begin(1000000);

    analogReadResolution(12);

    for (int i = 0; i < sensorCount; i++) {
        analogSetPinAttenuation(sensorPins[i], ADC_11db);
    }
}

void loop() {
    for (int i = 0; i < sensorCount; i++) {
        Serial.print("Sensor");
        Serial.print(i + 1);
        Serial.print(":");
        Serial.print(analogRead(sensorPins[i]));
        if (i < sensorCount - 1) {
            Serial.print(",");
        }
    }
    Serial.println();

    delayMicroseconds(1000);
}
