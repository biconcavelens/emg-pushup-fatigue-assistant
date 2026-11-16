// EMG logger for Advancer Technologies Muscle Sensor v3 on Arduino Uno.
//
// The v3 board outputs an analog *envelope* (rectified + smoothed), not raw EMG,
// so a modest sample rate is enough. We sample A0 at 500 Hz and stream
// "<arduino_micros>,<adc 0..1023>\n" over serial at 115200 baud.
//
// Wiring (see README): sensor SIG -> A0, sensor GND -> Arduino GND.
// Power the sensor from its own +/-9V batteries, NOT from the Arduino 5V pin.

const unsigned long SAMPLE_PERIOD_US = 2000;  // 500 Hz
unsigned long nextSample;

void setup() {
  Serial.begin(115200);
  nextSample = micros();
}

void loop() {
  unsigned long now = micros();
  if ((long)(now - nextSample) >= 0) {
    nextSample += SAMPLE_PERIOD_US;
    int v = analogRead(A0);
    Serial.print(now);
    Serial.print(',');
    Serial.println(v);
  }
}
