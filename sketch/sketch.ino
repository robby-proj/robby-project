void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  Serial.println("UNO Q sketch ready.");
}

void loop() {
  delay(1000);
}