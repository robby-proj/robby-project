#include <Arduino.h>

HardwareSerial &DOG = Serial1;

#define CMD_BATTERY 0x01
#define CMD_MOVE_X  0x30
#define CMD_MOVE_Y  0x31
#define CMD_TURN    0x32
#define CMD_ACTION  0x3E

void printHex(uint8_t b) {
  if (b < 16) Serial.print("0");
  Serial.print(b, HEX);
  Serial.print(" ");
}

void sendFrame(uint8_t mode, uint8_t cmd, uint8_t value) {
  uint8_t pkt[9] = {
    0x55, 0x00, 0x09,
    mode,
    cmd,
    value,
    0x00,
    0x00,
    0xAA
  };

  pkt[6] = 255 - ((pkt[2] + pkt[3] + pkt[4] + pkt[5]) % 256);

  DOG.write(pkt, 9);

  Serial.print("TX: ");
  for (int i = 0; i < 9; i++) printHex(pkt[i]);
  Serial.println();
}

void readBattery() {
  sendFrame(0x02, CMD_BATTERY, 0x01);
}

void stopDog() {
  sendFrame(0x01, CMD_MOVE_X, 128);
  sendFrame(0x01, CMD_MOVE_Y, 128);
  sendFrame(0x01, CMD_TURN,   128);
}

void action(uint8_t id) {
  sendFrame(0x01, CMD_ACTION, id);
}

void setup() {
  Serial.begin(115200);
  DOG.begin(115200);

  delay(2000);

  Serial.println();
  Serial.println("DOGZILLA UNO Q CONTROLLER READY");
  Serial.println("Movement: w/s/a/d/q/e/x");
  Serial.println("Battery: b");
  Serial.println("Actions:");
  Serial.println("1-9 = action 1-9");
  Serial.println("0 = action 10");
  Serial.println("p = action 11");
  Serial.println("o = action 12");
  Serial.println("i = action 13");
  Serial.println("u = action 14");
  Serial.println("y = action 15");
  Serial.println("t = action 16");
  Serial.println("r = reset");
}

void loop() {
  while (DOG.available()) {
    printHex(DOG.read());
  }

  if (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') return;

    Serial.print("\nCMD: ");
    Serial.println(c);

    switch (c) {
      case 'w': sendFrame(0x01, CMD_MOVE_X, 180); break;
      case 's': sendFrame(0x01, CMD_MOVE_X, 70);  break;
      case 'a': sendFrame(0x01, CMD_MOVE_Y, 180); break;
      case 'd': sendFrame(0x01, CMD_MOVE_Y, 70);  break;
      case 'q': sendFrame(0x01, CMD_TURN, 180);   break;
      case 'e': sendFrame(0x01, CMD_TURN, 70);    break;
      case 'x': stopDog(); break;

      case 'b': readBattery(); break;

      case '1': action(1); break;
      case '2': action(2); break;
      case '3': action(3); break;
      case '4': action(4); break;
      case '5': action(5); break;
      case '6': action(6); break;
      case '7': action(7); break;
      case '8': action(8); break;
      case '9': action(9); break;
      case '0': action(10); break;
      case 'p': action(11); break;
      case 'o': action(12); break;
      case 'i': action(13); break;
      case 'u': action(14); break;
      case 'y': action(15); break;
      case 't': action(16); break;

      case 'r': action(255); break;

      default:
        Serial.println("Unknown command");
        break;
    }
  }
}
