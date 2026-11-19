// ROSPug UNO Q bridge
// UART protocol verified against the original ROSPug Jetson SDK.

#include <Arduino_RouterBridge.h>
#include <string.h>

static constexpr uint32_t ROSPUG_BAUD = 500000;
static constexpr uint8_t MAX_PAYLOAD = 64;
static constexpr uint32_t LINK_TIMEOUT_MS = 1500;
static constexpr uint32_t MOTION_WATCHDOG_MS = 650;

enum ParseState : uint8_t {
  WAIT_AA,
  WAIT_55,
  READ_FUNCTION,
  READ_LENGTH,
  READ_PAYLOAD,
  READ_CRC
};

ParseState parseState = WAIT_AA;
uint8_t packetFunction = 0;
uint8_t packetLength = 0;
uint8_t packetPayload[MAX_PAYLOAD];
uint8_t payloadIndex = 0;
uint8_t runningCrc = 0;

uint32_t validFrames = 0;
uint32_t imuFrames = 0;
uint32_t crcErrors = 0;
uint32_t lastValidFrameMs = 0;
uint32_t lastImuFrameMs = 0;
float imuValues[6] = {0, 0, 0, 0, 0, 0};
int32_t batteryMillivolts = -1;

bool softwareArmed = false;
bool motionActive = false;
uint32_t lastMotionCommandMs = 0;
uint8_t gaitPreset = 0;
bool servoReplyReady = false;
uint8_t servoReplyId = 0;
int8_t servoReplyStatus = -127;
int16_t servoReplyPosition = 0;

uint8_t crc8Update(uint8_t crc, uint8_t value);
void sendPacket(uint8_t functionId, const uint8_t *payload, uint8_t length);
void serviceRospugUart();
void processPacket(uint8_t functionId, const uint8_t *payload, uint8_t length);
bool controllerOnline();
float clampFloat(float value, float minimum, float maximum);
void putFloat(uint8_t *destination, float value);
void sendStopFrame(float duration = 0.0f);
void sendGaitFrame(float overlapTime, float swingTime, float clearanceTime, float zClearance);
void sendPoseFrame(float x, float y, float z, float xShift, float roll, float pitch, float duration);
String rospugStatus();
bool rospugSetArmed(bool enabled);
String rospugReadServo(int servoId);
bool rospugStop();
bool rospugSetGait(int preset);
bool rospugHome();
bool rospugMove(float x, float y, float yaw);
bool rospugSetPose(float x, float y, float z, float xShift, float roll, float pitch, float duration);

void setup() {
  // Zephyr 0.90.0 routes RouterBridge internally, leaving Serial1 on D0/D1.
  // Start the external ROSPug UART before Bridge, matching the proven
  // Robodog Pro UNO Q initialization order.
  Serial1.begin(ROSPUG_BAUD);
  Bridge.begin();

  // Safe handlers execute in the sketch loop context.
  Bridge.provide_safe("rospug_status", rospugStatus);
  Bridge.provide_safe("rospug_set_armed", rospugSetArmed);
  Bridge.provide_safe("rospug_read_servo", rospugReadServo);
  Bridge.provide_safe("rospug_stop", rospugStop);
  Bridge.provide_safe("rospug_set_gait", rospugSetGait);
  Bridge.provide_safe("rospug_home", rospugHome);
  Bridge.provide_safe("rospug_move", rospugMove);
  Bridge.provide_safe("rospug_set_pose", rospugSetPose);
}

void loop() {
  serviceRospugUart();
  if (motionActive && (uint32_t)(millis() - lastMotionCommandMs) > MOTION_WATCHDOG_MS) {
    sendStopFrame();
    motionActive = false;
  }
}

uint8_t crc8Update(uint8_t crc, uint8_t value) {
  crc ^= value;
  for (uint8_t bit = 0; bit < 8; ++bit) {
    crc = (crc & 0x01) ? (uint8_t)((crc >> 1) ^ 0x8C) : (uint8_t)(crc >> 1);
  }
  return crc;
}

void sendPacket(uint8_t functionId, const uint8_t *payload, uint8_t length) {
  uint8_t crc = 0;
  crc = crc8Update(crc, functionId);
  crc = crc8Update(crc, length);

  Serial1.write((uint8_t)0xAA);
  Serial1.write((uint8_t)0x55);
  Serial1.write(functionId);
  Serial1.write(length);
  for (uint8_t i = 0; i < length; ++i) {
    Serial1.write(payload[i]);
    crc = crc8Update(crc, payload[i]);
  }
  Serial1.write(crc);
}

void serviceRospugUart() {
  while (Serial1.available() > 0) {
    const uint8_t value = (uint8_t)Serial1.read();

    switch (parseState) {
      case WAIT_AA:
        if (value == 0xAA) parseState = WAIT_55;
        break;

      case WAIT_55:
        if (value == 0x55) {
          parseState = READ_FUNCTION;
        } else {
          parseState = (value == 0xAA) ? WAIT_55 : WAIT_AA;
        }
        break;

      case READ_FUNCTION:
        packetFunction = value;
        runningCrc = crc8Update(0, value);
        parseState = READ_LENGTH;
        break;

      case READ_LENGTH:
        packetLength = value;
        runningCrc = crc8Update(runningCrc, value);
        payloadIndex = 0;
        if (packetLength > MAX_PAYLOAD) {
          parseState = WAIT_AA;
        } else {
          parseState = (packetLength == 0) ? READ_CRC : READ_PAYLOAD;
        }
        break;

      case READ_PAYLOAD:
        packetPayload[payloadIndex++] = value;
        runningCrc = crc8Update(runningCrc, value);
        if (payloadIndex >= packetLength) parseState = READ_CRC;
        break;

      case READ_CRC:
        if (value == runningCrc) {
          ++validFrames;
          lastValidFrameMs = millis();
          processPacket(packetFunction, packetPayload, packetLength);
        } else {
          ++crcErrors;
        }
        parseState = WAIT_AA;
        break;
    }
  }
}

void processPacket(uint8_t functionId, const uint8_t *payload, uint8_t length) {
  // Function 7 is the ROSPug's continuous six-float IMU report.
  if (functionId == 0x07 && length == 24) {
    memcpy(imuValues, payload, sizeof(imuValues));
    ++imuFrames;
    lastImuFrameMs = millis();
    return;
  }

  // Function 0, command 0x04 reports a little-endian battery reading.
  if (functionId == 0x00 && length >= 3 && payload[0] == 0x04) {
    batteryMillivolts = (int32_t)((uint16_t)payload[1] | ((uint16_t)payload[2] << 8));
    return;
  }

  // Function 5 response: id, command, success, signed 16-bit position.
  if (functionId == 0x05 && length >= 5 && payload[1] == 0x05) {
    servoReplyId = payload[0];
    servoReplyStatus = (int8_t)payload[2];
    servoReplyPosition = (int16_t)((uint16_t)payload[3] | ((uint16_t)payload[4] << 8));
    servoReplyReady = true;
  }
}

bool controllerOnline() {
  serviceRospugUart();
  return validFrames > 0 && (uint32_t)(millis() - lastValidFrameMs) < LINK_TIMEOUT_MS;
}

float clampFloat(float value, float minimum, float maximum) {
  if (value < minimum) return minimum;
  if (value > maximum) return maximum;
  return value;
}

void putFloat(uint8_t *destination, float value) {
  memcpy(destination, &value, sizeof(value));
}

void sendStopFrame(float duration) {
  uint8_t payload[5] = {0x0F, 0, 0, 0, 0};
  putFloat(&payload[1], duration);
  sendPacket(0x0B, payload, sizeof(payload));
}

void sendGaitFrame(float overlapTime, float swingTime, float clearanceTime, float zClearance) {
  uint8_t payload[17] = {0x0B};
  putFloat(&payload[1], overlapTime);
  putFloat(&payload[5], swingTime);
  putFloat(&payload[9], clearanceTime);
  putFloat(&payload[13], zClearance);
  sendPacket(0x0B, payload, sizeof(payload));
}

void sendPoseFrame(float x, float y, float z, float xShift, float roll, float pitch, float duration) {
  uint8_t payload[29] = {0x09};
  putFloat(&payload[1], x);
  putFloat(&payload[5], y);
  putFloat(&payload[9], z);
  putFloat(&payload[13], xShift);
  // The Hiwonder SDK packs pitch before roll even though the API lists roll first.
  putFloat(&payload[17], pitch);
  putFloat(&payload[21], roll);
  putFloat(&payload[25], duration);
  sendPacket(0x0B, payload, sizeof(payload));
}

String rospugStatus() {
  serviceRospugUart();
  const uint32_t now = millis();
  const bool connected = validFrames > 0 && (uint32_t)(now - lastValidFrameMs) < LINK_TIMEOUT_MS;

  String json;
  json.reserve(360);
  json += F("{\"connected\":");
  json += connected ? F("true") : F("false");
  json += F(",\"armed\":");
  json += softwareArmed ? F("true") : F("false");
  json += F(",\"moving\":");
  json += motionActive ? F("true") : F("false");
  json += F(",\"gait\":");
  json += (int)gaitPreset;
  json += F(",\"motion_watchdog_ms\":");
  json += MOTION_WATCHDOG_MS;
  json += F(",\"uptime_ms\":");
  json += now;
  json += F(",\"last_frame_age_ms\":");
  json += validFrames ? (uint32_t)(now - lastValidFrameMs) : 0;
  json += F(",\"frames\":");
  json += validFrames;
  json += F(",\"imu_frames\":");
  json += imuFrames;
  json += F(",\"crc_errors\":");
  json += crcErrors;
  json += F(",\"battery_mv\":");
  json += batteryMillivolts;
  json += F(",\"imu\":[");
  for (uint8_t i = 0; i < 6; ++i) {
    if (i) json += ',';
    json += String(imuValues[i], 5);
  }
  json += F("]}");
  return json;
}

bool rospugSetArmed(bool enabled) {
  if (enabled) {
    if (!controllerOnline()) {
      softwareArmed = false;
      return false;
    }

    // Verified ROSPug command: KINEMATICS / set robot type / ROSPug.
    // The controller may immediately move the legs into its initialized stance.
    const uint8_t payload[] = {0x15, 0x01};
    sendPacket(0x0B, payload, sizeof(payload));
    softwareArmed = true;
  } else {
    // Stop first, then engage the software interlock. Servo power remains live.
    if (controllerOnline()) sendStopFrame();
    motionActive = false;
    softwareArmed = false;
  }
  return softwareArmed;
}

String rospugReadServo(int servoId) {
  if (servoId < 1 || servoId > 253) {
    return String(F("{\"ok\":false,\"error\":\"Servo ID must be 1-253\"}"));
  }

  servoReplyReady = false;
  const uint8_t payload[] = {0x05, (uint8_t)servoId};
  sendPacket(0x05, payload, sizeof(payload));

  const uint32_t started = millis();
  while ((uint32_t)(millis() - started) < 300) {
    serviceRospugUart();
    if (servoReplyReady && servoReplyId == (uint8_t)servoId) {
      String json;
      json.reserve(100);
      json += F("{\"ok\":");
      json += (servoReplyStatus == 0) ? F("true") : F("false");
      json += F(",\"id\":");
      json += (int)servoReplyId;
      json += F(",\"status\":");
      json += (int)servoReplyStatus;
      json += F(",\"position\":");
      json += servoReplyPosition;
      json += '}';
      return json;
    }
    delay(1);
  }

  String json = F("{\"ok\":false,\"id\":");
  json += servoId;
  json += F(",\"error\":\"No reply within 300 ms\"}");
  return json;
}

bool rospugStop() {
  if (!controllerOnline()) return false;
  sendStopFrame(0.0f);
  motionActive = false;
  return true;
}

bool rospugSetGait(int preset) {
  if (!softwareArmed || !controllerOnline()) return false;

  switch (preset) {
    case 1:  // Hiwonder tutorial: Trot
      sendGaitFrame(0.20f, 0.18f, 0.02f, 0.04f);
      break;
    case 2:  // Hiwonder tutorial: Amble
      sendGaitFrame(0.20f, 0.15f, 0.05f, 0.04f);
      break;
    case 3:  // Hiwonder tutorial: Walk
      sendGaitFrame(0.24f, 0.12f, 0.05f, 0.04f);
      break;
    case 4:  // Fast Trot: shorter positive phases, zero diagonal delay
      sendGaitFrame(0.14f, 0.14f, 0.00f, 0.04f);
      break;
    case 5:  // Smooth Fast Trot: full stride with shorter, lower foot swing
      sendGaitFrame(0.14f, 0.13f, 0.00f, 0.03f);
      break;
    default:
      return false;
  }

  gaitPreset = (uint8_t)preset;
  return true;
}

bool rospugHome() {
  if (!softwareArmed || !controllerOnline()) return false;

  sendStopFrame(0.0f);
  motionActive = false;
  delay(10);

  if (gaitPreset == 0) {
    sendGaitFrame(0.20f, 0.18f, 0.02f, 0.04f);
    gaitPreset = 1;
    delay(10);
  }

  // Exact neutral pose used by Hiwonder's ROSPug motion-course examples.
  sendPoseFrame(0.0f, 0.0f, -0.13f, 0.01f, 0.0f, 0.0f, 0.5f);
  return true;
}

bool rospugMove(float x, float y, float yaw) {
  if (!softwareArmed || !controllerOnline()) return false;

  // Match the limits enforced by Hiwonder's ROS node.
  x = clampFloat(x, -0.10f, 0.10f);
  y = clampFloat(y, -0.10f, 0.10f);
  yaw = clampFloat(yaw, -0.40f, 0.40f);

  uint8_t payload[13] = {0x0D};
  putFloat(&payload[1], x);
  putFloat(&payload[5], y);
  putFloat(&payload[9], yaw);
  sendPacket(0x0B, payload, sizeof(payload));

  motionActive = (x != 0.0f || y != 0.0f || yaw != 0.0f);
  lastMotionCommandMs = millis();
  return true;
}

bool rospugSetPose(float x, float y, float z, float xShift, float roll, float pitch, float duration) {
  if (!softwareArmed || !controllerOnline()) return false;

  // Bounds mirror the ranges used by Hiwonder's factory gamepad node.
  x = clampFloat(x, -0.03f, 0.03f);
  y = clampFloat(y, -0.03f, 0.03f);
  z = clampFloat(z, -0.15f, -0.10f);
  xShift = clampFloat(xShift, -0.03f, 0.03f);
  roll = clampFloat(roll, -0.17f, 0.17f);
  pitch = clampFloat(pitch, -0.35f, 0.35f);
  duration = clampFloat(duration, 0.05f, 2.0f);

  if (motionActive) {
    sendStopFrame(0.0f);
    motionActive = false;
    delay(10);
  }
  sendPoseFrame(x, y, z, xShift, roll, pitch, duration);
  return true;
}
