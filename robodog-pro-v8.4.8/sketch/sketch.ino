
// ============================================================================
// Robodog Pro v8.4.8 - UNO Q MCU sketch
// DOGZILLA UART + built-in LED matrix + external Modulino LED Matrix faces
// + OEM OLED + Modulino Movement + Buzzer
// + MCU LED3/LED4 RGB LEDs
// + UPDATED: full sketch with EMA motion filtering + sensitivity bridge + diagnostics + stable 0x3C eyes
// ============================================================================
// Robot eyes OLED:
//   - Connected on Wire1, same Qwiic/Modulino I2C bus.
//   - Current fixed address is 0x3C.
//   - If you resolder the OLED to 0x3D, change EYES_ADDR_FIXED below to 0x3D.
//   - Eyes are intentionally simple: two stable hollow white rings, no pupils.
//   - The two rings move together in parallel and keep a constant distance apart.
//   - App Lab may auto-start Python without reflashing the MCU. If you only
//     see the older blinking-eye sketch, re-upload the MCU sketch and then run
//     eyes_demo or eyes_mode IDs from Python/Bridge.
//   - Modulino Movement now drives reactive faces: long rest = sleep, small
//     motion = idle/scan, strong vibration = thick X-eyes + alarm tune, impacts = X/alert.
// ============================================================================

#include <Wire.h>
#include <Arduino_RouterBridge.h>
#include <Arduino_Modulino.h>
#include "Modulino_LED_Matrix.h"
#include <Arduino_LED_Matrix.h>

HardwareSerial &DOG = Serial1;
#define CMD_BATTERY 0x01
#define CMD_PERFORM 0x03
#define CMD_FIRMWARE_VER 0x07
#define CMD_GAIT_TYPE 0x09
#define CMD_UNLOAD 0x20
#define CMD_LOAD 0x20
#define CMD_MOVE_X 0x30
#define CMD_MOVE_Y 0x31
#define CMD_TURN 0x32
#define CMD_TRANSLATE 0x33
#define CMD_ATTITUDE 0x36
#define CMD_PERIODIC_ROT 0x39
#define CMD_MARK_TIME 0x3C
#define CMD_MOVE_MODE 0x3D
#define CMD_ACTION 0x3E
#define CMD_IMU 0x61
#define CMD_ROLL 0x62
#define CMD_PITCH 0x63
#define CMD_YAW 0x64
#define CMD_PERIODIC_TRAN 0x80

static const uint8_t SOFT_OFF_PIN = 2;
static const bool SOFT_OFF_ACTIVE_HIGH = true;
static const uint32_t SOFT_OFF_DEBOUNCE_MS = 40;
// The UNO Q Zephyr Arduino core lacks the conventional GPIO interrupt linker symbols.
// Soft-off therefore uses deterministic polling with
// debounce and one-shot latching instead of an ISR.
static bool softOffRawState = false;
static bool softOffStableState = false;
static bool softOffHandled = false;
static uint32_t softOffLastChangeMs = 0;

// OEM 128x32 OLED on Wire.
#define OLED_W 128
#define OLED_H 32
#define OLED_PAGES (OLED_H / 8)
static uint8_t oledAddr = 0x3C;
static bool oledPresent = false;
static uint8_t oledBuf[OLED_W * OLED_PAGES];
static String oledIP = "", oledStatus = "ROBODOG V8.4.6", oledDetection = "BOOT";
static int oledBatteryPct = -1;
static uint32_t oledLastUpdate = 0;

// NEW robot eyes OLED on Wire1.
#define EYES_W 128
#define EYES_H 64
#define EYES_PAGES (EYES_H / 8)
#define EYES_ADDR_FIXED 0x3C
#define EYES_BOOT_DEMO_MS 42000UL
static uint8_t eyesAddr = EYES_ADDR_FIXED;
static bool eyesPresent = false;
static bool eyesBusFault = false;
static uint32_t eyesLastRecoveryAttemptMs = 0;
static uint32_t eyesFlushOkCount = 0;
static uint32_t eyesFlushErrorCount = 0;
static uint8_t eyesConsecutiveErrors = 0;
static uint8_t eyesBuf[EYES_W * EYES_PAGES];

enum EyesMode : uint8_t {
  EYES_BOOT_DEMO = 0,
  EYES_IDLE = 1,
  EYES_LEFT = 2,
  EYES_RIGHT = 3,
  EYES_HAPPY = 4,
  EYES_ANGRY = 5,
  EYES_SLEEP = 6,
  EYES_SCAN = 7,
  EYES_ALERT = 8,
  EYES_PERSON = 9,
  EYES_CLOSED = 10,
  EYES_SQUINT = 11,
  EYES_SURPRISED = 12,
  EYES_SUSPICIOUS = 13,
  EYES_HEARTS = 14,
  EYES_DIZZY = 15,
  EYES_SAD = 16
};
static EyesMode eyesMode = EYES_BOOT_DEMO;
static uint32_t eyesLastMs = 0, eyesDemoStartMs = 0, eyesNextBlinkMs = 0;
static bool eyesFrameDirty = true;
static uint16_t eyesFrame = 0;
static uint32_t eyesWakeUntilMs = 0;
static int8_t eyesLookX = 0, eyesLookY = 0;
static bool eyesBlinking = false;
static uint8_t eyesBlinkPhase = 0;

// Matrix.
ArduinoLEDMatrix matrix;
#define MX_PIXELS 104
static char mxText[96] = "ROBODOG V8.4.6";
static int mxX = 13;
static uint32_t mxLast = 0, mxStep = 18, mxLastCommandMs = 0, mxLastIdleSwapMs = 0, mxScrollMs = 0;
static bool mxScrollComplete = true;
static uint8_t mxIdleSlot = 0;
static int mxClients = 0;

// Modulino.
ModulinoMovement movement;
ModulinoBuzzer buzzer;
ModulinoLEDMatrix faceMatrix;
static bool modulinoOk = false, buzzerPresent = false;
static bool faceMatrixPresent = false;
static float accel_x = 0, accel_y = 0, accel_z = 0, vibration_value = 0, vibration_peak = 0;
static uint32_t vibrationPeakMs = 0;
static const uint32_t PEAK_HOLD_MS = 750;
static volatile int personDetectedFlag = 0;

// External 12x8 Modulino LED Matrix face policy.
// Priority: aggressive acceleration > low battery > active/funny > idle/happy.
enum FaceExpression : uint8_t {
  FACE_HAPPY = 0,
  FACE_SAD = 1,
  FACE_SERIOUS = 2,
  FACE_FUNNY = 3
};
static FaceExpression faceExpression = FACE_HAPPY;
static int8_t faceRenderedFrame = -1;
static bool faceMoving = false;
static bool faceSearching = false;
static bool faceBatteryLow = false;
static uint32_t facePresetUntilMs = 0;
static uint32_t faceSeriousUntilMs = 0;
static const uint8_t FACE_LOW_BATTERY_PCT = 20;
static const uint8_t FACE_LOW_BATTERY_RECOVERY_PCT = 25;
static const float FACE_AGGRESSIVE_DELTA_G = 0.90f;
static const uint32_t FACE_SERIOUS_HOLD_MS = 2500UL;
static const uint32_t FACE_PRESET_HOLD_MS = 6500UL;
static const uint32_t FACE_FUNNY_FRAME_MS = 350UL;

// Column-major frames for the Modulino's default monochromatic-vertical mode.
// These are the three primary expressions requested. FACE_FUNNY_WINK is only
// an animation frame used with FACE_HAPPY while the dog is active.
static const uint8_t FACE_FRAME_HAPPY[12] = {
  0x00, 0x00, 0x16, 0x26, 0x40, 0x40, 0x40, 0x40, 0x26, 0x16, 0x00, 0x00
};
static const uint8_t FACE_FRAME_SAD[12] = {
  0x00, 0x00, 0x46, 0x26, 0x10, 0x10, 0x10, 0x10, 0x26, 0x46, 0x00, 0x00
};
static const uint8_t FACE_FRAME_SERIOUS[12] = {
  0x00, 0x01, 0x05, 0x46, 0x40, 0x40, 0x40, 0x40, 0x46, 0x05, 0x01, 0x00
};
static const uint8_t FACE_FRAME_FUNNY_WINK[12] = {
  0x00, 0x00, 0x16, 0x26, 0x40, 0x40, 0xC0, 0xC0, 0x24, 0x14, 0x04, 0x00
};

// -----------------------------------------------------------------------------
// Motion sensitivity policy
// -----------------------------------------------------------------------------
// mode 0 = off       no local MCU impact alert, still reports accel/vibration
// mode 1 = low       hard impacts only, recommended default for walking robot
// mode 2 = medium    moderate impacts
// mode 3 = high      sensitive bench/demo mode only
//
// IMPORTANT:
// - This policy controls MCU-local alert/eyes/buzzer behavior only
// - It does not have to stop the robot
// - MCU local stop is disabled by default so resting/noise cannot randomly STOP
// -----------------------------------------------------------------------------
static uint8_t motionSensitivityMode = 1;
static bool mcuLocalImpactStopEnabled = false;
static float impactCandidateG = 2.40f;
static uint8_t impactConfirmSamples = 8;
static uint32_t impactCooldownMs = 3500;
static uint8_t impactConfirmCnt = 0;
static uint32_t impactCooldownUntil = 0;

// v8.4.5 reporting filters restored from the safest v8.4.2 behavior.
// The MCU local impact detector can look at raw instant movement, but Python
// receives smoothed acceleration and smoothed vibration so one hard footfall
// cannot look like a real collision or tilt event.
static const float VIB_REPORT_EMA_ALPHA = 0.40f;
static const float ACC_REPORT_EMA_ALPHA = 0.50f;
static float vibReportEma = 0.0f;
static float rptX = 0.0f;
static float rptY = 0.0f;
static float rptZ = 0.0f;
static bool rptSeeded = false;
static uint32_t diagMotionReads = 0;
static uint32_t diagMotionRejected = 0;
static uint32_t movementLastUpdateMs = 0;
static const uint32_t MOVEMENT_UPDATE_INTERVAL_MS = 50;
static const bool EYES_WIRE1_ENABLED = true;
static uint8_t motionStableSamples = 0;
static bool motionFeedbackArmed = false;
static uint32_t motionFeedbackCooldownUntilMs = 0;
static const uint8_t MOTION_REARM_SAMPLES = 12;
static const uint32_t MOTION_FEEDBACK_COOLDOWN_MS = 5000;
static bool motionPreviousValid = false;
static float motionPreviousX = 0.0f;
static float motionPreviousY = 0.0f;
static float motionPreviousZ = 0.0f;
static float motionGesturePeak = 0.0f;
static uint32_t diagMotionActive = 0;
static uint32_t diagLastHeartbeatMs = 0;
#define ROBODOG_SAFE_MONITOR_DIAG 1

// Eyes reactive motion policy.  Kept on the MCU so it works even if Python is
// busy.  Bridge can enable/disable it with eyes_motion_reactive(1/0).
static bool eyesMotionReactive = false;
static uint32_t eyesMotionLastActiveMs = 0;
static uint32_t eyesMotionFaceLockUntilMs = 0;
static uint32_t eyesMotionLastFaceChangeMs = 0;
static EyesMode eyesMotionLastMode = EYES_BOOT_DEMO;

// RGB LED status layer. Active low: LOW = on, HIGH = off.
struct Rgb01 {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};
static const Rgb01 C_OFF = { 0, 0, 0 }, C_RED = { 1, 0, 0 }, C_GREEN = { 0, 1, 0 }, C_BLUE = { 0, 0, 1 }, C_AMBER = { 1, 1, 0 }, C_CYAN = { 0, 1, 1 }, C_PURPLE = { 1, 0, 1 }, C_WHITE = { 1, 1, 1 };
static Rgb01 led3Steady = C_OFF, led4Steady = C_OFF, led3BlinkColor = C_OFF, led4BlinkColor = C_OFF;
static uint8_t ledBlinkToggles = 0;
static bool ledBlinkOn = false;
static uint32_t ledBlinkNextMs = 0;
static uint16_t ledBlinkOnMs = 110, ledBlinkOffMs = 110;
static inline void rgbWritePin(uint8_t pin, uint8_t on) {
  digitalWrite(pin, on ? LOW : HIGH);
}
static inline void led3Physical(Rgb01 c) {
  rgbWritePin(LED3_R, c.r);
  rgbWritePin(LED3_G, c.g);
  rgbWritePin(LED3_B, c.b);
}
static inline void led4Physical(Rgb01 c) {
  rgbWritePin(LED4_R, c.r);
  rgbWritePin(LED4_G, c.g);
  rgbWritePin(LED4_B, c.b);
}
static void ledsApplySteady() {
  led3Physical(led3Steady);
  led4Physical(led4Steady);
}
static void ledsSetSplit(Rgb01 a, Rgb01 b) {
  led3Steady = a;
  led4Steady = b;
  ledBlinkToggles = 0;
  ledBlinkOn = false;
  ledsApplySteady();
}
static void ledsSet(Rgb01 c) {
  ledsSetSplit(c, c);
}
static void ledsBlink(Rgb01 a, Rgb01 b, uint8_t times, uint16_t onMs = 110, uint16_t offMs = 110) {
  led3BlinkColor = a;
  led4BlinkColor = b;
  ledBlinkToggles = max((uint8_t)1, (uint8_t)(times * 2));
  ledBlinkOn = false;
  ledBlinkNextMs = 0;
  ledBlinkOnMs = onMs;
  ledBlinkOffMs = offMs;
}
static void ledsTick() {
  if (!ledBlinkToggles || millis() < ledBlinkNextMs) return;
  ledBlinkOn = !ledBlinkOn;
  if (ledBlinkOn) {
    led3Physical(led3BlinkColor);
    led4Physical(led4BlinkColor);
    ledBlinkNextMs = millis() + ledBlinkOnMs;
  } else {
    ledsApplySteady();
    ledBlinkNextMs = millis() + ledBlinkOffMs;
  }
  if (--ledBlinkToggles == 0) ledsApplySteady();
}
static void ledsBootSelfTest() {
  const Rgb01 seq[] = { C_RED, C_GREEN, C_BLUE, C_AMBER, C_CYAN, C_PURPLE, C_WHITE, C_OFF };
  for (uint8_t i = 0; i < sizeof(seq) / sizeof(seq[0]); i++) {
    ledsSet(seq[i]);
    delay(110);
  }
}
static void ledsReadyManual() {
  ledsSetSplit(C_GREEN, C_OFF);
}
static void ledsAutonomous() {
  ledsSetSplit(C_BLUE, C_CYAN);
}
static void ledsSearching() {
  ledsSetSplit(C_AMBER, C_OFF);
}
static void ledsDetected() {
  ledsSetSplit(C_CYAN, C_WHITE);
}
static void ledsAlert() {
  ledsSetSplit(C_RED, C_OFF);
}
static void ledsLowBattery() {
  ledsSetSplit(C_PURPLE, C_OFF);
}
static void ledsSleep() {
  ledsSetSplit(C_OFF, C_OFF);
}
static void ledsWallHit() {
  ledsBlink(C_RED, C_RED, 4, 100, 100);
}
static void ledsHardStop() {
  ledsBlink(C_RED, C_RED, 6, 80, 80);
}
static void ledsBridgeLost() {
  ledsBlink(C_AMBER, C_AMBER, 3, 120, 120);
}
void led_status(int id) {
  switch (id) {
    case 1: ledsReadyManual(); break;
    case 2: ledsAutonomous(); break;
    case 3: ledsSearching(); break;
    case 4: ledsDetected(); break;
    case 5: ledsAlert(); break;
    case 6: ledsLowBattery(); break;
    case 7: ledsSleep(); break;
    case 10: ledsWallHit(); break;
    case 11: ledsHardStop(); break;
    default: ledsSleep(); break;
  }
}

// Small 5x7 font used by matrix and OEM OLED.
static void glyph5x7(char c, uint8_t o[5]) {
  memset(o, 0, 5);
  if (c >= 'a' && c <= 'z') c -= 32;
  switch (c) {
    case ' ': break;
    case '-': o[0] = o[1] = o[2] = o[3] = o[4] = 0x08; break;
    case '.': o[1] = o[2] = 0x60; break;
    case ':': o[1] = o[2] = 0x36; break;
    case '%':
      o[0] = 0x23;
      o[1] = 0x13;
      o[2] = 0x08;
      o[3] = 0x64;
      o[4] = 0x62;
      break;
    case '0':
      o[0] = 0x3E;
      o[1] = 0x51;
      o[2] = 0x49;
      o[3] = 0x45;
      o[4] = 0x3E;
      break;
    case '1':
      o[1] = 0x42;
      o[2] = 0x7F;
      o[3] = 0x40;
      break;
    case '2':
      o[0] = 0x42;
      o[1] = 0x61;
      o[2] = 0x51;
      o[3] = 0x49;
      o[4] = 0x46;
      break;
    case '3':
      o[0] = 0x21;
      o[1] = 0x41;
      o[2] = 0x45;
      o[3] = 0x4B;
      o[4] = 0x31;
      break;
    case '4':
      o[0] = 0x18;
      o[1] = 0x14;
      o[2] = 0x12;
      o[3] = 0x7F;
      o[4] = 0x10;
      break;
    case '5':
      o[0] = 0x27;
      o[1] = 0x45;
      o[2] = 0x45;
      o[3] = 0x45;
      o[4] = 0x39;
      break;
    case '6':
      o[0] = 0x3C;
      o[1] = 0x4A;
      o[2] = 0x49;
      o[3] = 0x49;
      o[4] = 0x30;
      break;
    case '7':
      o[0] = 0x01;
      o[1] = 0x71;
      o[2] = 0x09;
      o[3] = 0x05;
      o[4] = 0x03;
      break;
    case '8':
      o[0] = 0x36;
      o[1] = 0x49;
      o[2] = 0x49;
      o[3] = 0x49;
      o[4] = 0x36;
      break;
    case '9':
      o[0] = 0x06;
      o[1] = 0x49;
      o[2] = 0x49;
      o[3] = 0x29;
      o[4] = 0x1E;
      break;
    case 'A':
      o[0] = 0x7E;
      o[1] = o[2] = o[3] = 0x11;
      o[4] = 0x7E;
      break;
    case 'B':
      o[0] = 0x7F;
      o[1] = o[2] = o[3] = 0x49;
      o[4] = 0x36;
      break;
    case 'C':
      o[0] = 0x3E;
      o[1] = o[2] = o[3] = 0x41;
      o[4] = 0x22;
      break;
    case 'D':
      o[0] = 0x7F;
      o[1] = o[2] = 0x41;
      o[3] = 0x22;
      o[4] = 0x1C;
      break;
    case 'E':
      o[0] = 0x7F;
      o[1] = o[2] = o[3] = 0x49;
      o[4] = 0x41;
      break;
    case 'F':
      o[0] = 0x7F;
      o[1] = o[2] = 0x09;
      o[3] = o[4] = 0x01;
      break;
    case 'G':
      o[0] = 0x3E;
      o[1] = o[2] = 0x41;
      o[3] = 0x51;
      o[4] = 0x32;
      break;
    case 'H':
      o[0] = 0x7F;
      o[1] = o[2] = o[3] = 0x08;
      o[4] = 0x7F;
      break;
    case 'I':
      o[1] = 0x41;
      o[2] = 0x7F;
      o[3] = 0x41;
      break;
    case 'J':
      o[0] = 0x20;
      o[1] = 0x40;
      o[2] = 0x41;
      o[3] = 0x3F;
      o[4] = 0x01;
      break;
    case 'K':
      o[0] = 0x7F;
      o[1] = 0x08;
      o[2] = 0x14;
      o[3] = 0x22;
      o[4] = 0x41;
      break;
    case 'L':
      o[0] = 0x7F;
      o[1] = o[2] = o[3] = o[4] = 0x40;
      break;
    case 'M':
      o[0] = 0x7F;
      o[1] = 0x02;
      o[2] = 0x04;
      o[3] = 0x02;
      o[4] = 0x7F;
      break;
    case 'N':
      o[0] = 0x7F;
      o[1] = 0x04;
      o[2] = 0x08;
      o[3] = 0x10;
      o[4] = 0x7F;
      break;
    case 'O':
      o[0] = 0x3E;
      o[1] = o[2] = o[3] = 0x41;
      o[4] = 0x3E;
      break;
    case 'P':
      o[0] = 0x7F;
      o[1] = o[2] = o[3] = 0x09;
      o[4] = 0x06;
      break;
    case 'Q':
      o[0] = 0x3E;
      o[1] = 0x41;
      o[2] = 0x51;
      o[3] = 0x21;
      o[4] = 0x5E;
      break;
    case 'R':
      o[0] = 0x7F;
      o[1] = 0x09;
      o[2] = 0x19;
      o[3] = 0x29;
      o[4] = 0x46;
      break;
    case 'S':
      o[0] = 0x46;
      o[1] = o[2] = o[3] = 0x49;
      o[4] = 0x31;
      break;
    case 'T':
      o[0] = o[1] = o[3] = o[4] = 0x01;
      o[2] = 0x7F;
      break;
    case 'U':
      o[0] = 0x3F;
      o[1] = o[2] = o[3] = 0x40;
      o[4] = 0x3F;
      break;
    case 'V':
      o[0] = 0x1F;
      o[1] = 0x20;
      o[2] = 0x40;
      o[3] = 0x20;
      o[4] = 0x1F;
      break;
    case 'W':
      o[0] = 0x7F;
      o[1] = 0x20;
      o[2] = 0x18;
      o[3] = 0x20;
      o[4] = 0x7F;
      break;
    case 'X':
      o[0] = 0x63;
      o[1] = 0x14;
      o[2] = 0x08;
      o[3] = 0x14;
      o[4] = 0x63;
      break;
    case 'Y':
      o[0] = 0x03;
      o[1] = 0x04;
      o[2] = 0x78;
      o[3] = 0x04;
      o[4] = 0x03;
      break;
    case 'Z':
      o[0] = 0x61;
      o[1] = 0x51;
      o[2] = 0x49;
      o[3] = 0x45;
      o[4] = 0x43;
      break;
    default:
      o[0] = 0x7F;
      o[1] = 0x41;
      o[2] = 0x5D;
      o[3] = 0x41;
      o[4] = 0x7F;
  }
}

// OEM OLED.
static void oledCmd(uint8_t c) {
  Wire.beginTransmission(oledAddr);
  Wire.write(0);
  Wire.write(c);
  Wire.endTransmission();
}
static bool probeOLED() {
  Wire.begin();
  Wire.setClock(400000);
  for (uint8_t a = 0x3C; a <= 0x3D; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      oledAddr = a;
      return true;
    }
  }
  return false;
}
static bool oledInit() {
  if (!probeOLED()) return false;
  const uint8_t seq[] = { 0xAE, 0xD5, 0x80, 0xA8, 0x1F, 0xD3, 0x00, 0x40, 0x8D, 0x14, 0x20, 0x00, 0xA1, 0xC8, 0xDA, 0x02, 0x81, 0x8F, 0xD9, 0xF1, 0xDB, 0x40, 0xA4, 0xA6, 0x2E, 0xAF };
  for (uint8_t c : seq) oledCmd(c);
  return true;
}
static void oledFlush() {
  oledCmd(0x21);
  oledCmd(0);
  oledCmd(127);
  oledCmd(0x22);
  oledCmd(0);
  oledCmd(OLED_PAGES - 1);
  for (uint16_t i = 0; i < sizeof(oledBuf); i += 16) {
    Wire.beginTransmission(oledAddr);
    Wire.write(0x40);
    for (uint8_t j = 0; j < 16; j++) Wire.write(oledBuf[i + j]);
    Wire.endTransmission();
  }
}
static void oledText(int x, int page, const char *t) {
  while (*t && x + 5 < 128) {
    uint8_t g[5];
    glyph5x7(*t++, g);
    for (int i = 0; i < 5; i++) oledBuf[page * 128 + x + i] = g[i];
    x += 6;
  }
}
static void oledRedraw() {
  if (!oledPresent) return;
  memset(oledBuf, 0, sizeof(oledBuf));
  oledText(0, 0, oledStatus.c_str());
  char b[28];
  snprintf(b, sizeof(b), "IP:%s", oledIP.length() ? oledIP.c_str() : "--");
  oledText(0, 1, b);
  if (oledBatteryPct >= 0) snprintf(b, sizeof(b), "BAT:%d%% V:%d", oledBatteryPct, (int)(vibration_value * 100.0f));
  else snprintf(b, sizeof(b), "BAT:-- V:%d", (int)(vibration_value * 100.0f));
  oledText(0, 2, b);
  oledText(0, 3, oledDetection.c_str());
  oledFlush();
}

// Matrix.
static void mxPixel(uint8_t *f, int x, int y) {
  if (x >= 0 && x < 13 && y >= 0 && y < 8) f[y * 13 + x] = 1;
}
static void mxTick() {
  if (millis() - mxLast < mxStep) return;
  mxLast = millis();
  uint8_t f[104];
  memset(f, 0, 104);
  int x = mxX;
  for (const char *p = mxText; *p; p++, x += 6) {
    uint8_t g[5];
    glyph5x7(*p, g);
    for (int c = 0; c < 5; c++)
      for (int r = 0; r < 7; r++)
        if (g[c] & (1 << r)) mxPixel(f, x + c, r);
  }
  matrix.draw(f);
  if (--mxX < -(int)strlen(mxText) * 6 - 2) {
    mxX = 13;
    mxScrollComplete = true;
  }
}
static void mxSet(const char *t, uint32_t speed = 18) {
  strncpy(mxText, t, sizeof(mxText) - 1);
  mxText[sizeof(mxText) - 1] = 0;
  mxX = 13;
  mxStep = speed;
  mxLastCommandMs = millis();
  mxScrollMs = ((uint32_t)strlen(mxText) * 6 + 23) * mxStep;
  mxScrollComplete = false;
}
static void mxIdleTick() {
  if (millis() - mxLastCommandMs < 900) return;
  if (!mxScrollComplete) return;
  if (millis() - mxLastIdleSwapMs < 250) return;
  mxLastIdleSwapMs = millis();
  char b[48];
  if (mxClients > 0) {
    snprintf(b, sizeof(b), (mxIdleSlot++ % 2) == 0 ? "LINK" : "CLIENT %d", mxClients);
  } else {
    switch (mxIdleSlot++ % 4) {
      case 0: snprintf(b, sizeof(b), "ROBODOG V8.4.6"); break;
      case 1: snprintf(b, sizeof(b), "IP %s", oledIP.length() ? oledIP.c_str() : "WAIT"); break;
      case 2:
        if (oledBatteryPct >= 0) snprintf(b, sizeof(b), "BAT %d%%", oledBatteryPct);
        else snprintf(b, sizeof(b), "BAT WAIT");
        break;
      default: snprintf(b, sizeof(b), "READY"); break;
    }
  }
  mxSet(b, 18);
}

// External Modulino LED Matrix face scheduler.
static bool faceDeadlineActive(uint32_t deadline) {
  return deadline != 0 && (int32_t)(deadline - millis()) > 0;
}
static void faceRender(const uint8_t (&frame)[12], int8_t frameId, bool force = false) {
  if (!faceMatrixPresent) return;
  if (!force && faceRenderedFrame == frameId) return;
  faceMatrix.setFrame(frame);
  faceRenderedFrame = frameId;
}
static void faceTick(bool force = false) {
  if (!faceMatrixPresent) return;

  const bool seriousActive = faceDeadlineActive(faceSeriousUntilMs);
  if (!seriousActive) faceSeriousUntilMs = 0;
  if (seriousActive) {
    faceExpression = FACE_SERIOUS;
    faceRender(FACE_FRAME_SERIOUS, 2, force);
    return;
  }
  if (faceBatteryLow) {
    faceExpression = FACE_SAD;
    faceRender(FACE_FRAME_SAD, 1, force);
    return;
  }

  const bool presetActive = faceDeadlineActive(facePresetUntilMs);
  if (!presetActive) facePresetUntilMs = 0;
  const bool active = faceMoving || faceSearching || presetActive;
  if (active) {
    faceExpression = FACE_FUNNY;
    const bool wink = ((millis() / FACE_FUNNY_FRAME_MS) & 1U) != 0;
    if (wink) faceRender(FACE_FRAME_FUNNY_WINK, 3, force);
    else faceRender(FACE_FRAME_HAPPY, 0, force);
    return;
  }

  faceExpression = FACE_HAPPY;
  faceRender(FACE_FRAME_HAPPY, 0, force);
}
static void faceSetMoving(bool moving) {
  faceMoving = moving;
  faceTick(true);
}
static void faceSetSearching(bool searching) {
  faceSearching = searching;
  faceTick(true);
}
static void faceStartPreset(uint32_t holdMs = FACE_PRESET_HOLD_MS) {
  facePresetUntilMs = millis() + holdMs;
  faceTick(true);
}
static void faceNoteBattery(int pct) {
  if (pct < 0 || pct > 100) return;
  if (pct < FACE_LOW_BATTERY_PCT) faceBatteryLow = true;
  else if (pct >= FACE_LOW_BATTERY_RECOVERY_PCT) faceBatteryLow = false;
  faceTick(true);
}
static void faceNoteAggressiveAcceleration(float sampleDeltaG) {
  if (sampleDeltaG < FACE_AGGRESSIVE_DELTA_G) return;
  faceSeriousUntilMs = millis() + FACE_SERIOUS_HOLD_MS;
  faceTick(true);
}
int face_matrix_present() {
  return faceMatrixPresent ? 1 : 0;
}
int face_matrix_expression() {
  return (int)faceExpression;
}
int face_matrix_battery_low() {
  return faceBatteryLow ? 1 : 0;
}
int face_matrix_searching_set(int searching) {
  faceSetSearching(searching != 0);
  return faceSearching ? 1 : 0;
}
int face_matrix_aggressive_threshold_mg() {
  return (int)(FACE_AGGRESSIVE_DELTA_G * 1000.0f);
}

// New robot eyes Wire1 SSD1306 driver and animations.
static bool eyesCmdChecked(uint8_t c) {
  Wire1.beginTransmission(eyesAddr);
  Wire1.write(0x00);
  Wire1.write(c);
  return Wire1.endTransmission() == 0;
}
static void eyesCmd(uint8_t c) {
  (void)eyesCmdChecked(c);
}
static bool probeEyesOLED() {
  Wire1.beginTransmission(EYES_ADDR_FIXED);
  if (Wire1.endTransmission() == 0) {
    eyesAddr = EYES_ADDR_FIXED;
    return true;
  }
  return false;
}
static bool eyesInit() {
  if (!probeEyesOLED()) return false;
  const uint8_t seq[] = { 0xAE, 0xD5, 0x80, 0xA8, 0x3F, 0xD3, 0x00, 0x40, 0x8D, 0x14, 0x20, 0x00, 0xA1, 0xC8, 0xDA, 0x12, 0x81, 0xCF, 0xD9, 0xF1, 0xDB, 0x40, 0xA4, 0xA6, 0x2E, 0xAF };
  for (uint8_t i = 0; i < sizeof(seq); i++) eyesCmd(seq[i]);
  memset(eyesBuf, 0, sizeof(eyesBuf));
  eyesDemoStartMs = millis();
  eyesNextBlinkMs = millis() + 1800;
  return true;
}
static bool eyesFlushAttempt() {
  if (!eyesPresent) return false;

  // Program the complete SSD1306 window before every frame. A vibration-induced
  // I2C error can otherwise leave the controller mid-page, which appears as a
  // bright horizontal line that drifts as subsequent eye frames are written.
  if (!eyesCmdChecked(0x21) || !eyesCmdChecked(0) || !eyesCmdChecked(127) ||
      !eyesCmdChecked(0x22) || !eyesCmdChecked(0) || !eyesCmdChecked(7)) {
    return false;
  }

  for (uint16_t i = 0; i < sizeof(eyesBuf); i += 16) {
    Wire1.beginTransmission(eyesAddr);
    Wire1.write(0x40);
    for (uint8_t j = 0; j < 16; j++) Wire1.write(eyesBuf[i + j]);
    if (Wire1.endTransmission() != 0) return false;
  }
  return true;
}
static void eyesFlush() {
  if (!eyesPresent) return;

  if (eyesFlushAttempt()) {
    eyesFlushOkCount++;
    eyesConsecutiveErrors = 0;
    eyesBusFault = false;
    return;
  }

  eyesFlushErrorCount++;
  if (eyesConsecutiveErrors < 255) eyesConsecutiveErrors++;
  eyesBusFault = true;

  // One immediate retry is safe and repairs a transient incomplete frame.
  delay(2);
  if (eyesFlushAttempt()) {
    eyesFlushOkCount++;
    eyesConsecutiveErrors = 0;
    eyesBusFault = false;
    return;
  }

  eyesFlushErrorCount++;
  if (eyesConsecutiveErrors < 255) eyesConsecutiveErrors++;
}
static void eyesRecoverBusIfNeeded() {
  if (!eyesBusFault || millis() - eyesLastRecoveryAttemptMs < 500UL) return;
  eyesLastRecoveryAttemptMs = millis();

  // The shared Wire1 harness is physically exposed to vibration. Re-assert the
  // configured bus speed, reprobe the fixed OLED address, reinitialize the
  // controller, and overwrite every display byte with black before resuming.
  Wire1.setClock(400000);
  if (!probeEyesOLED()) {
    eyesPresent = false;
    return;
  }

  eyesPresent = true;
  if (!eyesInit()) {
    eyesPresent = false;
    return;
  }

  memset(eyesBuf, 0, sizeof(eyesBuf));
  if (eyesFlushAttempt()) {
    eyesBusFault = false;
    eyesConsecutiveErrors = 0;
    eyesFlushOkCount++;
    eyesFrame = 0;
    eyesLastMs = 0;
    Monitor.println("[eyes] Wire1 recovery completed; framebuffer cleared");
  }
}
static inline void eyesClear() {
  memset(eyesBuf, 0, sizeof(eyesBuf));
}
static inline void eyesPixel(int x, int y, bool on = true) {
  if (x < 0 || x >= 128 || y < 0 || y >= 64) return;
  uint16_t idx = x + (y / 8) * 128;
  uint8_t bit = 1 << (y & 7);
  if (on) eyesBuf[idx] |= bit;
  else eyesBuf[idx] &= ~bit;
}
static void eyesHLine(int x0, int x1, int y, bool on = true) {
  if (y < 0 || y >= 64) return;
  if (x0 > x1) {
    int t = x0;
    x0 = x1;
    x1 = t;
  }
  x0 = max(0, x0);
  x1 = min(127, x1);
  for (int x = x0; x <= x1; x++) eyesPixel(x, y, on);
}
static void eyesVLine(int x, int y0, int y1, bool on = true) {
  if (x < 0 || x >= 128) return;
  if (y0 > y1) {
    int t = y0;
    y0 = y1;
    y1 = t;
  }
  y0 = max(0, y0);
  y1 = min(63, y1);
  for (int y = y0; y <= y1; y++) eyesPixel(x, y, on);
}
static void eyesRect(int x, int y, int w, int h) {
  eyesHLine(x, x + w - 1, y);
  eyesHLine(x, x + w - 1, y + h - 1);
  eyesVLine(x, y, y + h - 1);
  eyesVLine(x + w - 1, y, y + h - 1);
}
static void eyesFillRect(int x, int y, int w, int h, bool on = true) {
  for (int yy = y; yy < y + h; yy++) eyesHLine(x, x + w - 1, yy, on);
}
static void eyesFillCircle(int cx, int cy, int r, bool on = true) {
  for (int y = -r; y <= r; y++) {
    int xs = (int)sqrtf((float)(r * r - y * y));
    eyesHLine(cx - xs, cx + xs, cy + y, on);
  }
}
static void eyesTinyZ(int x, int y) {
  eyesHLine(x, x + 8, y);
  eyesHLine(x, x + 8, y + 8);
  for (int i = 0; i <= 8; i++) eyesPixel(x + 8 - i, y + i);
}
static void eyesSparkles(uint8_t seed) {
  for (uint8_t i = 0; i < 12; i++) {
    int x = (seed * 17 + i * 23) % 128;
    int y = 4 + ((seed * 11 + i * 19) % 56);
    eyesPixel(x, y);
    eyesPixel(x - 1, y);
    eyesPixel(x + 1, y);
    eyesPixel(x, y - 1);
    eyesPixel(x, y + 1);
  }
}
static void eyesRoundEye(int cx, int cy, int rx, int ry, int px, int py, bool angry = false) {
  ry = constrain(ry, 2, 22);
  for (int y = -ry; y <= ry; y++) {
    float yf = (float)y / (float)ry;
    int xs = (int)((float)rx * sqrtf(max(0.0f, 1.0f - yf * yf)));
    eyesHLine(cx - xs, cx + xs, cy + y, true);
  }
  if (ry > 6) {
    eyesFillCircle(cx + px, cy + py, 7, false);
    eyesFillCircle(cx + px - 2, cy + py - 2, 2, true);
  }
  if (angry) {
    for (int i = 0; i < 12; i++) {
      if (cx < 64) eyesHLine(cx - rx - 2, cx - rx + 5 + i * 3, cy - ry + i, false);
      else eyesHLine(cx + rx - 5 - i * 3, cx + rx + 2, cy - ry + i, false);
    }
  }
}
static void eyesSleepEye(int cx, int cy) {
  eyesHLine(cx - 34, cx + 34, cy);
  eyesHLine(cx - 30, cx + 30, cy + 1);
  eyesHLine(cx - 22, cx + 22, cy + 2);
  eyesHLine(cx - 14, cx + 14, cy + 3);
}
static void eyesHappyEye(int cx, int cy) {
  eyesHLine(cx - 36, cx + 36, cy + 4);
  eyesHLine(cx - 32, cx + 32, cy + 3);
  eyesHLine(cx - 25, cx + 25, cy + 2);
  eyesHLine(cx - 16, cx + 16, cy + 1);
  eyesHLine(cx - 7, cx + 7, cy);
}
static void eyesScanBars(uint8_t pos) {
  eyesClear();
  eyesRect(2, 2, 124, 60);
  for (int x = 8; x < 120; x += 12) {
    int h = 8 + ((x + pos) % 20);
    eyesVLine(x, 32 - h, 32 + h);
  }
  eyesFillRect(4 + (pos % 120), 4, 4, 56, true);
}
// ---------------------------------------------------------------------------
// Robot dog ring-eye renderer.
// Design goal: two stable hollow white rings, no pupils, moving together as a
// fixed-distance pair.  Oversize rings intentionally clip at the OLED edges.
// ---------------------------------------------------------------------------
static void eyesEllipseOutline(int cx, int cy, int rx, int ry, uint8_t thick = 3) {
  if (rx < 2 || ry < 2) return;
  for (int y = -ry - thick; y <= ry + thick; y++) {
    for (int x = -rx - thick; x <= rx + thick; x++) {
      float nx = (float)x / (float)rx;
      float ny = (float)y / (float)ry;
      float v = nx * nx + ny * ny;
      // Hollow ring band around the ellipse edge.  Larger thickness is used so
      // the ring remains bright and visible on the small SSD1306.
      if (v >= 0.78f && v <= 1.22f) eyesPixel(cx + x, cy + y, true);
    }
  }
}
static void eyesRingPair(int dx, int dy, int rx, int ry, uint8_t thick = 3) {
  const int BASE_LX = 30;
  const int BASE_RX = 98;
  const int BASE_Y = 32;
  int lx = BASE_LX + dx, rx0 = BASE_RX + dx, cy = BASE_Y + dy;
  eyesEllipseOutline(lx, cy, rx, ry, thick);
  eyesEllipseOutline(rx0, cy, rx, ry, thick);
  // For the first few seconds after waking from sleep, add tiny solid pupils.
  // They disappear automatically so the normal look returns to clean hollow rings.
  if (millis() < eyesWakeUntilMs) {
    eyesFillCircle(lx, cy, 2, true);
    eyesFillCircle(rx0, cy, 2, true);
  }
}
static void eyesBigX(int cx, int cy, uint8_t thick = 4) {
  // Thick X-eye for strong vibration / confused robot-dog state.
  // Uses only pixels and clipping, so it is safe when the X sits near edges.
  for (int t = -((int)thick / 2); t <= ((int)thick / 2); t++) {
    for (int i = -25; i <= 25; i++) {
      eyesPixel(cx + i, cy + i + t, true);
      eyesPixel(cx + i, cy - i + t, true);
      eyesPixel(cx + i + t, cy + i, true);
      eyesPixel(cx + i + t, cy - i, true);
    }
  }
}
static void eyesXPair(int dx, int dy, uint8_t thick = 4) {
  eyesBigX(30 + dx, 32 + dy, thick);
  eyesBigX(98 + dx, 32 + dy, thick);
}
static void eyesScrollingZs() {
  // Multi-row sleeping Z field.
  // Three staggered rows move upward at slightly different offsets so the
  // sleeping mode looks alive instead of showing only one diagonal trail.
  // Zs are allowed to clip at the top/bottom through eyesPixel().
  const int row0X[5] = { 26, 40, 54, 68, 82 };
  const int row1X[5] = { 14, 28, 42, 56, 70 };
  const int row2X[5] = { 34, 48, 62, 76, 90 };

  for (uint8_t i = 0; i < 5; i++) {
    int y0 = 72 - (((int)eyesFrame + i * 16) % 96);
    int y1 = 76 - (((int)eyesFrame + i * 16 + 32) % 100);
    int y2 = 80 - (((int)eyesFrame + i * 16 + 64) % 104);
    eyesTinyZ(row0X[i], y0);
    eyesTinyZ(row1X[i], y1);
    eyesTinyZ(row2X[i], y2);
  }
}
static void eyesRingMotion(uint16_t frame, int ampX, int ampY, int &dx, int &dy) {
  float t = (float)frame * 0.075f;
  dx = (int)(sinf(t) * ampX);
  dy = (int)(cosf(t * 0.73f) * ampY);
}
static void eyesClosedRingPair(int dx, int dy, bool sleepy = false) {
  int cy = 32 + dy;
  int lx0 = 30 + dx;
  int rx0 = 98 + dx;
  for (int k = 0; k < 4; k++) {
    eyesHLine(lx0 - 28, lx0 + 28, cy + k, true);
    eyesHLine(rx0 - 28, rx0 + 28, cy + k, true);
  }
  if (sleepy) { eyesScrollingZs(); }
}
static void eyesAngryBrows(int dx, int dy) {
  int cy = 32 + dy;
  int lx0 = 30 + dx;
  int rx0 = 98 + dx;
  // Left brow slopes downward toward center. Right brow mirrors it.
  for (int i = 0; i < 20; i++) {
    eyesHLine(lx0 - 40 + i, lx0 - 10 + i, cy - 29 + i / 2, true);
    eyesHLine(rx0 + 10 - i, rx0 + 40 - i, cy - 20 - i / 2, true);
  }
}
static void eyesSadBrows(int dx, int dy) {
  int cy = 32 + dy;
  int lx0 = 30 + dx;
  int rx0 = 98 + dx;
  for (int i = 0; i < 18; i++) {
    eyesHLine(lx0 - 35 + i, lx0 - 6 + i, cy - 20 - i / 2, true);
    eyesHLine(rx0 + 6 - i, rx0 + 35 - i, cy - 29 + i / 2, true);
  }
}
static void eyesScanSweep(uint16_t frame) {
  // Search mode intentionally moves only the two eye rings. Earlier builds
  // drew a two-pixel vertical scanner bar here. That bar was mistaken for an
  // OLED/I2C corruption artifact because it swept across the moving circles.
  int dx, dy;
  eyesRingMotion(frame, 22, 7, dx, dy);
  eyesRingPair(dx, dy, 26, 26, 2);
}
static void eyesPersonFace(uint16_t frame) {
  // Unique detection face: centered double rings plus a pulsing target reticle.
  // This remains visually distinct from idle, scan, alert, and surprised modes.
  int pulse = 2 + ((frame / 4) % 4);
  eyesRingPair(0, 0, 23 + pulse, 23 + pulse, 3);

  // Four target-lock corner brackets
  eyesHLine(3, 20, 4);  eyesVLine(3, 4, 17);
  eyesHLine(107, 124, 4); eyesVLine(124, 4, 17);
  eyesHLine(3, 20, 59); eyesVLine(3, 46, 59);
  eyesHLine(107, 124, 59); eyesVLine(124, 46, 59);

  // Center lock marks and alternating sparkles
  eyesHLine(58, 70, 32);
  eyesVLine(64, 26, 38);
  if (((frame / 3) & 1U) == 0U) eyesSparkles((uint8_t)frame);
}
static void eyesDrawFace(EyesMode mode, uint16_t frame) {
  eyesClear();

  int dx = 0, dy = 0;
  int rx = 26;
  int ry = 26;
  uint8_t thick = 2;

  // Default idle: both rings drift together, always same spacing apart.
  eyesRingMotion(frame, 16, 10, dx, dy);

  if (mode == EYES_LEFT) {
    dx = -18;
    dy = 0;
  } else if (mode == EYES_RIGHT) {
    dx = 18;
    dy = 0;
  } else if (mode == EYES_SCAN) {
    eyesScanSweep(frame);
    return;
  } else if (mode == EYES_PERSON) {
    eyesPersonFace(frame);
    return;
  } else if (mode == EYES_ALERT) {
    eyesRingMotion(frame, 8, 4, dx, dy);
    rx = 26;
    ry = 26;
    thick = 3;
  } else if (mode == EYES_SURPRISED) {
    rx = 30;
    ry = 30;
    thick = 3;
    dx = 0;
    dy = 0;
  } else if (mode == EYES_SQUINT) {
    eyesClosedRingPair(dx, dy, false);
    return;
  } else if (mode == EYES_CLOSED) {
    eyesClosedRingPair(0, 0, false);
    return;
  } else if (mode == EYES_SLEEP) {
    eyesClosedRingPair(0, 4, true);
    return;
  } else if (mode == EYES_HAPPY) {
    eyesClosedRingPair(dx, dy, false);
    // Small cheek sparkles, still no pupils.
    eyesSparkles((uint8_t)frame);
    return;
  } else if (mode == EYES_DIZZY) {
    // Strong vibration / confused state: replace rings with thick X-eyes.
    eyesRingMotion(frame, 10, 5, dx, dy);
    eyesXPair(dx, dy, 5);
    return;
  }

  eyesRingPair(dx, dy, rx, ry, thick);

  if (mode == EYES_SAD) { /* no eyebrows: keep clean ring-only face */
  }

  // EYES_HEARTS is intentionally repurposed as an excited pulsing ring mode
  // so the visual language stays ring-only as requested.
  if (mode == EYES_HEARTS) {
    eyesSparkles((uint8_t)frame);
  }
}
static const char *eyesName(EyesMode m) {
  switch (m) {
    case EYES_BOOT_DEMO: return "DEMO";
    case EYES_IDLE: return "IDLE";
    case EYES_LEFT: return "LEFT";
    case EYES_RIGHT: return "RIGHT";
    case EYES_HAPPY: return "HAPPY";
    case EYES_ANGRY: return "ANGRY_RING";
    case EYES_SLEEP: return "SLEEP";
    case EYES_SCAN: return "SCAN";
    case EYES_ALERT: return "ALERT";
    case EYES_PERSON: return "PERSON";
    case EYES_CLOSED: return "CLOSED";
    case EYES_DIZZY: return "X_EYES";
    default: return "OTHER";
  }
}
static void eyesSetMode(EyesMode m) {
  EyesMode prev = eyesMode;
  if (prev == m) return;
  if (prev == EYES_SLEEP && m != EYES_SLEEP) eyesWakeUntilMs = millis() + 3000;
  eyesMode = m;
  eyesFrame = 0;
  eyesBlinking = false;
  eyesBlinkPhase = 0;
  eyesLastMs = 0;
  eyesFrameDirty = true;
  Monitor.print("[eyes] mode=");
  Monitor.println(eyesName(m));
}
void eyes_mode(int id) {
  switch (id) {
    case 0:
      eyesDemoStartMs = millis();
      eyesSetMode(EYES_BOOT_DEMO);
      break;
    case 1: eyesSetMode(EYES_IDLE); break;
    case 2: eyesSetMode(EYES_LEFT); break;
    case 3: eyesSetMode(EYES_RIGHT); break;
    case 4: eyesSetMode(EYES_HAPPY); break;
    case 5: eyesSetMode(EYES_ANGRY); break;
    case 6: eyesSetMode(EYES_SLEEP); break;
    case 7: eyesSetMode(EYES_SCAN); break;
    case 8: eyesSetMode(EYES_ALERT); break;
    case 9: eyesSetMode(EYES_PERSON); break;
    case 10: eyesSetMode(EYES_CLOSED); break;
    case 11: eyesSetMode(EYES_SQUINT); break;
    case 12: eyesSetMode(EYES_SURPRISED); break;
    case 13: eyesSetMode(EYES_SUSPICIOUS); break;
    case 14: eyesSetMode(EYES_HEARTS); break;
    case 15: eyesSetMode(EYES_DIZZY); break;
    case 16: eyesSetMode(EYES_SAD); break;
    default: eyesSetMode(EYES_IDLE); break;
  }
}
void eyes_demo() {
  eyesDemoStartMs = millis();
  eyesSetMode(EYES_BOOT_DEMO);
}
int eyes_present() {
  return eyesPresent ? 1 : 0;
}
int eyes_addr() {
  return eyesPresent ? eyesAddr : -1;
}
int eyes_motion_reactive(int enable) {
  eyesMotionReactive = (enable != 0);
  return eyesMotionReactive ? 1 : 0;
}
static void buzzSchedule(const uint16_t notes[][3], uint8_t count);
static void eyesMotionTone(EyesMode mode) {
  // Keep MCU audio useful but not annoying.
  // Sleep: one soft lullaby chirp max every 30 seconds.
  // Alert / X-eyes / person: short tones with wider cooldowns.
  // Idle and scan are intentionally silent because motion can call them often.
  static uint32_t lastAnyToneMs = 0;
  static uint32_t lastSleepToneMs = 0;
  static uint32_t lastAlertToneMs = 0;
  static uint32_t lastXToneMs = 0;
  static uint32_t lastPersonToneMs = 0;
  uint32_t now = millis();

  static const uint16_t sleep[][3] = { { 520, 70, 35 }, { 390, 90, 35 }, { 260, 140, 0 } };
  static const uint16_t alert[][3] = { { 1700, 45, 20 }, { 2150, 65, 0 } };
  static const uint16_t xeyes[][3] = { { 240, 110, 35 }, { 240, 110, 35 }, { 180, 220, 0 } };
  static const uint16_t person[][3] = { { 1300, 55, 25 }, { 1900, 70, 0 } };

  if (mode == EYES_SLEEP) {
    if (now - lastSleepToneMs >= 30000UL) {
      lastSleepToneMs = now;
      lastAnyToneMs = now;
      buzzSchedule(sleep, 3);
    }
    return;
  }
  if (mode == EYES_DIZZY) {
    if (now - lastXToneMs >= 8000UL && now - lastAnyToneMs >= 1200UL) {
      lastXToneMs = now;
      lastAnyToneMs = now;
      buzzSchedule(xeyes, 3);
    }
    return;
  }
  if (mode == EYES_ALERT) {
    if (now - lastAlertToneMs >= 10000UL && now - lastAnyToneMs >= 1200UL) {
      lastAlertToneMs = now;
      lastAnyToneMs = now;
      buzzSchedule(alert, 2);
    }
    return;
  }
  if (mode == EYES_PERSON) {
    if (now - lastPersonToneMs >= 12000UL && now - lastAnyToneMs >= 1200UL) {
      lastPersonToneMs = now;
      lastAnyToneMs = now;
      buzzSchedule(person, 2);
    }
    return;
  }
}
static void eyesMotionSet(EyesMode mode, uint32_t holdMs = 0) {
  if (!eyesPresent || !eyesMotionReactive) return;
  uint32_t now = millis();

  // Do not re-apply or re-log the same steady mode. This fixes repeated
  // [eyes] mode=SLEEP and [eyes-motion] SLEEP spam while resting.
  if (mode == eyesMode && holdMs == 0) return;

  if (now < eyesMotionFaceLockUntilMs && mode != EYES_ALERT && mode != EYES_DIZZY) return;
  if (mode == eyesMotionLastMode && now - eyesMotionLastFaceChangeMs < 5000UL) return;

  eyesSetMode(mode);
  eyesMotionTone(mode);
  Monitor.print("[eyes-motion] mode=");
  Monitor.print(eyesName(mode));
  Monitor.print(" hold_ms=");
  Monitor.println(holdMs);
  eyesMotionLastMode = mode;
  eyesMotionLastFaceChangeMs = now;
  if (holdMs > 0) eyesMotionFaceLockUntilMs = now + holdMs;
}
static void eyesMotionReact(float instant, float peak, float mag) {
  if (!eyesPresent || !eyesMotionReactive) return;
  uint32_t now = millis();

  // Treat any clear movement above the existing deadband as activity.
  if (instant > 0.08f || fabsf(mag - 1.0f) > 0.18f) eyesMotionLastActiveMs = now;

  // Hard impact has highest priority, but keep it short so angry/vibration can follow.
  if (peak >= 1.80f) {
    eyesMotionSet(EYES_DIZZY, 5000);
    return;
  }

  // Lots of vibration means thick X-eyes for ten seconds.  This uses instant
  // acceleration change, not the decaying peak, so it only triggers on real motion.
  if (instant >= 0.55f) {
    eyesMotionSet(EYES_DIZZY, 10000);
    return;
  }

  // Medium shake/pickup/tap gets an alert pulse.
  if (instant >= 0.25f) {
    eyesMotionSet(EYES_ALERT, 3500);
    return;
  }

  // Small but ongoing movement makes the eyes scan/roam.
  if (instant >= 0.10f) {
    eyesMotionSet(EYES_IDLE, 1500);
    return;
  }

  // Completely resting for a while makes sleepy/closed eyes.  Keep the delay
  // long enough that normal pauses do not immediately sleep the face.
  if (now - eyesMotionLastActiveMs > 30000UL) {
    eyesMotionSet(EYES_SLEEP, 0);
    return;
  }

  // Otherwise idle drifting rings.
  if (now > eyesMotionFaceLockUntilMs && now - eyesMotionLastFaceChangeMs > 5000UL) { eyesMotionSet(EYES_IDLE, 0); }
}
static void eyesTick() {
  eyesRecoverBusIfNeeded();
  if (!eyesPresent || !eyesFrameDirty) return;
  eyesFrameDirty = false;
  eyesLastMs = millis();
  eyesFrame++;
  eyesBlinking = false;
  eyesBlinkPhase = 0;
  eyesDrawFace(eyesMode, 0);
  eyesFlush();
}

// -----------------------------------------------------------------------------
// Motion sensitivity Bridge controls
// -----------------------------------------------------------------------------
static void applyMotionSensitivityDefaults() {
  switch (motionSensitivityMode) {
    case 0:
      impactCandidateG = 99.0f;
      impactConfirmSamples = 255;
      impactCooldownMs = 10000;
      mcuLocalImpactStopEnabled = false;
      break;
    case 1:
      impactCandidateG = 2.40f;
      impactConfirmSamples = 8;
      impactCooldownMs = 3500;
      mcuLocalImpactStopEnabled = false;
      break;
    case 2:
      impactCandidateG = 2.00f;
      impactConfirmSamples = 6;
      impactCooldownMs = 2500;
      mcuLocalImpactStopEnabled = false;
      break;
    case 3:
      impactCandidateG = 1.60f;
      impactConfirmSamples = 4;
      impactCooldownMs = 1500;
      mcuLocalImpactStopEnabled = false;
      break;
    default:
      motionSensitivityMode = 1;
      impactCandidateG = 2.40f;
      impactConfirmSamples = 8;
      impactCooldownMs = 3500;
      mcuLocalImpactStopEnabled = false;
      break;
  }
}

int motion_sensitivity(int mode) {
  mode = constrain(mode, 0, 3);
  motionSensitivityMode = (uint8_t)mode;
  applyMotionSensitivityDefaults();
  impactConfirmCnt = 0;
  impactCooldownUntil = millis() + 1000;
  Monitor.print("[motion-policy] sensitivity=");
  Monitor.print(motionSensitivityMode);
  Monitor.print(" threshold_g=");
  Monitor.print(impactCandidateG, 2);
  Monitor.print(" confirm=");
  Monitor.print(impactConfirmSamples);
  Monitor.print(" cooldown_ms=");
  Monitor.print(impactCooldownMs);
  Monitor.print(" local_stop=");
  Monitor.println(mcuLocalImpactStopEnabled ? 1 : 0);
  return motionSensitivityMode;
}

int motion_sensitivity_get() {
  return motionSensitivityMode;
}
float motion_threshold_g() {
  return impactCandidateG;
}

int motion_local_stop_enable(int enable) {
  mcuLocalImpactStopEnabled = (enable != 0);
  Monitor.print("[motion-policy] local impact stop enabled=");
  Monitor.println(mcuLocalImpactStopEnabled ? 1 : 0);
  return mcuLocalImpactStopEnabled ? 1 : 0;
}

float movement_mag() {
  if (!modulinoOk) return -1.0f;
  return sqrtf(accel_x * accel_x + accel_y * accel_y + accel_z * accel_z);
}


// Forward declaration because robot actions use the centralized sound map.
void sound_event(int id);

// DOGZILLA UART bridge commands.
void sendFrame(uint8_t mode, uint8_t cmd, uint8_t value) {
  uint8_t p[9] = { 0x55, 0, 9, mode, cmd, value, 0, 0, 0xAA };
  p[6] = 255 - ((p[2] + p[3] + p[4] + p[5]) % 256);
  DOG.write(p, 9);
}
void sendFrameN(uint8_t mode, uint8_t cmd, const uint8_t *d, uint8_t n) {
  if (n > 24) n = 24;
  uint8_t p[32];
  uint8_t len = 8 + n;
  p[0] = 0x55;
  p[1] = 0;
  p[2] = len;
  p[3] = mode;
  p[4] = cmd;
  uint16_t s = len + mode + cmd;
  for (uint8_t i = 0; i < n; i++) {
    p[5 + i] = d[i];
    s += d[i];
  }
  p[5 + n] = 255 - (s % 256);
  p[6 + n] = 0;
  p[7 + n] = 0xAA;
  DOG.write(p, len);
}
void dog_forward() {
  sendFrame(1, CMD_MOVE_X, 180);
  faceSetMoving(true);
  mxSet("FORWARD");
  oledStatus = "MOVE FWD";
}
void dog_backward() {
  sendFrame(1, CMD_MOVE_X, 70);
  faceSetMoving(true);
  mxSet("BACKWARD");
  oledStatus = "MOVE BACK";
}
void dog_left() {
  sendFrame(1, CMD_MOVE_Y, 180);
  faceSetMoving(true);
  mxSet("LEFT");
  oledStatus = "LEFT";
}
void dog_right() {
  sendFrame(1, CMD_MOVE_Y, 70);
  faceSetMoving(true);
  mxSet("RIGHT");
  oledStatus = "RIGHT";
}
void dog_turn_left() {
  sendFrame(1, CMD_TURN, 180);
  faceSetMoving(true);
  mxSet("TURN LEFT");
  oledStatus = "TURN L";
}
void dog_turn_right() {
  sendFrame(1, CMD_TURN, 70);
  faceSetMoving(true);
  mxSet("TURN RIGHT");
  oledStatus = "TURN R";
}
void dog_stop() {
  sendFrame(1, CMD_MOVE_X, 128);
  sendFrame(1, CMD_MOVE_Y, 128);
  sendFrame(1, CMD_TURN, 128);
  faceSetMoving(false);
  mxSet("STOP");
  oledStatus = "STOP";
}
static int readScalar(uint8_t cmd) {
  while (DOG.available()) DOG.read();
  sendFrame(2, cmd, 1);
  uint8_t b[9], n = 0;
  uint32_t t = millis();
  while (millis() - t < 100 && n < 9)
    if (DOG.available()) b[n++] = DOG.read();
  return (n == 9 && b[0] == 0x55 && b[8] == 0xAA) ? b[5] : -1;
}
int dog_battery() {
  int p = readScalar(CMD_BATTERY);
  if (p >= 0 && p <= 100) {
    oledBatteryPct = p;
    faceNoteBattery(p);
  }
  return p;
}
static float readFloat(uint8_t cmd) {
  while (DOG.available()) DOG.read();
  sendFrame(2, cmd, 4);
  uint8_t b[13], n = 0;
  uint32_t t = millis();
  while (millis() - t < 120 && n < 13)
    if (DOG.available()) b[n++] = DOG.read();
  if (n < 13) return NAN;
  uint8_t f[4] = { b[8], b[7], b[6], b[5] };
  float v;
  memcpy(&v, f, 4);
  return v;
}
float dog_read_roll_f() {
  return readFloat(CMD_ROLL);
}
float dog_read_pitch_f() {
  return readFloat(CMD_PITCH);
}
float dog_read_yaw_f() {
  return readFloat(CMD_YAW);
}
int dog_read_roll() {
  float v = dog_read_roll_f();
  return isnan(v) ? -1 : (int)v;
}
int dog_read_pitch() {
  float v = dog_read_pitch_f();
  return isnan(v) ? -1 : (int)v;
}
int dog_read_yaw() {
  float v = dog_read_yaw_f();
  return isnan(v) ? -1 : (int)v;
}
String dog_firmware_version() {
  while (DOG.available()) DOG.read();
  uint8_t sum = 255 - ((0x09 + 0x02 + CMD_FIRMWARE_VER + 10) % 256);
  uint8_t tx[9] = { 0x55, 0x00, 0x09, 0x02, CMD_FIRMWARE_VER, 10, sum, 0x00, 0xAA };
  DOG.write(tx, 9);
  uint32_t t0 = millis();
  uint8_t buf[24];
  uint8_t n = 0;
  while (millis() - t0 < 150 && n < 19) {
    if (DOG.available()) buf[n++] = DOG.read();
  }
  if (n < 15 || buf[0] != 0x55) return String("?");
  char result[11];
  uint8_t k = 0;
  for (uint8_t i = 0; i < 10; i++) {
    char c = (char)buf[7 + i];
    if (c == 0) break;
    result[k++] = c;
  }
  result[k] = 0;
  return String(result);
}
void dog_action(int id) {
  sendFrame(1, CMD_ACTION, (uint8_t)constrain(id, 1, 255));
  faceStartPreset();
  if (id == 15 || id == 11) sound_event(24);      // wave / handshake greeting
  else if (id == 16 || id == 4) sound_event(25); // dance / circle playful cue
  else sound_event(23);                           // short action confirmation
  char b[16];
  snprintf(b, sizeof(b), "ACTION %d", id);
  mxSet(b);
}
void dog_gait_trot() {
  sendFrame(1, CMD_GAIT_TYPE, 0);
}
void dog_gait_amble() {
  sendFrame(1, CMD_GAIT_TYPE, 1);
}
void dog_gait_high() {
  sendFrame(1, CMD_GAIT_TYPE, 2);
}
void dog_mark_time_25() {
  sendFrame(1, CMD_MARK_TIME, 25);
}
void dog_mark_time_off() {
  sendFrame(1, CMD_MARK_TIME, 0);
}
void dog_load_motors() {
  sendFrame(1, CMD_LOAD, 0);
  sound_event(26);
}
void dog_unload_motors() {
  sendFrame(1, CMD_UNLOAD, 1);
  sound_event(27);
}
void dog_unload_leg(int id) {
  if (id >= 1 && id <= 4) sendFrame(1, CMD_UNLOAD, 0x10 + id);
}
void dog_load_leg(int id) {
  if (id >= 1 && id <= 4) sendFrame(1, CMD_LOAD, 0x20 + id);
}
void dog_attitude(int r, int p, int y) {
  uint8_t d[3] = { (uint8_t)constrain(r, 0, 255), (uint8_t)constrain(p, 0, 255), (uint8_t)constrain(y, 0, 255) };
  sendFrameN(1, CMD_ATTITUDE, d, 3);
}
void dog_attitude_reset() {
  dog_attitude(128, 128, 128);
}
void dog_translate(int x, int y, int z) {
  uint8_t d[3] = { (uint8_t)constrain(x, 0, 255), (uint8_t)constrain(y, 0, 255), (uint8_t)constrain(z, 0, 255) };
  sendFrameN(1, CMD_TRANSLATE, d, 3);
}
void dog_translate_reset() {
  dog_translate(128, 128, 128);
}
void dog_perform_on() {
  sendFrame(1, CMD_PERFORM, 1);
}
void dog_perform_off() {
  sendFrame(1, CMD_PERFORM, 0);
}
void dog_pace_normal() {
  sendFrame(1, CMD_MOVE_MODE, 0);
}
void dog_pace_slow() {
  sendFrame(1, CMD_MOVE_MODE, 1);
}
void dog_pace_high() {
  sendFrame(1, CMD_MOVE_MODE, 2);
}
void dog_imu_on() {
  sendFrame(1, CMD_IMU, 1);
}
void dog_imu_off() {
  sendFrame(1, CMD_IMU, 0);
}
void dog_periodic_rot(int axis, int p) {
  uint8_t d[3] = { 0, 0, 0 };
  if (axis >= 0 && axis < 3) d[axis] = (uint8_t)constrain(p, 0, 255);
  sendFrameN(1, CMD_PERIODIC_ROT, d, 3);
}
void dog_periodic_tran(int axis, int p) {
  uint8_t d[3] = { 0, 0, 0 };
  if (axis >= 0 && axis < 3) d[axis] = (uint8_t)constrain(p, 0, 255);
  sendFrameN(1, CMD_PERIODIC_TRAN, d, 3);
}

// Buzzer + events.
static uint16_t buzzNotes[5][3];
static uint8_t buzzCount = 0, buzzIndex = 0;
static uint32_t buzzNextMs = 0;
static bool buzzActive = false;
static void buzzSchedule(const uint16_t notes[][3], uint8_t count) {
  if (!buzzerPresent || !notes || count == 0) return;
  if (count > 5) count = 5;
  for (uint8_t i = 0; i < count; i++) {
    buzzNotes[i][0] = notes[i][0];
    buzzNotes[i][1] = notes[i][1];
    buzzNotes[i][2] = notes[i][2];
  }
  buzzCount = count;
  buzzIndex = 0;
  buzzNextMs = 0;
  buzzActive = true;
}
static void buzzerTick() {
  if (!buzzActive || !buzzerPresent || millis() < buzzNextMs) return;
  if (buzzIndex >= buzzCount) {
    buzzer.noTone();
    buzzActive = false;
    return;
  }
  uint8_t i = buzzIndex++;
  buzzer.tone(buzzNotes[i][0], buzzNotes[i][1]);
  buzzNextMs = millis() + buzzNotes[i][1] + buzzNotes[i][2];
}
void sound_event(int id) {
  static const uint16_t rise[][3] = { { 900, 80, 35 }, { 1250, 80, 35 }, { 1800, 120, 0 } };
  static const uint16_t connect[][3] = { { 1100, 70, 30 }, { 1600, 90, 0 } };
  static const uint16_t down[][3] = { { 1200, 90, 35 }, { 650, 150, 0 } };
  static const uint16_t autoMode[][3] = { { 700, 60, 25 }, { 1050, 60, 25 }, { 1400, 90, 0 } };
  static const uint16_t manualMode[][3] = { { 950, 90, 35 }, { 650, 120, 0 } };
  static const uint16_t walk[][3] = { { 900, 45, 25 }, { 1200, 45, 25 }, { 1500, 60, 0 } };
  static const uint16_t found[][3] = { { 1500, 60, 25 }, { 2100, 60, 25 }, { 2500, 100, 0 } };
  static const uint16_t search[][3] = { { 650, 45, 25 }, { 900, 45, 25 }, { 1150, 45, 25 }, { 1400, 70, 0 } };
  static const uint16_t impact[][3] = { { 260, 120, 30 }, { 260, 120, 30 }, { 180, 250, 0 } };
  static const uint16_t wallHit[][3] = { { 300, 90, 25 }, { 260, 90, 25 }, { 220, 190, 0 } };
  static const uint16_t hardStop[][3] = { { 220, 550, 0 } };
  static const uint16_t lowBattery[][3] = { { 420, 160, 90 }, { 320, 240, 0 } };
  static const uint16_t bridge[][3] = { { 320, 130, 50 }, { 850, 150, 0 } };
  static const uint16_t sleepTone[][3] = { { 620, 90, 40 }, { 420, 120, 40 }, { 300, 180, 0 } };
  static const uint16_t wakeTone[][3] = { { 700, 70, 25 }, { 1100, 70, 25 }, { 1600, 90, 0 } };
  static const uint16_t actionTone[][3] = { { 880, 45, 20 }, { 1175, 55, 0 } };
  static const uint16_t greetingTone[][3] = { { 1200, 45, 20 }, { 1600, 45, 20 }, { 2050, 75, 0 } };
  static const uint16_t danceTone[][3] = { { 740, 55, 20 }, { 988, 55, 20 }, { 1319, 55, 20 }, { 1760, 90, 0 } };
  static const uint16_t motorsOnTone[][3] = { { 500, 70, 20 }, { 900, 70, 20 }, { 1400, 100, 0 } };
  static const uint16_t motorsOffTone[][3] = { { 1400, 70, 20 }, { 900, 70, 20 }, { 500, 120, 0 } };
  static const uint16_t modeConfirmTone[][3] = { { 1000, 55, 20 }, { 1000, 90, 0 } };
  static const uint16_t errorTone[][3] = { { 180, 100, 25 }, { 180, 100, 25 }, { 180, 180, 0 } };
  switch (id) {
    case 1:
      buzzSchedule(rise, 3);
      ledsReadyManual();
      eyesSetMode(EYES_IDLE);
      break;
    case 2:
    case 3:
    case 16:
      buzzSchedule(connect, 2);
      eyesSetMode(EYES_HAPPY);
      break;
    case 4:
      buzzSchedule(down, 2);
      eyesSetMode(EYES_IDLE);
      break;
    case 5:
    case 19:
      buzzSchedule(autoMode, 3);
      ledsAutonomous();
      eyesSetMode(EYES_SCAN);
      break;
    case 6:
    case 11:
      buzzSchedule(walk, 3);
      ledsAutonomous();
      eyesSetMode(EYES_IDLE);
      break;
    case 7:
      buzzSchedule(found, 3);
      ledsDetected();
      eyesSetMode(EYES_PERSON);
      break;
    case 8:
    case 9:
    case 10:
      buzzSchedule(search, 4);
      ledsSearching();
      eyesSetMode(EYES_SCAN);
      break;
    case 12:
      buzzSchedule(impact, 3);
      ledsWallHit();
      eyesSetMode(EYES_DIZZY);
      break;
    case 13:
      buzzSchedule(hardStop, 1);
      ledsHardStop();
      eyesSetMode(EYES_DIZZY);
      break;
    case 14:
      buzzSchedule(lowBattery, 2);
      ledsLowBattery();
      eyesSetMode(EYES_SLEEP);
      break;
    case 15:
      buzzSchedule(bridge, 2);
      ledsBridgeLost();
      eyesSetMode(EYES_ALERT);
      break;
    case 17:
      buzzSchedule(sleepTone, 3);
      ledsSleep();
      eyesSetMode(EYES_SLEEP);
      break;
    case 18:
      buzzSchedule(manualMode, 2);
      ledsReadyManual();
      eyesSetMode(EYES_IDLE);
      break;
    case 20:
      buzzSchedule(wallHit, 3);
      ledsWallHit();
      eyesSetMode(EYES_DIZZY);
      break;
    case 21:
      buzzSchedule(wakeTone, 3);
      eyesSetMode(EYES_IDLE);
      break;
    case 22:
      buzzSchedule(sleepTone, 3);
      eyesSetMode(EYES_SLEEP);
      break;
    case 23:
      buzzSchedule(actionTone, 2);
      break;
    case 24:
      buzzSchedule(greetingTone, 3);
      eyesSetMode(EYES_HAPPY);
      break;
    case 25:
      buzzSchedule(danceTone, 4);
      eyesSetMode(EYES_HEARTS);
      break;
    case 26:
      buzzSchedule(motorsOnTone, 3);
      break;
    case 27:
      buzzSchedule(motorsOffTone, 3);
      break;
    case 28:
      buzzSchedule(modeConfirmTone, 2);
      break;
    case 29:
      buzzSchedule(errorTone, 3);
      eyesSetMode(EYES_ALERT);
      break;
  }
}
int buzzer_present() {
  return buzzerPresent ? 1 : 0;
}

// Movement and bridge helpers.
void update_movement() {
  static bool haveBaseline = false;
  if (!modulinoOk) return;

  const uint32_t now = millis();
  if (now - movementLastUpdateMs < MOVEMENT_UPDATE_INTERVAL_MS) return;
  movementLastUpdateMs = now;

  movement.update();
  diagMotionReads++;

  const float nx = movement.getX();
  const float ny = movement.getY();
  const float nz = movement.getZ();
  const float mag = sqrtf(nx * nx + ny * ny + nz * nz);

  if (!isfinite(nx) || !isfinite(ny) || !isfinite(nz) || !isfinite(mag) ||
      mag < 0.10f || mag > 4.00f) {
    diagMotionRejected++;
    motionPreviousValid = false;
    motionStableSamples = 0;
    vibration_value = 0.0f;
    vibration_peak = 0.0f;
    impactConfirmCnt = 0;
    return;
  }

  if (!motionPreviousValid) {
    motionPreviousX = nx;
    motionPreviousY = ny;
    motionPreviousZ = nz;
    motionPreviousValid = true;
  }

  // Motion must be measured between consecutive coherent samples, not against
  // the original mounting vector. Otherwise picking the module up, rotating it,
  // and putting it down leaves a permanent delta even though it is stationary.
  const float sdx = nx - motionPreviousX;
  const float sdy = ny - motionPreviousY;
  const float sdz = nz - motionPreviousZ;
  float sampleDelta = sqrtf(sdx * sdx + sdy * sdy + sdz * sdz);
  motionPreviousX = nx;
  motionPreviousY = ny;
  motionPreviousZ = nz;
  if (sampleDelta < 0.035f) sampleDelta = 0.0f;

  if (sampleDelta > 3.50f) {
    diagMotionRejected++;
    motionPreviousValid = false;
    motionStableSamples = 0;
    vibration_value = 0.0f;
    vibration_peak = 0.0f;
    impactConfirmCnt = 0;
    return;
  }

  if (!haveBaseline) {
    if (mag < 0.60f || mag > 1.40f || sampleDelta > 0.08f) {
      diagMotionRejected++;
      return;
    }
    accel_x = nx; accel_y = ny; accel_z = nz;
    rptX = nx; rptY = ny; rptZ = nz;
    rptSeeded = true;
    haveBaseline = true;
    motionStableSamples = MOTION_REARM_SAMPLES;
    motionFeedbackArmed = true;
    vibration_value = 0.0f;
    vibration_peak = 0.0f;
    Monitor.println("[movement] baseline acquired; settle/rebaseline feedback armed");
    return;
  }

  rptX = nx; rptY = ny; rptZ = nz;
  const bool settled = sampleDelta <= 0.08f;

  // A hard, coherent acceleration delta temporarily overrides every other
  // external-matrix expression with the serious face. The existing Movement
  // validation above prevents invalid samples from triggering it.
  if (!settled) faceNoteAggressiveAcceleration(sampleDelta);

  if (settled) {
    if (motionStableSamples < 255) motionStableSamples++;

    // After the module settles in any new orientation, rebaseline to that
    // orientation. This is the critical fix for the permanent freak-out.
    if (motionStableSamples >= MOTION_REARM_SAMPLES) {
      accel_x = nx;
      accel_y = ny;
      accel_z = nz;
      vibration_value *= 0.35f;
      vibration_peak *= 0.35f;
      if (vibration_value < 0.01f) vibration_value = 0.0f;
      if (vibration_peak < 0.01f) vibration_peak = 0.0f;

      if (now >= motionFeedbackCooldownUntilMs) {
        motionFeedbackArmed = true;
        motionGesturePeak = 0.0f;
      }
    }
  } else {
    motionStableSamples = 0;
    if (sampleDelta > motionGesturePeak) motionGesturePeak = sampleDelta;
    vibration_value = sampleDelta;
    if (sampleDelta > vibration_peak) vibration_peak = sampleDelta;
    if (sampleDelta >= 0.12f) diagMotionActive++;
  }

  // One feedback event on the first coherent movement after a settled period.
  if (!settled && sampleDelta >= 0.22f && motionFeedbackArmed &&
      now >= motionFeedbackCooldownUntilMs) {
    motionFeedbackArmed = false;
    motionFeedbackCooldownUntilMs = now + MOTION_FEEDBACK_COOLDOWN_MS;

    if (sampleDelta >= 0.90f) {
      eyesSetMode(EYES_DIZZY);
      sound_event(12);
    } else if (sampleDelta >= 0.45f) {
      eyesSetMode(EYES_ALERT);
      sound_event(20);
    } else {
      eyesSetMode(EYES_SURPRISED);
      sound_event(23);
    }

    Monitor.println(String("[motion-feedback] one-shot delta=") +
                    String(sampleDelta, 3));
  }

  impactConfirmCnt = 0;
}

float get_accel_x() {
  return rptSeeded ? rptX : accel_x;
}
float get_accel_y() {
  return rptSeeded ? rptY : accel_y;
}
float get_accel_z() {
  return rptSeeded ? rptZ : accel_z;
}
float get_vibration() {
  return vibration_value;
}
float get_vibration_peak_clear() {
  float p = vibration_peak;
  vibration_peak = 0;
  vibration_value = 0;
  vibReportEma = 0;
  impactConfirmCnt = 0;
  vibrationPeakMs = millis();
  return p;
}
int movement_present() {
  Wire1.beginTransmission(0x6A);
  return Wire1.endTransmission() == 0;
}
int movement_responds() {
  return modulinoOk ? 1 : 0;
}
int movement_addr() {
  return 0x6A;
}
int impact_confirm_count() {
  return impactConfirmCnt;
}
int impact_cooldown_remaining_ms() {
  long remaining = (long)impactCooldownUntil - (long)millis();
  if (remaining < 0) remaining = 0;
  return (int)remaining;
}
int motion_reads_count() {
  return (int)diagMotionReads;
}
int motion_active_count() {
  return (int)diagMotionActive;
}
int eyes_mode_get() {
  return (int)eyesMode;
}
int eyes_flush_ok_count() {
  return (int)eyesFlushOkCount;
}
int eyes_flush_error_count() {
  return (int)eyesFlushErrorCount;
}
int eyes_bus_fault() {
  return eyesBusFault ? 1 : 0;
}

int movement_rebegin() {
  // Non-aggressive recovery: re-run the known-good Modulino Movement begin path only
  // No I2C scan, no software reset, no Wire1 re-clock, no OLED/buzzer changes
  modulinoOk = movement.begin();
  Monitor.print("[movement-rebegin] result=");
  Monitor.println(modulinoOk ? 1 : 0);
  return modulinoOk ? 1 : 0;
}
int wire1_scan_count() {
  int count = 0;
  for (uint8_t a = 1; a < 127; a++) {
    Wire1.beginTransmission(a);
    if (Wire1.endTransmission() == 0) count++;
  }
  return count;
}
void matrix_scroll(String s, int speed) {
  mxSet(s.c_str(), speed > 0 ? speed : 32);
}
void matrix_flash(String s) {
  mxSet(s.c_str(), 24);
}
void matrix_person_scroll(String s) {
  mxSet(s.c_str(), 24);
}
void matrix_text(String s, int hold) {
  (void)hold;
  mxSet(s.c_str());
}
void matrix_hello() {
  mxSet("HELLO");
}
void matrix_person() {
  personDetectedFlag = 1;
  faceSetSearching(false);
  oledDetection = "PERSON";
  mxSet("PERSON", 24);
  ledsDetected();
  eyesSetMode(EYES_PERSON);
}
void matrix_searching() {
  personDetectedFlag = 0;
  faceSetSearching(true);
  oledDetection = "SEARCHING";
  mxSet("SEARCHING");
  ledsSearching();
  eyesSetMode(EYES_SCAN);
}
void matrix_idle() {
  faceSetSearching(false);
  mxSet("IDLE");
  eyesSetMode(EYES_IDLE);
}
void matrix_alert() {
  oledDetection = "IMPACT";
  mxSet("ALERT", 24);
  ledsWallHit();
  eyesSetMode(EYES_ALERT);
}
void matrix_battery_show() {
  char b[16];
  snprintf(b, sizeof(b), "BAT %d%%", oledBatteryPct);
  mxSet(b);
}
void matrix_set_connected(int c) {
  int next = constrain(c, 0, 99);
  if (next != mxClients) {
    mxClients = next;
    mxSet(mxClients > 0 ? "LINK" : "WAITING", 30);
  }
}
void set_person_detected(int v) {
  const int next = v ? 1 : 0;
  if (next == personDetectedFlag) return;
  personDetectedFlag = next;
  if (next) {
    matrix_person();
    sound_event(7);  // unique detection melody, cyan/white LEDs, target-lock face
  } else {
    matrix_searching();
  }
}
void oled_set_ip(String s) {
  oledIP = s;
}
void oled_set_status(String s) {
  oledStatus = s;
}
void oled_set_detection(String s) {
  oledDetection = s;
}
void oled_clear() {
  oledIP = "";
  oledStatus = "";
  oledDetection = "";
}
void softOffInit() {
  pinMode(SOFT_OFF_PIN, SOFT_OFF_ACTIVE_HIGH ? INPUT_PULLDOWN : INPUT_PULLUP);
  const bool asserted = SOFT_OFF_ACTIVE_HIGH
      ? (digitalRead(SOFT_OFF_PIN) == HIGH)
      : (digitalRead(SOFT_OFF_PIN) == LOW);
  softOffRawState = asserted;
  softOffStableState = asserted;
  softOffHandled = asserted;  // never shut down solely because the pin was active at boot
  softOffLastChangeMs = millis();
}
void softOffPoll() {
  const uint32_t now = millis();
  const bool asserted = SOFT_OFF_ACTIVE_HIGH
      ? (digitalRead(SOFT_OFF_PIN) == HIGH)
      : (digitalRead(SOFT_OFF_PIN) == LOW);

  if (asserted != softOffRawState) {
    softOffRawState = asserted;
    softOffLastChangeMs = now;
  }

  if (now - softOffLastChangeMs < SOFT_OFF_DEBOUNCE_MS) return;
  if (softOffStableState != softOffRawState) {
    softOffStableState = softOffRawState;
    if (!softOffStableState) softOffHandled = false;
  }

  if (!softOffStableState || softOffHandled) return;
  softOffHandled = true;

  dog_stop();
  dog_unload_motors();
  oledStatus = "POWERING DOWN";
  oledDetection = "SOFT OFF";
  mxSet("POWER OFF", 18);
  sound_event(13);
  ledsSleep();
  Bridge.call("soft_off");
}


static void safeDiagTick() {
#if ROBODOG_SAFE_MONITOR_DIAG
  if (millis() - diagLastHeartbeatMs >= 5000UL) {
    diagLastHeartbeatMs = millis();
    Monitor.println(String("[diag] ms=") + String(millis()) + " movementOk=" + String(modulinoOk ? 1 : 0) + " movementPresent=" + String(movement_present()) + " buzzer=" + String(buzzerPresent ? 1 : 0) + " faceMatrix=" + String(faceMatrixPresent ? 1 : 0) + " face=" + String((int)faceExpression) + " faceLowBat=" + String(faceBatteryLow ? 1 : 0) + " eyes=" + String(eyesPresent ? 1 : 0) + " eyesFlushOk=" + String(eyesFlushOkCount) + " eyesFlushErr=" + String(eyesFlushErrorCount) + " eyesBusFault=" + String(eyesBusFault ? 1 : 0) + " sens=" + String(motionSensitivityMode) + " localStop=" + String(mcuLocalImpactStopEnabled ? 1 : 0) + " impactCnt=" + String(impactConfirmCnt) + " cooldown=" + String(impact_cooldown_remaining_ms()) + " reads=" + String(diagMotionReads) + " active=" + String(diagMotionActive) + " rejected=" + String(diagMotionRejected) + " stable=" + String(motionStableSamples) + " armed=" + String(motionFeedbackArmed ? 1 : 0) + " gPeak=" + String(motionGesturePeak, 3) + " vib=" + String(vibration_value, 4) + " peak=" + String(vibration_peak, 4) + " mag=" + String(movement_mag(), 4));
  }
#endif
}

void setup() {
  Monitor.begin();
  Monitor.println("[boot] Robodog Pro v8.4.8: external Modulino faces + EMA motion + diagnostics + 0x3C eyes");
  pinMode(LED3_R, OUTPUT);
  pinMode(LED3_G, OUTPUT);
  pinMode(LED3_B, OUTPUT);
  pinMode(LED4_R, OUTPUT);
  pinMode(LED4_G, OUTPUT);
  pinMode(LED4_B, OUTPUT);
  ledsSleep();
  ledsBootSelfTest();
  matrix.begin();
  matrix.setGrayscaleBits(1);
  {
    uint8_t f[MX_PIXELS];
    memset(f, 1, sizeof(f));
    matrix.draw(f);
    delay(120);
    memset(f, 0, sizeof(f));
    matrix.draw(f);
  }
  DOG.begin(115200);
  Bridge.begin();
  softOffInit();
  oledPresent = oledInit();
  if (oledPresent) {
    Monitor.println(String("[oled] OEM DOGZILLA OLED detected on Wire addr=0x") + String(oledAddr, HEX) + " geometry=128x32");
    memset(oledBuf, 0xFF, sizeof(oledBuf));
    oledFlush();
    delay(250);
    memset(oledBuf, 0, sizeof(oledBuf));
    oledFlush();
  } else Monitor.println("[oled] OEM DOGZILLA OLED NOT detected on Wire at 0x3C/0x3D");
  Wire1.begin();
  Wire1.setClock(400000); // responsive OLED animation; flush retry/recovery remains active
  Modulino.begin(Wire1);
  applyMotionSensitivityDefaults();
  Monitor.println("[i2c] Wire1/Qwiic boot scan skipped");
  Wire1.beginTransmission(0x39);
  if (Wire1.endTransmission() == 0) {
    faceMatrixPresent = faceMatrix.begin();
    if (!faceMatrixPresent) {
      // ModulinoLEDMatrix::begin() closes its Wire object on failure. Restore
      // the shared bus so Movement, Buzzer, and the eyes OLED remain usable.
      Wire1.begin();
      Wire1.setClock(400000);
      Modulino.begin(Wire1);
    }
  }
  Monitor.println(faceMatrixPresent ? "[modulino] LED Matrix OK addr=0x39" : "[modulino] LED Matrix absent optional");
  modulinoOk = movement.begin();
  Monitor.println(modulinoOk ? "[modulino] Movement OK" : "[modulino] Movement FAIL");
  if (EYES_WIRE1_ENABLED) {
    eyesPresent = eyesInit();
    if (eyesPresent) {
      Monitor.println(String("[eyes] SSD1306 eyes OLED detected on Wire1 addr=0x") + String(eyesAddr, HEX));
      memset(eyesBuf, 0, sizeof(eyesBuf));
      eyesSetMode(EYES_IDLE);
      eyesTick();
    }
  } else {
    eyesPresent = false;
    Monitor.println("[eyes] disabled for Movement isolation; no OLED traffic on Wire1");
  }
  buzzerPresent = buzzer.begin();
  if (!buzzerPresent) {
    Wire1.beginTransmission(0x3C);
    if (Wire1.endTransmission() == 0) {
      delay(20);
      buzzerPresent = buzzer.begin();
    }
  }
  if (buzzerPresent) {
    buzzer.tone(1200, 150);
    delay(180);
    buzzer.noTone();
    Monitor.println("[modulino] Buzzer OK");
  } else Monitor.println("[modulino] Buzzer absent optional");
  faceTick(true);
#define PROVIDE(n, f) Bridge.provide(n, f)
  PROVIDE("dog_forward", dog_forward);
  PROVIDE("dog_backward", dog_backward);
  PROVIDE("dog_left", dog_left);
  PROVIDE("dog_right", dog_right);
  PROVIDE("dog_turn_left", dog_turn_left);
  PROVIDE("dog_turn_right", dog_turn_right);
  PROVIDE("dog_stop", dog_stop);
  PROVIDE("dog_action", dog_action);
  PROVIDE("dog_battery", dog_battery);
  PROVIDE("dog_read_roll", dog_read_roll);
  PROVIDE("dog_read_pitch", dog_read_pitch);
  PROVIDE("dog_read_yaw", dog_read_yaw);
  PROVIDE("dog_read_roll_f", dog_read_roll_f);
  PROVIDE("dog_read_pitch_f", dog_read_pitch_f);
  PROVIDE("dog_read_yaw_f", dog_read_yaw_f);
  PROVIDE("dog_firmware_version", dog_firmware_version);
  PROVIDE("dog_gait_trot", dog_gait_trot);
  PROVIDE("dog_gait_amble", dog_gait_amble);
  PROVIDE("dog_gait_high", dog_gait_high);
  PROVIDE("dog_mark_time_25", dog_mark_time_25);
  PROVIDE("dog_mark_time_off", dog_mark_time_off);
  PROVIDE("dog_load_motors", dog_load_motors);
  PROVIDE("dog_unload_motors", dog_unload_motors);
  PROVIDE("dog_unload_leg", dog_unload_leg);
  PROVIDE("dog_load_leg", dog_load_leg);
  PROVIDE("dog_attitude", dog_attitude);
  PROVIDE("dog_attitude_reset", dog_attitude_reset);
  PROVIDE("dog_translate", dog_translate);
  PROVIDE("dog_translate_reset", dog_translate_reset);
  PROVIDE("dog_perform_on", dog_perform_on);
  PROVIDE("dog_perform_off", dog_perform_off);
  PROVIDE("dog_pace_normal", dog_pace_normal);
  PROVIDE("dog_pace_slow", dog_pace_slow);
  PROVIDE("dog_pace_high", dog_pace_high);
  PROVIDE("dog_imu_on", dog_imu_on);
  PROVIDE("dog_imu_off", dog_imu_off);
  PROVIDE("dog_periodic_rot", dog_periodic_rot);
  PROVIDE("dog_periodic_tran", dog_periodic_tran);
  PROVIDE("get_accel_x", get_accel_x);
  PROVIDE("get_accel_y", get_accel_y);
  PROVIDE("get_accel_z", get_accel_z);
  PROVIDE("get_vibration", get_vibration);
  PROVIDE("get_vibration_peak_clear", get_vibration_peak_clear);
  PROVIDE("movement_present", movement_present);
  PROVIDE("movement_responds", movement_responds);
  PROVIDE("movement_addr", movement_addr);
  PROVIDE("movement_rebegin", movement_rebegin);
  PROVIDE("wire1_scan_count", wire1_scan_count);
  PROVIDE("movement_mag", movement_mag);
  PROVIDE("motion_sensitivity", motion_sensitivity);
  PROVIDE("motion_sensitivity_get", motion_sensitivity_get);
  PROVIDE("motion_local_stop_enable", motion_local_stop_enable);
  PROVIDE("motion_threshold_g", motion_threshold_g);
  PROVIDE("impact_confirm_count", impact_confirm_count);
  PROVIDE("impact_cooldown_remaining_ms", impact_cooldown_remaining_ms);
  PROVIDE("motion_reads_count", motion_reads_count);
  PROVIDE("motion_active_count", motion_active_count);
  PROVIDE("face_matrix_present", face_matrix_present);
  PROVIDE("face_matrix_expression", face_matrix_expression);
  PROVIDE("face_matrix_battery_low", face_matrix_battery_low);
  PROVIDE("face_matrix_searching_set", face_matrix_searching_set);
  PROVIDE("face_matrix_aggressive_threshold_mg", face_matrix_aggressive_threshold_mg);
  PROVIDE("matrix_hello", matrix_hello);
  PROVIDE("matrix_person", matrix_person);
  PROVIDE("matrix_searching", matrix_searching);
  PROVIDE("matrix_idle", matrix_idle);
  PROVIDE("matrix_alert", matrix_alert);
  PROVIDE("matrix_battery", matrix_battery_show);
  PROVIDE("matrix_scroll", matrix_scroll);
  PROVIDE("matrix_flash", matrix_flash);
  PROVIDE("matrix_person_scroll", matrix_person_scroll);
  PROVIDE("matrix_text", matrix_text);
  PROVIDE("matrix_set_connected", matrix_set_connected);
  PROVIDE("set_person_detected", set_person_detected);
  PROVIDE("oled_set_ip", oled_set_ip);
  PROVIDE("oled_set_status", oled_set_status);
  PROVIDE("oled_set_detection", oled_set_detection);
  PROVIDE("oled_clear", oled_clear);
  PROVIDE("sound_event", sound_event);
  PROVIDE("buzzer_present", buzzer_present);
  PROVIDE("led_status", led_status);
  PROVIDE("eyes_demo", eyes_demo);
  PROVIDE("eyes_mode", eyes_mode);
  PROVIDE("eyes_present", eyes_present);
  PROVIDE("eyes_addr", eyes_addr);
  PROVIDE("eyes_mode_get", eyes_mode_get);
  PROVIDE("eyes_motion_reactive", eyes_motion_reactive);
  PROVIDE("eyes_flush_ok_count", eyes_flush_ok_count);
  PROVIDE("eyes_flush_error_count", eyes_flush_error_count);
  PROVIDE("eyes_bus_fault", eyes_bus_fault);
  delay(300);
  dog_stop();
  oledDetection = "READY";
  mxSet("ROBODOG V8.4.6", 18);
  // Quiet boot for Movement isolation.
  Monitor.println("[boot] READY");
}

void loop() {
  softOffPoll();
  update_movement();
  buzzerTick();
  ledsTick();
  eyesTick();
  faceTick();
  mxIdleTick();
  mxTick();
  safeDiagTick();
  if (oledPresent && millis() - oledLastUpdate >= 500) {
    oledLastUpdate = millis();
    oledRedraw();
  }
  delay(15);
}
