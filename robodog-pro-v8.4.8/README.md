# Robodog Pro v8.4

A direct-import Arduino App Lab project for an Arduino UNO Q controlling a DOGZILLA robot dog. Robodog Pro combines an Arduino Zephyr MCU sketch, Python services, a browser-based control panel, an isolated Linux gamepad reader, camera-based object detection, autonomous behavior, safety supervision, an OEM OLED, the UNO Q LED matrix, optional Modulino hardware, and optional USB audio.

This README describes the current **v8.4** project as a complete product. Historical release notes are intentionally omitted because older packages are maintained separately as backups.

## Current platform baseline

- **Board:** Arduino UNO Q
- **MCU platform:** `arduino:zephyr` **0.90.0**
- **FQBN:** `arduino:zephyr:unoq`
- **Application runtime:** Arduino App Lab with CPython 3.13
- **Web interface:** Arduino App Lab WebUI on TCP port **7000**
- **Linux/MCU communication:** Arduino RouterBridge RPC
- **Default client URL:** `http://<UNO-Q-IP>:7000`
- **Optional AP URL:** `http://10.42.0.1:7000`

The first run after installing or changing Zephyr may download or reinstall the `arm-zephyr-eabi` 0.16.8 toolchain. App Lab can remain at `Installing tool zephyr:arm-zephyr-eabi@0.16.8` while extracting and registering the approximately 400 MB installed toolchain. Do not interrupt the first installation while `arduino-app-cli` is active.

## Major capabilities

### DOGZILLA control

- Forward, backward, lateral, and turning commands
- Manual and autonomous operating modes
- Preset DOGZILLA actions and named actions
- Attitude, translation, periodic rotation, and periodic translation controls
- Individual leg load and unload controls
- Walkaround sequence
- Battery and firmware telemetry when returned by the DOGZILLA interface
- Soft-off handling through the MCU bridge

### OEM wireless controller

A dedicated `gamepad-reader` Brick owns `/dev/input` and relays Linux joystick records over the internal Docker network. The main application never needs direct access to the host input devices.

The v8.4 controller policy deliberately separates:

1. **USB receiver present**
2. **Powered handheld controller active**
3. **Controller neutral and armed**

Linux joystick initialization records, including events with the `JS_EVENT_INIT` bit, do not count as user input and cannot trigger movement or STOP commands. A receiver that remains enumerated while the handheld controller is off is reported as present but inactive.

Safety behavior includes:

- Neutral-stick arming before movement
- Hard-stop latching
- A single serialized command worker
- Axis activation and release hysteresis
- Button debounce and lockout
- Bridge-loss protection
- Active-controller disconnect STOP
- Duplicate STOP suppression during intentional disable or rescan
- IMU collision and tilt protection
- Local disarming when movement is blocked

Disabling the controller sends one defensive STOP and then closes the controller session intentionally. An idle receiver must not produce a continuous STOP stream.

### Camera and object detection

- Automatic probing of available `/dev/video*` devices
- USB camera stream in the WebUI
- Optional 180-degree rotation
- Continuous object detection through the App Lab object-detection runner
- Configurable target, with `person` as the default
- Bounding-box overlay and detection metadata
- Optional live-autonomy steering hook
- Camera reset and reprobe endpoint

### Zero image persistence

Camera captures and annotations are handled in memory only.

The project must not contain or create:

- `CAPTURE_DIR`
- image files in `/tmp`
- saved captures under `/home/arduino`
- `cv2.imwrite(...)`
- an `image_path` API field

Frames may be returned as base64 data in API responses or held in the single in-memory frame slot. They are discarded when replaced or when the application exits.

### Displays

#### External Modulino 12x8 LED Matrix faces

The optional Modulino LED Matrix is connected to `Wire1` at its default I2C
address `0x39`. It is dedicated to Robodog's three primary expressions:

- **Happy:** normal idle state
- **Sad:** DOGZILLA battery below 20%; remains latched until battery reaches 25%
- **Serious:** coherent acceleration delta of 0.90 g or greater; held for 2.5 seconds

While Robodog is moving, searching, or executing a DOGZILLA preset, the happy
face alternates with a wink frame every 350 ms to create a funny/playful
animation. The wink is an animation frame, not a fourth persistent mood.

Expression priority is:

```text
aggressive acceleration > low battery > active/funny > idle/happy
```

The matrix is optional. If address `0x39` is not detected, the sketch skips its
initialization and continues operating the Movement, Buzzer, and eyes OLED on
the shared `Wire1` bus.

#### UNO Q 8x13 LED matrix

The matrix provides concise operational feedback for readiness, IP address, battery, connection state, movement, actions, detection, and safety events.

The matrix scheduler guarantees that the current scrolling message completes a full pass before the normal idle rotation replaces it. A full address such as `IP 192.168.7.112` must scroll completely off the display before battery or ready text begins.

Project version text is intentionally not displayed on the matrix. The startup message is:

```text
ROBODOG READY
```

Urgent event feedback such as STOP, IMPACT, PERSON, movement commands, and explicit actions may interrupt idle rotation.

#### OEM DOGZILLA OLED

The 128x32 OLED is connected to `Wire` and shows operational information such as:

- Current status or mode
- Advertised IP address
- Battery and vibration values
- Detection or safety state
- Camera status/FPS where applicable

### Modulino hardware

- **Modulino Movement:** connected through `Wire1` and used for motion, vibration, impact, tilt, and autonomous recovery logic
- **Modulino LED Matrix:** optional 12x8 face display on `Wire1` at `0x39`; happy/sad/serious faces plus the active wink animation
- **Modulino Buzzer:** optional; used for local status and safety cues when detected

The OEM OLED remains on `Wire`, while Modulino devices remain on `Wire1` to avoid bus conflicts.

### Audio feedback

The project supports optional USB audio through an ALSA-compatible adapter and includes original synthesized WAV cues for:

- Startup
- Short and double bark
- Sniff
- Pant
- Curious
- Playful
- Alert
- Low battery
- Sleep
- Hard stop
- Bridge loss

Audio playback is queued and has ALSA-device failover. The Modulino Buzzer remains an independent local feedback path.

### Autonomous behavior

Autonomous mode can use Modulino Movement telemetry to:

- Walk forward
- Turn periodically
- React to impacts or vibration spikes
- React to excessive tilt
- Back away and recover
- Perform occasional dog-like feedback actions

Live object detection can also call the vision-autonomy hook when enabled.

## Runtime architecture

### MCU sketch

`sketch/sketch.ino` owns deterministic hardware operations:

- DOGZILLA serial protocol
- RouterBridge providers
- UNO Q LED matrix
- OEM OLED
- Modulino Movement
- Optional Modulino Buzzer
- Soft-off input
- Local safety feedback

### Main Python application

`python/main.py` is the v8.4 production orchestrator. It:

- Creates the WebUI service
- Registers the complete HTTP API surface
- Rejects duplicate routes and non-callable handlers
- Registers all supervised worker targets
- Initializes watchdog heartbeat timestamps
- Starts hardware precheck
- Starts all daemon workers
- Starts the watchdog
- Publishes coordinated matrix and OLED readiness feedback
- Calls `App.run()`

### Supervised workers

- `audio`: queued USB audio feedback
- `gamepad`: controller receiver, activity, arming, and safety state
- `camera`: camera acquisition and RAM-only frame publication
- `live-detect`: continuous object detection and autonomy hook
- `autonomy`: autonomous DOGZILLA behavior
- `wifi-status`: advertised IP and AP/client status publisher
- `bridge-liveness`: RouterBridge health monitor
- `oled-publisher`: throttled OLED IP, mode, and FPS updates
- `matrix-worker`: serialized matrix event queue
- `motion`: Modulino Movement telemetry
- `status-line`: concise periodic runtime diagnostics

The watchdog monitors worker heartbeat timestamps and restarts stale worker targets.

### Gamepad Brick

`bricks/gamepad_input` contains:

- `brick_config.yaml`: App Lab Brick identity
- `brick_compose.yaml`: isolated reader service definition
- `reader.py`: `/dev/input/js*` reader and framed TCP relay
- `README.md`: Brick-specific documentation

The sidecar listens on internal TCP port **8765**. The port is not published outside the App Lab Docker network.

## Project structure

```text
app.yaml
README.md
assets/
  index.html
  app.js
  style.css
bricks/
  gamepad_input/
    __init__.py
    brick_config.yaml
    brick_compose.yaml
    reader.py
    README.md
host/
  dogzilla-ap.sh
python/
  main.py
  state.py
  hw_bridge.py
  autonomy.py
  perception.py
  input_controller.py
  audio_feedback.py
  ui_api.py
  self_check.py
  audio/
    *.wav
sketch/
  sketch.ino
  sketch.yaml
```

`app.yaml` must remain at the archive root. The ZIP must not contain an extra outer directory.

## Arduino dependencies

`sketch/sketch.yaml` pins Zephyr 0.90.0 and the complete RouterBridge dependency chain.

Required platform:

```text
arduino:zephyr (0.90.0)
```

Required RouterBridge chain:

```text
Arduino_RouterBridge 0.4.3
Arduino_RPClite 0.3.0
MsgPack 0.4.2
DebugLog 0.8.4
ArxTypeTraits 0.3.2
ArxContainer 0.7.0
```

Additional project libraries include Arduino Modulino, ArduinoGraphics, Arduino HS300x, Arduino LPS22HB, Arduino LSM6DSOX, Arduino LTR381RGB, STM32duino VL53L4CD, and STM32duino VL53L4ED.

The explicit RouterBridge declaration is intentional. The UNO Q Zephyr 0.90.0 platform includes a RouterBridge stub that requests installation of the actual library. RouterBridge 0.4.3 also requires the RPClite and supporting serialization/container libraries listed above.

## Import and run

1. Install or verify Zephyr 0.90.0 on the UNO Q host:

   ```bash
   arduino-cli core install arduino:zephyr@0.90.0
   arduino-cli core list | grep -i zephyr
   ```

2. Import the project ZIP into Arduino App Lab.
3. Select the UNO Q.
4. Click **Run** once.
5. Allow the initial platform/tool installation, compilation, MCU flashing, Python provisioning, and Docker startup to finish.
6. Open the Network URL printed by App Lab, normally:

   ```text
   http://<UNO-Q-IP>:7000
   ```

A healthy startup includes:

```text
Board platform: arduino:zephyr (0.90.0)
Robodog Pro v8.4 production ready.
App started
```

## First-run tool installation

If App Lab reports:

```text
Task Installing tool zephyr:arm-zephyr-eabi@0.16.8: 0.00%
```

wait while the archive is extracted. In another terminal, progress can be checked with:

```bash
ps -eo pid,etime,%cpu,%mem,stat,cmd | grep 'arduino-app-cli daemon' | grep -v grep
du -sh "$HOME/.arduino15/packages/zephyr/tools/arm-zephyr-eabi" 2>/dev/null || true
df -h "$HOME" /tmp
```

The compiler is ready when this command succeeds:

```bash
"$HOME/.arduino15/packages/zephyr/tools/arm-zephyr-eabi/0.16.8/bin/arm-zephyr-eabi-g++" --version | head -n 1
```

## Build-cache recovery

Errors involving a truncated or corrupt `core.a` or `WInterrupts.cpp.o` are generated-cache failures, not project source failures. Typical messages include:

```text
file truncated
invalid string offset ... for section '.strtab'
```

Stop the project and remove only its generated cache:

```bash
APP="$HOME/ArduinoApps/robodog-pro-v8-4"
rm -rf "$APP/.cache" "$APP/sketch/build" "$APP/sketch/.build"
```

Then reopen App Lab and run once. Do not reinstall Zephyr solely because a project cache archive is damaged.

## Network operation

### Normal client mode

The preferred address is the `HOST_IP` value injected by App Lab. If that value is unavailable, the configured static fallback is used for display and diagnostics.

Verify the active address with:

```bash
ip -4 -brief address show wlan0
```

### DOGZILLA access point

AP switching is a host-level NetworkManager operation. The App Lab container does not have `nmcli` or permission to reconfigure host `wlan0`.

From the UNO Q host shell:

```bash
APP="$HOME/ArduinoApps/robodog-pro-v8-4"
chmod +x "$APP/host/dogzilla-ap.sh"
"$APP/host/dogzilla-ap.sh" install
"$APP/host/dogzilla-ap.sh" up
```

Connect a computer or phone using:

```text
SSID: DOGZILLA-UNOQ
Password: dogzilla123
URL: http://10.42.0.1:7000
```

Starting AP mode reconfigures `wlan0`, disconnects normal Wi-Fi client mode, and may terminate the SSH session used to start the AP.

Stop AP mode from the host:

```bash
"$HOME/ArduinoApps/robodog-pro-v8-4/host/dogzilla-ap.sh" down
```

The WebUI provides copyable host commands rather than pretending the container can execute privileged NetworkManager operations.

## API overview

### Core and diagnostics

```text
GET /api/state
GET /api/health
GET /api/precheck
GET /api/telemetry
GET /api/vibration
```

### DOGZILLA commands and actions

```text
GET /api/cmd/{cmd}
GET /api/action/{action_id}
GET /api/action_named/{name}
GET /api/mode/{mode}
GET /api/attitude/{roll}/{pitch}/{yaw}
GET /api/translate/{x}/{y}/{z}
GET /api/periodic_rot/{axis}/{period_x10}
GET /api/periodic_tran/{axis}/{period_x10}
GET /api/unload_leg/{leg_id}
GET /api/load_leg/{leg_id}
```

### Controller and safety

```text
GET /api/gamepad/{enabled}
GET /api/controller/status
GET /api/controller/rescan
GET /api/safety/status
GET /api/safety/reset
```

### Audio

```text
GET /api/audio/status
GET /api/audio/test/{event}
GET /api/audio/mute/{enabled}
```

### Camera and detection

```text
GET /api/camera
GET /api/camera_reset
GET /api/camera_rotate/{enabled}
GET /api/detect/{target}/{language}
GET /api/live_detect/{action}
GET /api/live_detect_target/{target}
GET /api/live_detect_state
```

### Displays, mood, sequences, chat, and Wi-Fi status

```text
GET /api/matrix_scroll/{text}
GET /api/matrix_flash/{text}
GET /api/mood/{name}
GET /api/sequence/walkaround
GET /api/sequence/stop
GET /api/chat/{prompt}
GET /api/wifi_status
```

Legacy AP compatibility routes remain registered, but host AP switching must be performed using `host/dogzilla-ap.sh`.

## Verification

### Containers and WebUI

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
ss -ltnp | grep ':7000'
curl -fsS http://127.0.0.1:7000/api/health
```

Expected services:

- Main Robodog application
- Gamepad reader
- Object-detection runner

### Controller-off acceptance test

Leave the USB receiver installed and the handheld controller off:

```bash
curl -fsS http://127.0.0.1:7000/api/controller/status
```

Expected policy state:

```text
receiver_present: true
controller_active: false
connected: false
armed: false
events: 0
```

Initialization records may increment a separate diagnostic initialization counter, but must not increment the user event count or generate command traffic.

### STOP-loop check

```bash
docker logs --since 2m robodog-pro-v8-4-main-1 2>&1 |
  grep -E 'gamepad-device|gamepad-safety|gamepad-command|Command: stop' ||
  echo 'No controller STOP activity.'
```

One defensive STOP may occur when RouterBridge first becomes available. A powered-off controller with an enumerated receiver must not create a continuing STOP stream.

### Matrix acceptance test

Confirm that the complete IP message scrolls across and fully leaves the display before the next idle message begins:

```text
IP 192.168.7.112
```

The matrix must not display a project version.

### External face-matrix acceptance test

With the Modulino LED Matrix connected to the `Wire1` Qwiic chain:

1. Boot the app and confirm the happy face appears while idle.
2. Send any movement command and confirm the happy/wink animation starts.
3. Send STOP and confirm the steady happy face returns.
4. Start object search or a preset action and confirm the funny animation.
5. Return a DOGZILLA battery value below 20% and confirm the sad face. The sad
   face clears only after a reading of 25% or higher.
6. Move the Modulino Movement aggressively enough to exceed the 0.90 g
   consecutive-sample delta and confirm the serious face for about 2.5 seconds.
7. In the MCU monitor, confirm `faceMatrix=1`. Expression IDs are `0=happy`,
   `1=sad`, `2=serious`, and `3=funny`.

## Self-check

Run from the project root:

```bash
python3 python/self_check.py
```

The self-check validates:

- Python syntax
- Gamepad reader syntax
- Required project files
- WAV format and presence
- Zephyr FQBN and platform pin
- RouterBridge dependency chain
- Required MCU bridge markers
- Audio registration
- API handler consistency
- Fallback-IP definition order
- Controller receiver/activity regression guards
- Absence of camera persistence markers
- Absence of nonfunctional AP buttons
- Matrix full-cycle and no-version policy

Expected result:

```text
PASS
```

## Safety notes

- Support the robot securely during initial movement tests.
- Keep controller sticks centered while the controller becomes active and neutral arming completes.
- Treat `/api/safety/reset` as an intentional operator action.
- Do not remove Bridge-loss, disconnect, impact, or tilt protections to simplify testing.
- Expect network interruption when switching `wlan0` into AP mode.
- Do not store secrets beyond local development credentials without adding an appropriate hardware-backed security design.

## Release identity

All shareable v8.4 project files should use a synchronized identity:

```text
App name: Robodog Pro v8.4
Runtime build ID: robodog-2026.08.12-v8.4-production
Matrix startup: ROBODOG READY
WebUI badge: v8.4
```

Version history belongs in external backups or release artifacts, not in this operational README.
