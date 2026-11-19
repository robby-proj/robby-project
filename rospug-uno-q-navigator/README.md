# ROSPug UNO Q Console

An Arduino App Lab project that replaces the original Jetson-to-controller UART link and adds the factory USB remote, direct LD19 decoding, LiDAR-guarded driving, and autonomous obstacle avoidance.

This is **v0.7.0**. It uses self-contained App Lab Bricks for the USB gamepad and LD19, pins the same Zephyr/RouterBridge versions proven by Robodog Pro, and adds an optional persistent `ROSPug-Pro` Wi-Fi access point for an offline showcase dashboard. No separate sensor host service is required.

The UNO Q 7.0 kernel image used during validation enumerated the factory CH340 adapter but did not include `ch341.ko`, so no `/dev/ttyUSB0` was created. v0.7.0 retains the proven direct USB transport that configures `1a86:7523` at 230400 baud through Linux usbfs. It automatically prefers `/dev/ttyUSB*` when the kernel driver exists and otherwise uses direct USB; no kernel-module installation is required. The isolated LiDAR Brick joins supplementary group `0` so it can open the UNO Q's `0664 root:root` usbfs node read/write without elevating the main application.

## What this release does

- Opens UNO Q `Serial1` on D0/D1 at **500000 baud**.
- Validates ROSPug frames using the controller's CRC-8 algorithm.
- Shows link state, raw six-float IMU reports, CRC errors, and battery reports when available.
- Sends the verified ROSPug initialization command: `AA 55 0B 02 15 01 96`.
- Reads the position of bus servos 1–12 without commanding movement.
- Uses the original Hiwonder SDK's exact kinematics payloads for Stop, Gait, Pose, and Move.
- Includes Hiwonder's documented Trot, Amble, Walk, and neutral-pose values, plus Fast Trot and a lower-lift Smooth Fast Trot.
- Uses one coordinated, reverse-referenced envelope for forward, reverse, strafe, rotation, mixed commands, and Auto Avoid; sharp direction changes brake before the new gait vector is applied. Emergency and LiDAR hard stops remain immediate.
- Starts software-locked and rejects movement until valid controller frames are arriving.
- Refreshes movement while a direction button is held and automatically sends Stop after **650 ms** without a refresh.
- Reads the `2563:0526` wireless controller through an isolated `gamepad_input` Brick with explicit read-only `/dev/input` access.
- Reads LD19 packets through an isolated `lidar_input` Brick at **230400 baud**, validates LDROBOT CRC-8, and converts its clockwise angles to counterclockwise navigation angles.
- Displays a responsive top-down 360° LiDAR view with scan returns, selectable 1.5/3/6/12 m range, distance rings, front/rear orientation, safety zones, and commanded motion.
- Pins `arduino:zephyr:unoq` core **0.90.0**, RouterBridge **0.4.3**, RPClite **0.3.0**, and MsgPack **0.4.2** so RouterBridge and the D0/D1 `Serial1` ROSPug link can operate together.
- Provides Manual, LiDAR Guard, and Auto Avoid modes without requiring ROS on the UNO Q.
- Stops Auto mode if LiDAR data becomes stale, the remote disconnects, the UART controller goes offline, or the path is trapped.
- Cancels Auto when the operator deliberately moves a joystick beyond 32% or presses a directional control; ordinary stick-center drift is ignored.
- Includes host-side setup helpers that create a WPA-protected `ROSPug-Pro` access point at `10.42.0.1` while preserving the existing router profile for later use.

## Offline Wi-Fi access point

The Web UI already runs through the App Lab Web UI Brick. v0.7.0 adds a one-time UNO Q host configuration so a phone, tablet, or laptop can open the dashboard without a router.

Run the setup from a local UNO Q terminal, USB/ADB shell, or wired Ethernet session. Do not start it from Wi-Fi SSH unless you are prepared for that session to disconnect when `wlan0` changes from the current router network to access-point mode.

```bash
cd ~/ArduinoApps/rospug-uno-q-navigator-v0.7.0
bash scripts/setup_rospug_ap.sh
```

Defaults:

| Setting | Value |
|---|---|
| SSID | `ROSPug-Pro` |
| Password | `<your password>` |
| UNO Q address | `10.42.0.1` |
| Dashboard | `http://10.42.0.1:7000` |
| Band | 2.4 GHz by default; selectable 5 GHz |

An alternate SSID, password, and band can be supplied as positional arguments. WPA passwords must contain 8–63 characters. Use `2.4` or `5` for the third argument:

```bash
bash scripts/setup_rospug_ap.sh My-ROSPug 'ChooseASecretPassword' 5
```

To use the default name and password on 5 GHz:

```bash
bash scripts/setup_rospug_ap.sh ROSPug-Pro 'ROSPugDemo2026' 5
```

To switch the same profile back to the more widely compatible 2.4 GHz band, rerun it with `2.4`. The UNO Q's single `wlan0` radio exposes one selected band at a time; this configuration does not create simultaneous 2.4 GHz and 5 GHz SSIDs. Availability of 5 GHz access-point mode depends on the installed Wi-Fi hardware, driver, and regulatory domain. NetworkManager will report an activation error instead of silently falling back if 5 GHz AP mode is unavailable.

The existing infrastructure profile, such as `<your preferred wifi>`, is not deleted. To leave access-point mode and reconnect to it:

```bash
bash scripts/switch_to_wifi.sh <your preferred wifi>
```

To return to the access point later:

```bash
sudo nmcli connection modify ROSPug-AP connection.autoconnect yes
sudo nmcli connection up ROSPug-AP
```

For a standalone demonstration, also enable **Run at startup** for this App in App Lab. Wi-Fi access-point mode normally replaces the UNO Q's router Wi-Fi connection. AWS/TPS connectivity therefore requires wired Ethernet through the USB-C hub or a second compatible network adapter.

## USB connections

The USB gamepad receiver and LD19 USB-to-serial adapter were used on the Jetson only to identify their behavior. For this App Lab release, move both devices to the **UNO Q Linux side**, normally through a powered USB hub:

| USB device | Preferred path | Fallback |
|---|---|---|
| `2563:0526` wireless gamepad | `/dev/input/by-id/usb-2563_USB_WirelessGamepad-joystick` | `/dev/input/js0` |
| LD19 / CH340 `1a86:7523` | `/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0` | Direct USB (`/dev/bus/usb`) when `ch341.ko` is absent |

Do not leave the LD19 serial adapter open on the Jetson and UNO Q simultaneously. A USB device can be attached to only one host.

The Web UI reports the resolved device paths, LD19 packet count, CRC errors, scan age, rotation speed, nearest distance in four sectors, and up to 180 current scan points. The visualization is a robot-centered local obstacle view, not accumulated SLAM or global localization.

## App Lab input Bricks

The main App Lab Python container intentionally does not open arbitrary host devices. v0.7.0 deploys two local sidecars automatically:

| Brick service | Host access | Internal output |
|---|---|---|
| `gamepad-reader` | `/dev/input` read-only, character major 13 | Linux joystick events on TCP 8765 |
| `lidar-reader` | Host serial major 188 or direct USB major 189 for CH340 only | CRC-validated sector JSON on TCP 8766 |

The sidecars cannot command the ROSPug. All arming, motion arbitration, watchdogs, and obstacle decisions remain in the main application and MCU sketch.

If v0.3.1's temporary host service was previously installed, disable it once before starting v0.7.0 so it cannot compete for the LD19 serial stream:

```bash
sudo systemctl disable --now rospug-host-io 2>/dev/null || true
```

## Navigation modes

| Mode | Behavior |
|---|---|
| Manual | Direct web/gamepad commands. LiDAR protection is bypassed. |
| LiDAR Guard | Operator remains in control. Only the commanded direction is limited: a side obstacle does not block forward/reverse travel, and a front obstacle does not block a clear reverse command. This is the default. |
| Auto Avoid | Uses directional clearance to select forward travel, a left/right turning route, or a rear escape. Empty sectors are treated as open only while the overall scan remains healthy. |

Safety thresholds are intentionally conservative:

- Forward hard stop: **0.32 m**
- Forward slowdown zone: **0.32–0.75 m**
- Auto turn decision: **0.36 m**
- Auto cautious zone: **0.36–0.70 m**
- Auto usable side route: **0.38 m**
- Auto rear escape: **0.46 m**, stopping at **0.30 m**
- Auto trapped-route threshold: **0.22 m**
- Auto minimum recent scan: **18 points**
- LiDAR stale timeout: **500 ms**
- Controller motion watchdog: **650 ms**

Auto Avoid can be entered only after explicit Web UI confirmation and only while the robot, gamepad, and LD19 are online. A fresh scan with at least 18 recent points is required. Within a healthy scan, a sector with no return is considered open rather than blocking the robot. Auto prefers forward travel, turns toward the clearer side when front is blocked, uses a short reverse escape when both turns are blocked and rear is clear, and stops only when every available route is too tight.

If the physical LiDAR zero-degree direction is not facing the robot's front, set `ROSPUG_LIDAR_FRONT_OFFSET_DEG` for the `lidar-reader` service. Start at `180` if the reported front and rear sectors appear swapped. The default is `0` in `bricks/lidar_input/brick_compose.yaml`.

The gamepad requires one real control event and then 0.45 seconds with all movement controls centered before it reports **Gamepad online**. This prevents an off-center stick from moving the robot when the app starts.

## Factory remote mapping

| Control | Function |
|---|---|
| Left stick Y / D-pad up-down | Forward / backward |
| Left stick X | Strafe left / right |
| Right stick X / D-pad left-right | Rotate left / right |
| Square / Circle | Strafe left / right |
| Cross / Triangle | Raise / lower body |
| L1 / L2 | Pitch up / down |
| R1 / R2 | Roll left / right |
| START | Stop and return to neutral pose |
| SELECT | Toggle Manual / LiDAR Guard; it never starts Auto |
| L3 | Immediate protocol-level Stop and return to Guard |

## Verified motion mapping

All kinematics commands use controller function `0x0B`.

| Operation | Subcommand | Payload |
|---|---:|---|
| Set pose | `0x09` | `x, y, z, x_shift, pitch, roll, duration` as 7 little-endian floats |
| Set gait | `0x0B` | `overlap, swing, clearance, z_clearance` as 4 little-endian floats |
| Move | `0x0D` | `x, y, yaw` as 3 little-endian floats |
| Stop | `0x0F` | `duration` as 1 little-endian float |
| Set ROSPug type | `0x15` | `0x01` |

The UI uses conservative movement values within the limits enforced by Hiwonder's ROS node:

- Web forward/backward: `x = ±0.09`
- Web left/right: `y = ±0.07`
- Web rotate: `yaw = ±0.30`
- Remote full-stick forward/backward: `x = ±0.10`
- Auto open-path forward: `x = 0.10` (controller maximum), with proportional slowdown inside the caution zone
- MCU hard limits: `x/y = ±0.10`, `yaw = ±0.40`

Factory example presets:

| Preset | overlap | swing | clearance | step height |
|---|---:|---:|---:|---:|
| Trot | 0.20 | 0.18 | 0.02 | 0.04 |
| Amble | 0.20 | 0.15 | 0.05 | 0.04 |
| Walk | 0.24 | 0.12 | 0.05 | 0.04 |
| Fast Trot | 0.14 | 0.14 | 0.00 | 0.04 |
| Smooth Fast Trot | 0.14 | 0.13 | 0.00 | 0.03 |

Smooth Fast Trot uses a slightly shorter swing and a lower 0.03 m foot lift to reduce the vertical landing impact while retaining the full verified movement range. v0.7.0 normalizes forward/reverse, strafe, and yaw before applying a common 25% acceleration and 40% ordinary-braking envelope. Combined commands therefore stay synchronized instead of allowing each axis to reach its target at a different time. A sharp direction change brakes the current vector before applying the next one. Emergency, stale-sensor, controller-loss, and LiDAR hard stops bypass the envelope. Test it with the robot supported before floor operation; revert to factory Trot if feet slip, the body bounces, or servos sound strained.

The neutral pose is `x=0`, `y=0`, `z=-0.13 m`, `x_shift=0.01 m`, `roll=0`, `pitch=0`, and `duration=0.5 s`.

## Wiring

The Jetson Nano must be completely disconnected while the UNO Q supplies the adapter's 3.3 V logic rail.

| Hiwonder adapter | Arduino UNO Q |
|---|---|
| Pin 1 — 3.3 V | 3.3 V |
| Pin 6 — GND | GND |
| Pin 8 — TX | D0 / RX |
| Pin 10 — RX | D1 / TX |
| Pin 2 — 5 V | **Disconnected** |

SDA and SCL are not required for ROSPug control. The previously detected I2C address `0x3C` is the OLED, not the motion controller.

## Safety

1. Support the robot so its feet cannot catch the table.
2. Keep hands, loose wires, and tools clear of all 12 joints.
3. Power the ROSPug and wait for the controller to boot.
4. Start this App Lab project and confirm that **ROSPug online** appears.
5. Check the safety box before pressing **Initialize & Stand**.
6. Press **Neutral pose** before the first walking test.
7. Verify **Gamepad online** and **LD19 live**, and confirm the front distance changes when an object is placed in front of the LiDAR.
8. Start with Trot and briefly press-and-hold Forward in **LiDAR Guard** while the robot is supported.
9. Release the button and verify that it stops immediately.
10. Test Auto Avoid while the robot is still supported before placing it on the floor.

The red **STOP** button sends the ROSPug protocol-level Stop command. **Software Lock** sends Stop and then blocks future movement commands. Neither action turns off servo torque or removes power. Use the robot's physical power switch as the emergency stop.

Recommended shutdown order:

1. Switch off robot power.
2. Stop or power off the UNO Q.
3. Disconnect wiring.

Never reinstall the Jetson while the UNO Q 3.3 V wire is connected to adapter pin 1.

## Install in App Lab

1. In Arduino App Lab, import `ROSPug UNO Q Navigator v0.7.0.zip` manually.
2. If v0.3.1 was previously tested, disable its old `rospug-host-io` service using the command above.
3. Connect the gamepad receiver and LD19 USB adapter to the UNO Q, preferably through a powered hub.
4. Press **Run**. App Lab builds and flashes the pinned MCU sketch, deploys both input Bricks, and starts the Web UI.
5. Move a gamepad control once, return all sticks/D-pad to neutral, and wait approximately one second for **Gamepad online**.
6. Confirm **ROSPug online** and **LD19 live** before initialization or movement.

If the UI reports a Bridge timeout, inspect the App Lab build console and verify that Zephyr 0.90.0 and RouterBridge 0.4.3 compiled and the sketch upload completed. Also ensure no second App Lab app is running against the MCU.

## Motion safety design

Movement is a dead-man control: the browser intent expires after 600 ms, the Linux supervisor refreshes accepted motion every 220 ms, and the MCU independently sends Stop after 650 ms without a refresh. Releasing the browser button, centering the gamepad, losing the controller UART, or violating the active mode's safety condition produces a Stop.

The LD19 parser accepts only the official 47-byte `0x54 0x2C` frame structure with a valid CRC-8. It uses the second-nearest valid reflection in each sector to reject a single isolated noise return without ignoring narrow two-sample obstacles.
