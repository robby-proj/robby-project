# Linux Gamepad Input

Reads `/dev/input/js*` in the dedicated `gamepad-reader` service and relays framed joystick events to `python/input_controller.py` over the internal App Lab Docker network. The main application retains neutral arming, hard-stop latching, debounce, hysteresis, disconnect STOP, and Bridge-loss STOP.
