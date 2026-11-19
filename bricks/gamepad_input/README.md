# Linux Gamepad Input

Reads `/dev/input/js*` in an isolated App Lab service and relays Linux joystick
events to the main application over the internal Docker network. The main
application retains all ROSPug movement arbitration and safety behavior.
