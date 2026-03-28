# 😀 Running UNO Q Local LLM Chat

Local LLM chat demo for **Arduino UNO Q** using: **App Lab WebUI** for the browser interface, a **host-side Python backend** on Linux, and a **yzma** model

## Architecture

```text
App Lab WebUI (port 7000)
        |
        v
Host backend (port 8090)
        |
        v
yzma local chat example
        |
        v
SmolLM2-135M-Instruct-Q4_K_M.gguf


