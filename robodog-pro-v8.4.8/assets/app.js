// ============================================================================
// Robodog Pro v8.4.6 - WebUI frontend
//
// Features:
// - Live accelerometer and vibration graph
// - Motion policy and diagnostic information
// - Confirmed-event handling without classifying one sample as an impact
// - Manual, autonomous IMU, and autonomous vision ownership
// - Camera streaming and camera-health diagnostics
// - Vision target summary and target reacquisition controls
// - OLED eyes status and controls
// - Controller, audio, Wi-Fi, orchestrator, and watchdog diagnostics
// - RAM-only camera display
// ============================================================================

"use strict";

(function () {
  // ==========================================================================
  // DOM and API helpers
  // ==========================================================================

  const $ = (id) => document.getElementById(id);

  const setText = (id, value) => {
    const element = $(id);
    if (element) {
      element.textContent = value;
    }
  };

  const AP_URL = "http" + "://10.42.0.1:7000";
  const AP_HOST_COMMAND_BASE =
    "$HOME/ArduinoApps/robodog-pro-v8-4-6/host/dogzilla-ap.sh ";

  async function api(path) {
    try {
      const response = await fetch(path, { cache: "no-store" });
      const contentType = response.headers.get("content-type") || "";

      if (!contentType.includes("application/json")) {
        return {
          ok: false,
          error: "Unexpected response type from " + path,
          http_status: response.status
        };
      }

      const payload = await response.json();

      if (!response.ok && payload && payload.ok !== false) {
        payload.ok = false;
        payload.http_status = response.status;
      }

      return payload;
    } catch (error) {
      return {
        ok: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  function addChat(role, text) {
    const chatElement = $("chat");
    if (!chatElement) {
      return;
    }

    const message = document.createElement("div");
    message.className = "msg " + String(role);

    const label = document.createElement("b");
    label.textContent = String(role) + ": ";

    const content = document.createTextNode(String(text));

    message.appendChild(label);
    message.appendChild(content);

    chatElement.appendChild(message);
    chatElement.scrollTop = chatElement.scrollHeight;
  }

  function formatNumber(value, digits = 2, fallback = "-") {
    if (
      value === null ||
      value === undefined ||
      value === "" ||
      Number.isNaN(Number(value))
    ) {
      return fallback;
    }
    return Number(value).toFixed(digits);
  }

  function formatBoolean(value, trueText = "yes", falseText = "no") {
    return value ? trueText : falseText;
  }

  // ==========================================================================
  // Live accelerometer graph
  // ==========================================================================

  let canvas = null;
  let context = null;

  const MAX_POINTS = 160;
  const GRAPH_RANGE_G = 2.0;

  const dataX = [];
  const dataY = [];
  const dataZ = [];
  const dataVibration = [];

  let graphFrozen = false;
  let graphHitUntil = 0;

  let lastVibration = 0;
  let vibrationPeak = 0;
  let vibrationPeakAt = 0;

  const VIBRATION_PEAK_HOLD_MS = 1500;
  const VIBRATION_PEAK_DECAY = 0.6;

  function graphEnabled() {
    return Boolean(context);
  }

  function pushSample(x, y, z, vibration) {
    if (!graphEnabled() || graphFrozen || Date.now() < graphHitUntil) {
      return;
    }

    dataX.push(x);
    dataY.push(y);
    dataZ.push(z);
    dataVibration.push(vibration);

    if (dataX.length > MAX_POINTS) {
      dataX.shift();
      dataY.shift();
      dataZ.shift();
      dataVibration.shift();
    }
  }

  function drawSeries(values, color, scale) {
    if (!canvas || !context || values.length === 0) {
      return;
    }

    const width = canvas.width;
    const height = canvas.height;
    const midpoint = height / 2;

    context.strokeStyle = color;
    context.lineWidth = 1.6;
    context.beginPath();

    for (let index = 0; index < values.length; index += 1) {
      const x = (index / (MAX_POINTS - 1)) * width;
      const y = midpoint - (values[index] / scale) * (height / 2 - 8);

      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    }

    context.stroke();
  }

  function drawGraph() {
    if (!graphEnabled() || !canvas || !context) {
      return;
    }

    const width = canvas.width;
    const height = canvas.height;
    const midpoint = height / 2;

    context.clearRect(0, 0, width, height);

    context.strokeStyle = "#1e293b";
    context.lineWidth = 1;
    context.fillStyle = "#334155";
    context.font = "10px monospace";

    for (
      let graphValue = -GRAPH_RANGE_G;
      graphValue <= GRAPH_RANGE_G;
      graphValue += 1
    ) {
      const y = midpoint - (graphValue / GRAPH_RANGE_G) * (height / 2 - 8);

      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(width, y);
      context.stroke();

      context.fillText(graphValue.toFixed(0), 2, y - 2);
    }

    drawSeries(dataX, "#38bdf8", GRAPH_RANGE_G);
    drawSeries(dataY, "#22c55e", GRAPH_RANGE_G);
    drawSeries(dataZ, "#f97316", GRAPH_RANGE_G);
    drawSeries(dataVibration, "#a78bfa", GRAPH_RANGE_G);

    requestAnimationFrame(drawGraph);
  }

  function flashConfirmedEvent(reason) {
    graphHitUntil = Date.now() + 2000;

    const flashElement = $("graph_flash");
    const stateElement = $("graph_state");

    if (flashElement) {
      flashElement.classList.add("hit");
    }

    if (stateElement) {
      stateElement.classList.add("hit");
      stateElement.textContent = reason || "CONFIRMED";
    }

    setTimeout(() => {
      if (flashElement) {
        flashElement.classList.remove("hit");
      }
      if (stateElement && !graphFrozen) {
        stateElement.classList.remove("hit");
        stateElement.textContent = "idle";
      }
    }, 2000);
  }

  window.toggleFreeze = function toggleFreeze() {
    graphFrozen = !graphFrozen;

    const stateElement = $("graph_state");
    const button = $("freeze_btn");

    if (stateElement) {
      stateElement.classList.toggle("frozen", graphFrozen);
      stateElement.textContent = graphFrozen ? "frozen" : "idle";
    }

    if (button) {
      button.textContent = graphFrozen ? "▶ Resume" : "⏸ Freeze";
    }
  };

  window.clearGraph = function clearGraph() {
    dataX.length = 0;
    dataY.length = 0;
    dataZ.length = 0;
    dataVibration.length = 0;
  };

  // ==========================================================================
  // Event banner
  // ==========================================================================

  let eventUntil = 0;

  const EVENT_COLORS = {
    manual: "evt-green",
    autonomous: "evt-blue",
    walkaround: "evt-blue",
    detected: "evt-cyan",
    searching: "evt-amber",
    wall: "evt-red",
    impact: "evt-red",
    lowbatt: "evt-purple"
  };

  function noteEvent(text, kind, holdMs = 3000) {
    const banner = $("event_banner");
    const bannerText = $("event_text");

    if (!banner || !bannerText) {
      return;
    }

    bannerText.textContent = text;
    banner.className = "event active " + (EVENT_COLORS[kind] || "");

    void banner.offsetWidth;
    banner.classList.add("active");

    eventUntil = Date.now() + holdMs;

    setTimeout(() => {
      if (Date.now() >= eventUntil) {
        banner.className = "event idle";
        bannerText.textContent = "Idle";
      }
    }, holdMs + 50);
  }

  window.noteEvent = noteEvent;

  // ==========================================================================
  // Live state tracking
  // ==========================================================================

  let previousMode = null;
  let previousImpacts = 0;
  let previousPhase = "";
  let previousMood = "";

  let liveMode = "manual";
  let liveMood = "normal";

  function effectiveLedMood() {
    if (
      liveMood === "alert" ||
      liveMood === "detected" ||
      liveMood === "searching" ||
      liveMood === "low_energy" ||
      liveMood === "sleeping"
    ) {
      return liveMood;
    }

    if (liveMode === "autonomous_imu" || liveMode === "autonomous_vision") {
      return "happy";
    }

    return "normal";
  }

  function updateLedLegend() {
    const effectiveMood = effectiveLedMood();
    const rows = document.querySelectorAll(".led-legend tr[data-mood]");

    rows.forEach((row) => {
      row.classList.toggle(
        "active",
        row.getAttribute("data-mood") === effectiveMood
      );
    });
  }

  // ==========================================================================
  // State and autonomy
  // ==========================================================================

  async function refreshState() {
    const state = await api("/api/state");

    if (!state || state.ok === false) {
      setText(
        "status",
        "Backend not reachable: " +
          ((state && state.error) || "unknown error")
      );
      return;
    }

    if (state.mode) {
      setText(
        "status",
        "Mode: " + state.mode + " | Last: " + (state.last_command || "none")
      );
    }

    setText("wifi_ssid", state.wifi_ssid || "DOGZILLA-UNOQ");
    setText("wifi_password", state.wifi_password || "dogzilla123");
    setText("wifi_url", state.wifi_url || AP_URL);
    setText("wifi_ap", state.wifi_ap || "unknown");

    const chatElement = $("chat");

    if (chatElement && Array.isArray(state.chat)) {
      chatElement.replaceChildren();
      for (const message of state.chat) {
        addChat(message.role || "robot", message.text || "");
      }
    }

    if (state.sequence) {
      const sequence = state.sequence;

      if (sequence.running) {
        setText("sequence_status", "Running: " + (sequence.step || "unknown"));
      } else if (sequence.step === "cancelled") {
        setText("sequence_status", "Cancelled.");
      } else if (sequence.step === "complete") {
        setText("sequence_status", "Complete.");
      } else {
        setText("sequence_status", "Sequence idle.");
      }
    }

    if (state.mode) {
      if (state.mode !== liveMode) {
        if (previousMode !== null) {
          let eventLabel = "MANUAL";
          let eventKind = "manual";

          if (state.mode === "autonomous_vision") {
            eventLabel = "AUTONOMOUS VISION";
            eventKind = "autonomous";
          } else if (state.mode === "autonomous_imu") {
            eventLabel = "AUTONOMOUS IMU";
            eventKind = "autonomous";
          }

          noteEvent(eventLabel, eventKind);
        }
        previousMode = state.mode;
      }

      liveMode = state.mode;
      setText("autonomy_mode", state.mode);
      updateLedLegend();
    }

    if (state.autonomy) {
      const autonomy = state.autonomy;

      let owner = autonomy.owner;
      if (!owner) {
        if (state.mode === "autonomous_vision") {
          owner = "vision";
        } else if (state.mode === "autonomous_imu") {
          owner = "imu";
        } else {
          owner = "manual";
        }
      }

      setText("autonomy_owner", owner);
      setText("autonomy_phase", autonomy.phase || "idle");

      if (autonomy.active) {
        setText(
          "autonomy_status",
          "Active: " +
            (autonomy.phase || "unknown") +
            " (" +
            (autonomy.reason || "") +
            ") vib=" +
            (autonomy.vibration != null ? autonomy.vibration : "-")
        );
      } else {
        setText("autonomy_status", "Autonomy idle.");
      }

      setText("anom_impacts", autonomy.impacts != null ? autonomy.impacts : 0);
      setText("anom_z", autonomy.z != null ? autonomy.z : 0);
      setText("anom_mean", autonomy.vib_mean != null ? autonomy.vib_mean : 0);
      setText("anom_src", autonomy.anom_src || "-");
      setText("lz_score", autonomy.z != null ? autonomy.z : 0);
      setText(
        "graph_src",
        autonomy.anom_src === "edge_impulse" ? "EI model" : "z-score"
      );

      const phase = String(autonomy.phase || "");
      const impacts = autonomy.impacts != null ? autonomy.impacts : 0;

      if (impacts > previousImpacts) {
        noteEvent("WALL HIT #" + impacts, "wall", 3500);
        flashConfirmedEvent("OBSTACLE");
        previousImpacts = impacts;
      }

      if (phase !== previousPhase) {
        if (phase.startsWith("recover_wall")) {
          noteEvent("OBSTACLE, turning", "wall", 3500);
          flashConfirmedEvent("OBSTACLE");
        } else if (phase.startsWith("recover_tilt")) {
          noteEvent("TILT recovery", "wall", 3000);
          flashConfirmedEvent("TILT");
        } else if (phase === "sniff") {
          noteEvent("Sniffing around", "searching", 2000);
        }
        previousPhase = phase;
      }
    }
  }

  // ==========================================================================
  // System health
  // ==========================================================================

  async function refreshHealth() {
    const health = await api("/api/health");
    if (!health || !health.ok) {
      return;
    }

    setText("diag_ip", health.advertised_ip || "-");
    setText(
      "diag_firmware",
      health.hardware && health.hardware.dog_firmware
        ? health.hardware.dog_firmware
        : "?"
    );

    const currentMood =
      health.mood && health.mood.current ? health.mood.current : "?";

    setText("diag_mood", currentMood);
    liveMood = currentMood;
    updateLedLegend();

    if (currentMood !== previousMood) {
      if (previousMood && currentMood === "detected") {
        noteEvent("PERSON detected", "detected", 3000);
      } else if (previousMood && currentMood === "searching") {
        noteEvent("Searching", "searching", 2000);
      } else if (previousMood && currentMood === "low_energy") {
        noteEvent("LOW BATTERY", "lowbatt", 4000);
      } else if (previousMood && currentMood === "alert") {
        noteEvent("ALERT", "wall", 3000);
      }
      previousMood = currentMood;
    }

    if (health.bridge && health.bridge.ok) {
      setText("diag_bridge", "OK");
    } else {
      const errorCount =
        health.bridge && health.bridge.consecutive_errors !== undefined
          ? health.bridge.consecutive_errors
          : "?";
      setText("diag_bridge", "ERR (" + errorCount + " errors)");
    }

    if (health.camera && health.camera.ok) {
      setText("diag_camera", "OK (" + (health.camera.fps || 0) + " fps)");
    } else {
      setText(
        "diag_camera",
        "ERR (" +
          (health.camera && health.camera.error
            ? health.camera.error
            : "unknown") +
          ")"
      );
    }

    setText("diag_uptime", health.uptime_s || 0);
    if (health.eyes && health.eyes.bus_fault) {
      noteEvent("EYES OLED BUS RECOVERY", "wall", 3000);
    }
    setText(
      "diag_captures",
      health.captures_persisted ? "YES, persistence violation" : "NO"
    );
  }

  // ==========================================================================
  // Controller
  // ==========================================================================

  async function refreshController() {
    const response = await api("/api/controller/status");
    if (!response || response.ok === false) {
      return;
    }

    const controller = response.controller || {};

    if (controller.connected) {
      setText(
        "controller_status",
        "Connected: " +
          (controller.name || controller.device || "controller") +
          " (" +
          (controller.api || "unknown API") +
          ") | events: " +
          (controller.events || 0) +
          " | last: " +
          (controller.last_command || "none")
      );
    } else {
      setText(
        "controller_status",
        "Not connected: " + (controller.error || "scanning")
      );
    }

    const details = $("controller_details");
    if (details) {
      const lines = [
        "Receiver present: " + formatBoolean(controller.receiver_present),
        "Controller active: " + formatBoolean(controller.controller_active),
        "Armed: " + formatBoolean(controller.armed),
        "Hard stop latched: " + formatBoolean(controller.hard_stop_latched),
        "Owner: " + (controller.owner || "stopped"),
        "Safety blocked: " + formatBoolean(controller.safety_blocked),
        "Safety reason: " + (controller.safety_reason || "none")
      ];
      details.textContent = lines.join(" | ");
    }
  }

  // ==========================================================================
  // Camera
  // ==========================================================================

  async function refreshCamera() {
    const camera = await api("/api/camera");

    const image = $("camera_view");
    const status = $("camera_status");

    if (!camera || !camera.ok || !camera.frame_base64) {
      if (status) {
        status.textContent =
          "Camera: " + ((camera && camera.error) || "no frame available");
      }
      if (image) {
        image.removeAttribute("src");
      }
      return;
    }

    if (image) {
      image.src = "data:image/jpeg;base64," + camera.frame_base64;
    }

    if (status) {
      status.textContent = "Camera running | FPS: " + (camera.fps || 0);
    }
  }

  async function refreshCameraHealth() {
    const response = await api("/api/camera_health");
    if (!response || response.ok === false) {
      return;
    }

    const camera = response.camera || response;

    setText(
      "camera_node",
      camera.index === null || camera.index === undefined
        ? "-"
        : "/dev/video" + camera.index
    );

    setText(
      "camera_age",
      camera.age_s === null || camera.age_s === undefined
        ? "-"
        : formatNumber(camera.age_s, 1) + " s"
    );

    setText(
      "camera_reopens",
      camera.reopens != null ? camera.reopens : "-"
    );

    setText("camera_rotation", camera.rotate_180 ? "180 degrees" : "off");
  }

  // ==========================================================================
  // Motion and vibration
  // ==========================================================================

  async function refreshVibration() {
    const vibrationState = await api("/api/vibration");
    if (!vibrationState || !vibrationState.ok) {
      return;
    }

    const vibration = Number(vibrationState.vibration || 0);
    const x = Number(vibrationState.x || 0);
    const y = Number(vibrationState.y || 0);
    const z = Number(vibrationState.z || 0);

    setText("vibration_value", vibration.toFixed(3));
    setText("accel_x", x.toFixed(3));
    setText("accel_y", y.toFixed(3));
    setText("accel_z", z.toFixed(3));
    setText("lx", x.toFixed(3));
    setText("ly", y.toFixed(3));
    setText("lz", z.toFixed(3));

    const liveVibration = $("lv");
    if (liveVibration) {
      liveVibration.textContent = vibration.toFixed(3);
      liveVibration.className =
        "v" + (vibration >= 1.2 ? " bad" : vibration >= 0.4 ? " warn" : "");
    }

    const now = Date.now();
    if (vibration >= vibrationPeak) {
      vibrationPeak = vibration;
      vibrationPeakAt = now;
    } else if (now - vibrationPeakAt > VIBRATION_PEAK_HOLD_MS) {
      vibrationPeak = Math.max(vibration, vibrationPeak * VIBRATION_PEAK_DECAY);
      if (vibrationPeak < 0.01) {
        vibrationPeak = 0;
      }
    }

    setText("lpk", vibrationPeak.toFixed(3));

    const peakElement = $("lpk");
    if (peakElement) {
      peakElement.className =
        "v" +
        (vibrationPeak >= 1.2 ? " bad" : vibrationPeak >= 0.4 ? " warn" : "");
    }

    const vibrationValue = $("vibration_value");
    if (vibrationValue) {
      vibrationValue.classList.toggle("peak", vibrationPeak >= 1.2);
    }

    // A single high sample is diagnostic information, not a confirmed impact.
    if (vibration >= 1.2 && lastVibration < 1.2) {
      noteEvent("HIGH SAMPLE " + vibration.toFixed(2) + "g", "searching", 1800);
    }

    lastVibration = vibration;
    pushSample(x, y, z, vibration);

    const vibrationStatus = $("vibration_status");
    if (vibrationStatus) {
      if (vibration < 0.05) {
        vibrationStatus.textContent = "Stable / idle";
        vibrationStatus.className = "vibration-status stable";
      } else if (vibration < 0.4) {
        vibrationStatus.textContent = "Normal movement";
        vibrationStatus.className = "vibration-status normal";
      } else if (vibration < 1.2) {
        vibrationStatus.textContent = "High vibration";
        vibrationStatus.className = "vibration-status warning";
      } else {
        vibrationStatus.textContent = "High sample, awaiting confirmation";
        vibrationStatus.className = "vibration-status danger";
      }
    }
  }
  
async function refreshMotionDiagnostics() {
const diagnostics = await api("/api/motion_diagnostics");
if (!diagnostics || diagnostics.ok === false) {
  return;
}

setText("motion_present", diagnostics.present ? "yes" : "no");
setText("motion_responds", diagnostics.responds ? "yes" : "no");

setText(
  "motion_addr",
  diagnostics.addr === null || diagnostics.addr === undefined
	? "-"
	: "0x" + Number(diagnostics.addr).toString(16).toUpperCase()
);
    setText(
      "motion_sensitivity",
      diagnostics.motion_sensitivity != null
        ? diagnostics.motion_sensitivity
        : "-"
    );

    setText(
      "motion_threshold",
      diagnostics.threshold_g === null || diagnostics.threshold_g === undefined
        ? "-"
        : formatNumber(diagnostics.threshold_g, 2)
    );

    setText(
      "motion_local_stop",
      diagnostics.local_stop_enabled ? "ENABLED" : "disabled"
    );

    setText(
      "motion_mag",
      diagnostics.movement_mag === null || diagnostics.movement_mag === undefined
        ? "-"
        : formatNumber(diagnostics.movement_mag, 4)
    );

    setText(
      "motion_vibration",
      diagnostics.vibration === null || diagnostics.vibration === undefined
        ? "-"
        : formatNumber(diagnostics.vibration, 4)
    );

    setText(
      "motion_confirm",
      diagnostics.impact_confirm_count != null
        ? diagnostics.impact_confirm_count
        : "-"
    );

    setText(
      "motion_cooldown",
      diagnostics.impact_cooldown_remaining_ms != null
        ? diagnostics.impact_cooldown_remaining_ms
        : "-"
    );

    setText(
      "motion_reads",
      diagnostics.motion_reads_count != null
        ? diagnostics.motion_reads_count
        : "-"
    );

    setText(
      "motion_active",
      diagnostics.motion_active_count != null
        ? diagnostics.motion_active_count
        : "-"
    );

    if (diagnostics.interpretation) {
      setText("motion_note", diagnostics.interpretation);
    }
  }

  // ==========================================================================
  // DOGZILLA telemetry
  // ==========================================================================

  async function refreshTelemetry() {
    const telemetry = await api("/api/telemetry");
    if (!telemetry || !telemetry.ok) {
      return;
    }

    setText("imu_roll", formatNumber(telemetry.roll, 2));
    setText("imu_pitch", formatNumber(telemetry.pitch, 2));
    setText("imu_yaw", formatNumber(telemetry.yaw, 2));
    setText(
      "imu_battery",
      telemetry.battery === null || telemetry.battery === undefined
        ? "-"
        : telemetry.battery
    );
  }

  // ==========================================================================
  // OLED eyes
  // ==========================================================================

  async function refreshEyesStatus() {
    const eyes = await api("/api/eyes_status");
    if (!eyes || eyes.ok === false) {
      return;
    }

    setText("eyes_present", eyes.present ? "yes" : "no");

    setText(
      "eyes_addr",
      eyes.addr_hex ||
        (eyes.addr === null || eyes.addr === undefined ? "-" : eyes.addr)
    );

    setText("eyes_mode", eyes.mode != null ? eyes.mode : "-");
    setText("eyes_flush_ok", eyes.flush_ok_count != null ? eyes.flush_ok_count : "-");
    setText("eyes_flush_errors", eyes.flush_error_count != null ? eyes.flush_error_count : "-");
    setText("eyes_bus_fault", eyes.bus_fault === true ? "FAULT" : eyes.bus_fault === false ? "clear" : "-");
    setText("eyes_bus_hz", eyes.bus_hz != null ? eyes.bus_hz : "-");

    const motionReactive =
      eyes.motion_reactive_requested != null
        ? eyes.motion_reactive_requested
        : eyes.motion_reactive;

    setText("eyes_motion", motionReactive ? "enabled" : "disabled");
  }

  // ==========================================================================
  // Vision summary
  // ==========================================================================

  async function refreshVisionSummary() {
    const vision = await api("/api/vision_summary");
    if (!vision || vision.ok === false) {
      return;
    }

    setText("vision_target", vision.target || "person");
    setText("vision_found", vision.found_target ? "yes" : "no");

    setText(
      "vision_confidence",
      vision.confidence === null || vision.confidence === undefined
        ? "-"
        : (Number(vision.confidence) * 100).toFixed(1) + "%"
    );

    setText(
      "vision_labels",
      Array.isArray(vision.labels) && vision.labels.length > 0
        ? vision.labels.join(", ")
        : "none"
    );

    setText("vision_summary", vision.summary || "No vision summary available.");
  }

  // ==========================================================================
  // Vision reacquisition
  // ==========================================================================

  async function refreshReacquireStatus() {
    const response = await api("/api/autonomy/reacquire_status");
    if (!response || response.ok === false) {
      return;
    }

    setText("reacquire_enabled", response.enabled ? "on" : "off");

    setText(
      "follow_profile",
      response.follow_profile && response.follow_profile.name
        ? response.follow_profile.name
        : "safe"
    );

    setText(
      "reacquire_action",
      response.state && response.state.last_lost_action
        ? response.state.last_lost_action
        : "none"
    );
  }

  // ==========================================================================
  // Orchestrator and watchdog
  // ==========================================================================

  async function refreshOrchestrator() {
    const response = await api("/api/orchestrator");
    if (!response || response.ok === false) {
      return;
    }

    const orchestrator = response.orchestrator || {};

    setText("diag_build", orchestrator.build_id || "-");
    setText(
      "diag_routes",
      orchestrator.routes_registered != null
        ? orchestrator.routes_registered
        : "-"
    );
    setText(
      "diag_workers",
      orchestrator.workers_registered != null
        ? orchestrator.workers_registered
        : "-"
    );

    const restarts = orchestrator.watchdog_restarts || {};
    setText(
      "diag_restarts",
      Object.keys(restarts).length > 0 ? JSON.stringify(restarts) : "none"
    );
  }

  // ==========================================================================
  // Public action handlers
  // ==========================================================================

  window.cmd = async function command(name) {
    addChat("owner", "Command: " + name);
    await api("/api/cmd/" + encodeURIComponent(name));
    await refreshState();
    await refreshVibration();
  };

  window.action = async function action(id) {
    addChat("owner", "Action " + id);
    await api("/api/action/" + id);
    await refreshState();
    await refreshVibration();
  };

  window.mode = async function setMode(name) {
    await api("/api/mode/" + encodeURIComponent(name));
    await refreshState();
  };

  window.motionSensitivity = async function motionSensitivity(level) {
    await api("/api/motion_sensitivity/" + encodeURIComponent(level));
    await refreshMotionDiagnostics();
  };

  window.motionRebegin = async function motionRebegin() {
    const result = await api("/api/motion_rebegin");
    if (result && result.ok !== false) {
      noteEvent("Movement re-begin complete", "searching", 2000);
    }
    await refreshMotionDiagnostics();
  };

  window.scanWire1 = async function scanWire1() {
    const result = await api("/api/wire1_scan_count");
    setText(
      "wire1_count",
      result &&
        result.wire1_scan_count !== null &&
        result.wire1_scan_count !== undefined
        ? result.wire1_scan_count
        : "error"
    );
  };

  window.eyesMode = async function eyesMode(name) {
    await api("/api/eyes_mode/" + encodeURIComponent(name));
    await refreshEyesStatus();
  };

  window.eyesMotion = async function eyesMotion(enabled) {
    await api("/api/eyes_motion/" + (enabled ? "true" : "false"));
    await refreshEyesStatus();
  };

  window.followProfile = async function followProfile(name) {
    await api("/api/autonomy/follow_config/" + encodeURIComponent(name));
    await refreshReacquireStatus();
  };

  window.setReacquire = async function setReacquire(enabled) {
    await api("/api/autonomy/reacquire/" + (enabled ? "true" : "false"));
    await refreshReacquireStatus();
  };

  window.controllerEnable = async function controllerEnable(enabled) {
    await api("/api/gamepad/" + (enabled ? "true" : "false"));
    await refreshController();
  };

  window.controllerRescan = async function controllerRescan() {
    await api("/api/controller/rescan");
    await refreshController();
  };

  window.walkaroundStart = async function walkaroundStart() {
    addChat("owner", "Start walkaround");
    await api("/api/sequence/walkaround");
    await refreshState();
  };

  window.walkaroundStop = async function walkaroundStop() {
    addChat("owner", "Cancel walkaround");
    await api("/api/sequence/stop");
    await refreshState();
  };

  window.sfx = async function soundEffect(event) {
    await api("/api/audio/test/" + encodeURIComponent(event));
  };

  window.audioMute = async function audioMute(enabled) {
    await api("/api/audio/mute/" + (enabled ? "true" : "false"));
  };

  window.copyApCommand = async function copyApCommand(action) {
    const command = AP_HOST_COMMAND_BASE + action;
    try {
      await navigator.clipboard.writeText(command);
      addChat("robot", "Copied host command: " + command);
    } catch (error) {
      window.prompt("Copy this UNO Q host command:", command);
    }
  };

  window.cameraReset = async function cameraReset() {
    await api("/api/camera_reset");
    await refreshCameraHealth();
  };

  window.cameraRotate = async function cameraRotate(enabled) {
    await api("/api/camera_rotate/" + (enabled ? "on" : "off"));
    await refreshCameraHealth();
  };

  window.mood = async function setMood(name) {
    await api("/api/mood/" + encodeURIComponent(name));
    await refreshHealth();
  };

  window.unloadLeg = async function unloadLeg(id) {
    addChat("owner", "Free leg " + id);
    await api("/api/unload_leg/" + id);
  };

  window.rerunPrecheck = async function rerunPrecheck() {
    addChat("owner", "Re-run precheck");
    await api("/api/precheck");
    await refreshHealth();
    await refreshMotionDiagnostics();
    await refreshEyesStatus();
  };

  window.refreshCamera = refreshCamera;
  window.refreshState = refreshState;

  // ==========================================================================
  // Preset action buttons
  // ==========================================================================

  const ACTION_LABELS = {
    1: "Lie Down",
    2: "Stand Up",
    3: "Creep Fwd",
    4: "Circle",
    5: "Squat",
    6: "Roll",
    7: "Stretch",
    8: "Snake",
    9: "Pray",
    10: "Look Food",
    11: "Handshake",
    12: "Nod",
    13: "Pushup",
    14: "Look Around",
    15: "Wave",
    16: "Dance",
    17: "Pee",
    18: "Sit Beg",
    19: "Sway Roll",
    20: "Sway Pitch",
    21: "Sway Yaw",
    22: "Chicken",
    23: "Twist",
    24: "Reset"
  };

  function initializeActionButtons() {
    const actionsElement = $("actions");
    if (!actionsElement) {
      return;
    }

    actionsElement.replaceChildren();

    for (let actionId = 1; actionId <= 24; actionId += 1) {
      const button = document.createElement("button");
      button.textContent =
        actionId + ". " + (ACTION_LABELS[actionId] || "Action " + actionId);
      button.addEventListener("click", () => window.action(actionId));
      actionsElement.appendChild(button);
    }
  }

  // ==========================================================================
  // Chat form
  // ==========================================================================

  function initializeChatForm() {
    const composer = $("composer");
    const promptBox = $("prompt");

    if (!composer || !promptBox) {
      return;
    }

    composer.addEventListener("submit", async (event) => {
      event.preventDefault();

      const prompt = promptBox.value.trim();
      if (!prompt) {
        return;
      }

      promptBox.value = "";
      addChat("owner", prompt);
      await api("/api/chat/" + encodeURIComponent(prompt));
      await refreshState();
    });
  }

  // ==========================================================================
  // Initialization
  // ==========================================================================

  function initializeGraph() {
    canvas = $("accel_canvas");
    context = canvas ? canvas.getContext("2d") : null;
    if (context) {
      requestAnimationFrame(drawGraph);
    }
  }

  async function initialRefresh() {
    await Promise.allSettled([
      refreshState(),
      refreshVibration(),
      refreshCamera(),
      refreshHealth(),
      refreshTelemetry(),
      refreshController(),
      refreshMotionDiagnostics(),
      refreshEyesStatus(),
      refreshCameraHealth(),
      refreshVisionSummary(),
      refreshReacquireStatus(),
      refreshOrchestrator()
    ]);
  }

  function startPolling() {
    setInterval(refreshVibration, 250);
    setInterval(refreshCamera, 300);
    setInterval(refreshMotionDiagnostics, 1000);
    setInterval(refreshVisionSummary, 1000);
    setInterval(refreshController, 1000);
    setInterval(refreshState, 2000);
    setInterval(refreshReacquireStatus, 2000);
    setInterval(refreshTelemetry, 3000);
    setInterval(refreshEyesStatus, 3000);
    setInterval(refreshCameraHealth, 3000);
    setInterval(refreshHealth, 5000);
    setInterval(refreshOrchestrator, 5000);
  }

  function init() {
    initializeGraph();
    initializeActionButtons();
    initializeChatForm();
    updateLedLegend();
    initialRefresh();
    startPolling();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();





    
  

















































      
      
      
/*     
// ============================================================================
// Robodog Pro v8.4.1 - WebUI frontend (robust)
// Live accel graph + anomaly flash/freeze + event banner + peak-hold +
// LED legend that mirrors the physical LED3 (mode + mood).
// ============================================================================
"use strict";

(function () {
  const $ = (id) => document.getElementById(id);
  const setText = (id, val) => { const el = $(id); if (el) el.textContent = val; };

  function api(path) {
    return fetch(path, { cache: "no-store" })
      .then((r) => r.json())
      .catch((e) => ({ ok: false, error: e.message }));
  }
  function addChat(role, text) {
    const chatEl = $("chat"); if (!chatEl) return;
    const div = document.createElement("div");
    div.className = "msg " + role;
    div.innerHTML = "<b>" + role + ":</b> " + text;
    chatEl.appendChild(div); chatEl.scrollTop = chatEl.scrollHeight;
  }

  // -------- live accelerometer graph (optional/fail-soft) --------
  let CANVAS = null, CTX = null;
  const MAXPTS = 160, RANGE = 2.0;
  const dataX = [], dataY = [], dataZ = [], dataV = [];
  let frozen = false, hitUntil = 0, lastVib = 0;

  // peak-hold for header vib so spikes linger and are noticeable
  let vibPeak = 0, vibPeakAt = 0;
  const VIB_PEAK_HOLD_MS = 1500, VIB_PEAK_DECAY = 0.6;

  // event banner
  let eventUntil = 0;
  const EVENT_COLORS = { manual: "evt-green", autonomous: "evt-blue", walkaround: "evt-blue",
    detected: "evt-cyan", searching: "evt-amber", wall: "evt-red", impact: "evt-red", lowbatt: "evt-purple" };
  function noteEvent(text, kind, holdMs) {
    const el = $("event_banner"), tx = $("event_text"); if (!el || !tx) return;
    tx.textContent = text; el.className = "event active " + (EVENT_COLORS[kind] || "");
    void el.offsetWidth; el.classList.add("active");
    eventUntil = Date.now() + (holdMs || 3000);
    setTimeout(() => { if (Date.now() >= eventUntil) { el.className = "event idle"; tx.textContent = "Idle"; } }, (holdMs || 3000) + 50);
  }
  window.noteEvent = noteEvent;

  // change trackers
  let prevMode = null, prevImpacts = 0, prevPhase = "", prevMood = "";
  // live state shared between pollers so the LED legend can combine mode + mood
  let liveMode = "manual", liveMood = "normal", liveActive = false;

  function graphEnabled() { return !!CTX; }
  function pushSample(x, y, z, v) {
    if (!graphEnabled() || frozen || Date.now() < hitUntil) return;
    dataX.push(x); dataY.push(y); dataZ.push(z); dataV.push(v);
    if (dataX.length > MAXPTS) { dataX.shift(); dataY.shift(); dataZ.shift(); dataV.shift(); }
  }
  function drawSeries(arr, color, scale) {
    const w = CANVAS.width, h = CANVAS.height, mid = h / 2, n = arr.length; if (!n) return;
    CTX.strokeStyle = color; CTX.lineWidth = 1.6; CTX.beginPath();
    for (let i = 0; i < n; i++) { const px = (i / (MAXPTS - 1)) * w, py = mid - (arr[i] / scale) * (h / 2 - 8); i ? CTX.lineTo(px, py) : CTX.moveTo(px, py); }
    CTX.stroke();
  }
  function drawGraph() {
    if (!graphEnabled()) return;
    const w = CANVAS.width, h = CANVAS.height, mid = h / 2;
    CTX.clearRect(0, 0, w, h);
    CTX.strokeStyle = "#1e293b"; CTX.lineWidth = 1; CTX.fillStyle = "#334155"; CTX.font = "10px monospace";
    for (let g = -RANGE; g <= RANGE; g += 1) { const py = mid - (g / RANGE) * (h / 2 - 8); CTX.beginPath(); CTX.moveTo(0, py); CTX.lineTo(w, py); CTX.stroke(); CTX.fillText(g.toFixed(0), 2, py - 2); }
    drawSeries(dataX, "#38bdf8", RANGE); drawSeries(dataY, "#22c55e", RANGE);
    drawSeries(dataZ, "#f97316", RANGE); drawSeries(dataV, "#a78bfa", RANGE);
    requestAnimationFrame(drawGraph);
  }
  function flashHit(reason) {
    hitUntil = Date.now() + 2000;
    const fl = $("graph_flash"), gs = $("graph_state");
    if (fl) fl.classList.add("hit");
    if (gs) { gs.classList.add("hit"); gs.textContent = reason || "IMPACT"; }
    setTimeout(() => { if (fl) fl.classList.remove("hit"); if (gs && !frozen) { gs.classList.remove("hit"); gs.textContent = "idle"; } }, 2000);
  }
  window.toggleFreeze = function () {
    frozen = !frozen; const gs = $("graph_state"), b = $("freeze_btn");
    if (gs) { gs.classList.toggle("frozen", frozen); gs.textContent = frozen ? "frozen" : "idle"; }
    if (b) b.textContent = frozen ? "▶ Resume" : "⏸ Freeze";
  };
  window.clearGraph = function () { dataX.length = dataY.length = dataZ.length = dataV.length = 0; };

  // -------------------------------------------------------------------------
  // Effective LED state = the same priority the sketch uses to drive LED3.
  //   alert/detected/searching/low_energy/sleeping override the base mode.
  //   Otherwise: autonomous -> happy(blue), manual -> normal(green).
  // This makes the highlighted legend row match the physical LED3 even
  // between mood pings (e.g. blue the whole time while wandering).
  // -------------------------------------------------------------------------
  function effectiveLedMood() {
    const m = liveMood;
    if (m === "alert" || m === "detected" || m === "searching" || m === "low_energy" || m === "sleeping") return m;
    if (liveMode === "autonomous") return "happy";     // blue while wandering
    return "normal";                                    // green in manual/ready
  }
  function updateLedLegend() {
    const eff = effectiveLedMood();
    const rows = document.querySelectorAll(".led-legend tr[data-mood]");
    rows.forEach((tr) => tr.classList.toggle("active", tr.getAttribute("data-mood") === eff));
  }

  // -------- pollers --------
  async function refreshState() {
    try {
      const s = await api("/api/state");
      if (s && s.mode) setText("status", "Mode: " + s.mode + " | Last: " + s.last_command);
      setText("wifi_ssid", (s && s.wifi_ssid) || "DOGZILLA-UNOQ");
      setText("wifi_password", (s && s.wifi_password) || "dogzilla123");
      setText("wifi_url", (s && s.wifi_url) || "http://10.42.0.1:7000");
      setText("wifi_ap", (s && s.wifi_ap) || "unknown");

      const chatEl = $("chat");
      if (chatEl && s && s.chat) { chatEl.innerHTML = ""; for (const m of s.chat) addChat(m.role, m.text); }

      if (s && s.sequence) {
        const seq = s.sequence;
        setText("sequence_status", seq.running ? "Running: " + seq.step
          : (seq.step === "cancelled" ? "Cancelled." : seq.step === "complete" ? "Complete." : "Sequence idle."));
      }
      if (s && s.mode) {
        if (s.mode !== liveMode) {
          if (prevMode !== null) noteEvent(s.mode === "autonomous" ? "AUTONOMOUS" : "MANUAL", s.mode === "autonomous" ? "autonomous" : "manual");
          prevMode = s.mode;
        }
        liveMode = s.mode;
        updateLedLegend();   // mode change can flip green<->blue immediately
      }
      if (s && s.autonomy) {
        const a = s.autonomy;
        liveActive = !!a.active;
        setText("autonomy_status", a.active ? "Active: " + a.phase + " (" + (a.reason || "") + ") vib=" + a.vibration : "Autonomy idle.");
        setText("anom_impacts", a.impacts != null ? a.impacts : 0);
        setText("anom_z", a.z != null ? a.z : 0);
        setText("anom_mean", a.vib_mean != null ? a.vib_mean : 0);
        setText("anom_src", a.anom_src || "-");
        setText("lz_score", a.z != null ? a.z : 0);
        setText("graph_src", a.anom_src === "edge_impulse" ? "EI model" : "z-score");
        const ph = String(a.phase || "");
        if (ph.startsWith("recover")) flashHit("OBSTACLE");
        const imp = a.impacts != null ? a.impacts : 0;
        if (imp > prevImpacts) { noteEvent("WALL HIT #" + imp, "wall", 3500); prevImpacts = imp; }
        if (ph !== prevPhase) {
          if (ph.startsWith("recover_wall")) noteEvent("OBSTACLE — turning", "wall", 3500);
          else if (ph.startsWith("recover_tilt")) noteEvent("TILT recover", "wall", 3000);
          else if (ph === "sniff") noteEvent("Sniffing around", "searching", 2000);
          prevPhase = ph;
        }
      }
    } catch (e) { setText("status", "Backend not reachable: " + e.message); }
  }

  async function refreshHealth() {
    try {
      const h = await api("/api/health");
      if (!h || !h.ok) return;
      setText("diag_ip", h.advertised_ip || "-");
      setText("diag_firmware", (h.hardware && h.hardware.dog_firmware) || "?");
      const mc = (h.mood && h.mood.current) || "?";
      setText("diag_mood", mc);
      liveMood = mc;
      updateLedLegend();   // combine with liveMode -> highlight the LED3 row
      if (mc !== prevMood) {
        if (prevMood && mc === "detected") noteEvent("PERSON detected", "detected", 3000);
        else if (prevMood && mc === "searching") noteEvent("Searching", "searching", 2000);
        else if (prevMood && mc === "low_energy") noteEvent("LOW BATTERY", "lowbatt", 4000);
        else if (prevMood && mc === "alert") noteEvent("ALERT", "wall", 3000);
        prevMood = mc;
      }
      setText("diag_bridge", h.bridge && h.bridge.ok ? "OK" : "ERR (" + (h.bridge && h.bridge.consecutive_errors) + " errs)");
      setText("diag_camera", h.camera && h.camera.ok ? "OK (" + h.camera.fps + " fps)" : "ERR (" + ((h.camera && h.camera.error) || "") + ")");
      setText("diag_uptime", h.uptime_s || 0);
      setText("diag_captures", h.captures_persisted ? "YES (leak!)" : "NO ✓");
    } catch (e) {}
  }

  async function refreshController() {
    try {
      const r = await api("/api/controller/status");
      const c = (r && r.controller) || {};
      setText("controller_status", c.connected
        ? "Connected: " + (c.name || c.device) + " (" + c.api + ") | events: " + c.events + " | last: " + c.last_command
        : "Not connected: " + (c.error || "scanning"));
    } catch (e) {}
  }

  async function refreshCamera() {
    try {
      const c = await api("/api/camera");
      const img = $("camera_view"), st = $("camera_status");
      if (!c || !c.ok || !c.frame_base64) { if (st) st.textContent = "Camera error: " + ((c && c.error) || "no frame yet"); if (img) img.removeAttribute("src"); return; }
      if (img) img.src = "data:image/jpeg;base64," + c.frame_base64;
      if (st) st.textContent = "Camera running | FPS: " + (c.fps || 0);
    } catch (e) { setText("camera_status", "Camera backend not reachable: " + e.message); }
  }

  async function refreshVibration() {
    try {
      const v = await api("/api/vibration");
      if (!v || !v.ok) return;
      const vib = Number(v.vibration || 0), x = Number(v.x || 0), y = Number(v.y || 0), z = Number(v.z || 0);
      setText("vibration_value", vib.toFixed(3));
      setText("accel_x", x.toFixed(3)); setText("accel_y", y.toFixed(3)); setText("accel_z", z.toFixed(3));
      setText("lx", x.toFixed(3)); setText("ly", y.toFixed(3)); setText("lz", z.toFixed(3));
      const lv = $("lv"); if (lv) { lv.textContent = vib.toFixed(3); lv.className = "v" + (vib >= 1.2 ? " bad" : vib >= 0.4 ? " warn" : ""); }

      const now = Date.now();
      if (vib >= vibPeak) { vibPeak = vib; vibPeakAt = now; }
      else if (now - vibPeakAt > VIB_PEAK_HOLD_MS) { vibPeak = Math.max(vib, vibPeak * VIB_PEAK_DECAY); if (vibPeak < 0.01) vibPeak = 0; }
      setText("lpk", vibPeak.toFixed(3));
      const lpk = $("lpk"); if (lpk) lpk.className = "v" + (vibPeak >= 1.2 ? " bad" : vibPeak >= 0.4 ? " warn" : "");
      const vv = $("vibration_value"); if (vv) vv.classList.toggle("peak", vibPeak >= 1.2);

      if (vib >= 1.2 && lastVib < 1.2) { flashHit("BLIP"); noteEvent("IMPACT " + vib.toFixed(2) + "g", "impact", 2500); }
      lastVib = vib; pushSample(x, y, z, vib);

      const s = $("vibration_status");
      if (s) {
        if (vib < 0.05) { s.textContent = "Stable / idle"; s.className = "vibration-status stable"; }
        else if (vib < 0.40) { s.textContent = "Normal movement"; s.className = "vibration-status normal"; }
        else if (vib < 1.20) { s.textContent = "High vibration"; s.className = "vibration-status warning"; }
        else { s.textContent = "Strong impact / fall"; s.className = "vibration-status danger"; }
      }
    } catch (e) {}
  }

  async function refreshTelemetry() {
    try {
      const t = await api("/api/telemetry");
      if (!t || !t.ok) return;
      const fmt = (val) => (val === null || val === undefined || isNaN(val)) ? "-" : Number(val).toFixed(2);
      setText("imu_roll", fmt(t.roll)); setText("imu_pitch", fmt(t.pitch)); setText("imu_yaw", fmt(t.yaw));
      setText("imu_battery", (t.battery === null || t.battery === undefined) ? "-" : t.battery);
    } catch (e) {}
  }

  // -------- actions (exposed for inline onclick) --------
  window.cmd = async function (name) { addChat("owner", "Command: " + name); await api("/api/cmd/" + encodeURIComponent(name)); refreshState(); refreshVibration(); };
  window.action = async function (id) { addChat("owner", "Action " + id); await api("/api/action/" + id); refreshState(); refreshVibration(); };
  window.mode = async function (name) { await api("/api/mode/" + encodeURIComponent(name)); refreshState(); };
  window.controllerEnable = async function (en) { await api("/api/gamepad/" + (en ? "true" : "false")); refreshController(); };
  window.controllerRescan = async function () { await api("/api/controller/rescan"); refreshController(); };
  window.walkaroundStart = async function () { addChat("owner", "Start walkaround"); await api("/api/sequence/walkaround"); refreshState(); };
  window.walkaroundStop = async function () { addChat("owner", "Cancel walkaround"); await api("/api/sequence/stop"); refreshState(); };
  window.sfx = async function (event) { await api("/api/audio/test/" + encodeURIComponent(event)); };
  window.audioMute = async function (on) { await api("/api/audio/mute/" + (on ? "true" : "false")); };
  window.copyApCommand = async function (action) {
    const c = "$HOME/ArduinoApps/robodog-pro-v8-3-2/host/dogzilla-ap.sh " + action;
    try { await navigator.clipboard.writeText(c); addChat("robot", "Copied host command: " + c); } catch (e) { window.prompt("Copy this UNO Q host command:", c); }
  };
  window.cameraReset = async function () { await api("/api/camera_reset"); };
  window.cameraRotate = async function (on) { await api("/api/camera_rotate/" + (on ? "on" : "off")); };
  window.mood = async function (name) { await api("/api/mood/" + encodeURIComponent(name)); refreshHealth(); };
  window.unloadLeg = async function (id) { addChat("owner", "Free leg " + id); await api("/api/unload_leg/" + id); };
  window.rerunPrecheck = async function () { addChat("owner", "Re-run precheck"); await api("/api/precheck"); refreshHealth(); };
  window.refreshCamera = refreshCamera;
  window.refreshState = refreshState;

  const ACTION_LABELS = { 1:"Lie Down",2:"Stand Up",3:"Creep Fwd",4:"Circle",5:"Squat",6:"Roll",7:"Stretch",8:"Snake",9:"Pray",10:"Look Food",11:"Handshake",12:"Nod",13:"Pushup",14:"Look Around",15:"Wave",16:"Dance",17:"Pee",18:"Sit Beg",19:"Sway Roll",20:"Sway Pitch",21:"Sway Yaw",22:"Chicken",23:"Twist",24:"Reset" };

  function init() {
    CANVAS = $("accel_canvas"); CTX = CANVAS ? CANVAS.getContext("2d") : null;
    if (CTX) requestAnimationFrame(drawGraph);
    const actionsEl = $("actions");
    if (actionsEl) for (let i = 1; i <= 24; i++) { const b = document.createElement("button"); b.textContent = i + ". " + (ACTION_LABELS[i] || "Action " + i); b.onclick = () => window.action(i); actionsEl.appendChild(b); }
    const composer = $("composer"), promptBox = $("prompt");
    if (composer && promptBox) composer.addEventListener("submit", async (e) => { e.preventDefault(); const p = promptBox.value.trim(); if (!p) return; promptBox.value = ""; addChat("owner", p); await api("/api/chat/" + encodeURIComponent(p)); refreshState(); });
    updateLedLegend();
    refreshState(); refreshVibration(); refreshCamera(); refreshHealth(); refreshTelemetry(); refreshController();
    setInterval(refreshState, 2000);
    setInterval(refreshVibration, 250);
    setInterval(refreshCamera, 300);
    setInterval(refreshHealth, 5000);
    setInterval(refreshTelemetry, 3000);
    setInterval(refreshController, 1000);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
*/





























