const ui = new WebUI();

const el = (id) => document.getElementById(id);
const armButton = el('arm-button');
const lockButton = el('lock-button');
const safetyCheck = el('safety-check');
const actionMessage = el('action-message');
const servoId = el('servo-id');
const driveButtons = [...document.querySelectorAll('.drive')];
const motionControls = [...document.querySelectorAll('[data-motion-control]')];
const modeButtons = [...document.querySelectorAll('.mode-button')];
const lidarCanvas = el('lidar-map');
const mapRangeSelect = el('map-range');
const gaitSelect = el('gait-select');
let rospugConnected = false;
let rospugArmed = false;
let rospugMode = 'guard';
let gamepadOnline = false;
let lidarFresh = false;
let activeDirection = null;
let motionTimer = null;
let latestLidar = {};
let latestCommanded = [0, 0, 0];
let gaitSelectionDirty = false;
let pendingGait = null;

const dashboardUrl = window.location.origin || 'http://10.42.0.1:7000';
el('dashboard-url').textContent = dashboardUrl;
el('copy-dashboard-url').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(dashboardUrl);
    setAction(`Dashboard address copied: ${dashboardUrl}`, true);
  } catch (_error) {
    setAction(`Open this address on the ROSPug-Pro network: ${dashboardUrl}`, true);
  }
});

for (let id = 1; id <= 12; id += 1) {
  const option = document.createElement('option');
  option.value = id;
  option.textContent = id;
  servoId.appendChild(option);
}

ui.on_connect(() => {
  setAction('Connected to App Lab. Waiting for ROSPug telemetry...', true);
  ui.send_message('get_state');
});

ui.on_disconnect(() => {
  clearMotionTimer();
  updateConnection(false);
  el('bridge-error').textContent = 'Web UI connection lost.';
});

ui.on_message('rospug_state', updateState);
ui.on_message('action_result', (result) => {
  if (result.action === 'set_gait') {
    pendingGait = null;
    gaitSelectionDirty = !result.ok;
  }
  setAction(result.message, result.ok);
});
ui.on_message('servo_result', updateServoResult);
ui.on_message('motion_result', (result) => setAction(result.message, result.ok));

safetyCheck.addEventListener('change', () => {
  updateArmAvailability();
});

armButton.addEventListener('click', () => {
  if (!safetyCheck.checked) return;
  armButton.disabled = true;
  setAction('Sending verified initialization frame...', true);
  ui.send_message('set_armed', { enabled: true, confirmed: true });
});

lockButton.addEventListener('click', () => {
  ui.send_message('set_armed', { enabled: false });
});

el('read-servo').addEventListener('click', () => {
  el('servo-result').textContent = `Reading servo ${servoId.value}...`;
  ui.send_message('read_servo', { id: Number(servoId.value) });
});

gaitSelect.addEventListener('change', () => {
  gaitSelectionDirty = true;
});

el('apply-gait').addEventListener('click', () => {
  pendingGait = Number(gaitSelect.value);
  gaitSelectionDirty = false;
  setAction(`Loading ${gaitSelect.options[gaitSelect.selectedIndex].text.split(' · ')[0]} gait...`, true);
  ui.send_message('set_gait', { preset: pendingGait });
});

el('home-button').addEventListener('click', () => {
  ui.send_message('home');
});

el('stop-button').addEventListener('click', () => {
  stopMotion(true);
});

modeButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const mode = button.dataset.mode;
    let confirmed = false;
    if (mode === 'manual') {
      confirmed = window.confirm('Manual mode bypasses LiDAR obstacle protection. Continue?');
      if (!confirmed) return;
    }
    if (mode === 'auto') {
      confirmed = window.confirm(
        'Auto Avoid starts the robot walking immediately. Support the robot for the first test and keep the remote ready. Start Auto Avoid?'
      );
      if (!confirmed) return;
    }
    ui.send_message('set_mode', { mode, confirmed });
  });
});

driveButtons.forEach((button) => {
  button.disabled = true;
  button.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    if (button.disabled) return;
    activeDirection = button.dataset.direction;
    button.setPointerCapture(event.pointerId);
    sendMotionRefresh();
    clearInterval(motionTimer);
    motionTimer = setInterval(sendMotionRefresh, 250);
  });
  button.addEventListener('pointerup', () => { if (activeDirection) stopMotion(true); });
  button.addEventListener('pointercancel', () => { if (activeDirection) stopMotion(true); });
  button.addEventListener('lostpointercapture', () => { if (activeDirection) stopMotion(true); });
});

window.addEventListener('blur', () => stopMotion(Boolean(activeDirection)));
document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopMotion(Boolean(activeDirection));
});
mapRangeSelect.addEventListener('change', drawLidarMap);
window.addEventListener('resize', drawLidarMap);

function updateConnection(connected) {
  const pill = el('connection-pill');
  pill.className = `pill ${connected ? 'online' : 'offline'}`;
  pill.innerHTML = `<span></span>${connected ? 'ROSPug online' : 'No ROSPug frames'}`;
}

function updateState(state) {
  rospugConnected = Boolean(state.connected);
  rospugArmed = Boolean(state.armed);
  rospugMode = state.mode || 'guard';
  gamepadOnline = Boolean(state.gamepad && state.gamepad.online);
  lidarFresh = Boolean(state.lidar && state.lidar.fresh);
  updateConnection(rospugConnected);
  el('frames').textContent = Number(state.frames || 0).toLocaleString();
  el('crc-errors').textContent = Number(state.crc_errors || 0).toLocaleString();
  el('frame-age').textContent = state.frames ? `${state.last_frame_age_ms} ms` : '—';
  el('battery').textContent = state.battery_mv >= 0 ? `${state.battery_mv} mV` : 'not reported';
  el('imu-count').textContent = `${Number(state.imu_frames || 0).toLocaleString()} frames`;
  el('bridge-error').textContent = state.bridge_error || '';

  const imu = Array.isArray(state.imu) ? state.imu : [];
  for (let i = 0; i < 6; i += 1) {
    el(`imu-${i}`).textContent = Number(imu[i] || 0).toFixed(5);
  }

  const badge = el('armed-badge');
  badge.textContent = rospugArmed ? 'Initialized' : 'Locked';
  badge.className = `badge ${rospugArmed ? 'armed' : 'locked'}`;

  const motionBadge = el('motion-badge');
  motionBadge.textContent = state.moving ? 'Moving' : 'Stopped';
  motionBadge.className = `badge ${state.moving ? 'armed' : 'locked'}`;
  if (state.gait >= 1 && state.gait <= 5) {
    if (pendingGait === state.gait) {
      pendingGait = null;
      gaitSelectionDirty = false;
    }
    if (!gaitSelectionDirty && pendingGait === null) {
      gaitSelect.value = String(state.gait);
    }
  }

  updateNavigation(state);

  updateArmAvailability();
  updateMotionAvailability();
}

function updateNavigation(state) {
  const gamepad = state.gamepad || {};
  const lidar = state.lidar || {};
  const sectors = lidar.sectors || {};
  latestLidar = lidar;
  latestCommanded = Array.isArray(state.commanded) ? state.commanded : [0, 0, 0];
  el('gamepad-metric').textContent = gamepadOnline ? 'online' : 'offline';
  el('lidar-metric').textContent = lidarFresh ? `${Number(lidar.speed_dps || 0)}°/s` : 'offline';

  const gamepadBadge = el('gamepad-badge');
  gamepadBadge.textContent = gamepadOnline ? 'Gamepad online' : 'Gamepad offline';
  gamepadBadge.className = `badge ${gamepadOnline ? 'armed' : 'locked'}`;
  el('gamepad-path').textContent = gamepadOnline
    ? (gamepad.path || 'connected')
    : (gamepad.error || gamepad.path || 'not detected');

  const lidarBadge = el('lidar-badge');
  lidarBadge.textContent = lidarFresh ? 'LD19 live' : 'LD19 offline';
  lidarBadge.className = `badge ${lidarFresh ? 'armed' : 'locked'}`;
  el('lidar-path').textContent = lidarFresh
    ? (lidar.path || 'connected')
    : (lidar.error || lidar.path || 'not detected');

  el('sector-front').textContent = formatDistance(sectors.front);
  el('sector-left').textContent = formatDistance(sectors.left);
  el('sector-right').textContent = formatDistance(sectors.right);
  el('sector-rear').textContent = formatDistance(sectors.rear);
  el('lidar-packets').textContent = Number(lidar.valid_packets || 0).toLocaleString();
  el('lidar-crc').textContent = Number(lidar.crc_errors || 0).toLocaleString();
  el('input-source').textContent = state.input_source || 'none';
  el('auto-state').textContent = state.auto_state || 'idle';
  el('blocked-reason').textContent = state.blocked_reason || 'Path clear.';

  const labels = { manual: 'Manual', guard: 'LiDAR Guard', auto: 'Auto Avoid' };
  const modeBadge = el('mode-badge');
  modeBadge.textContent = labels[rospugMode] || rospugMode;
  modeBadge.className = `badge ${rospugMode === 'auto' ? 'armed' : 'locked'}`;
  modeButtons.forEach((button) => {
    button.classList.toggle('active', button.dataset.mode === rospugMode);
    button.disabled = button.dataset.mode === 'auto'
      && (!rospugConnected || !rospugArmed || !gamepadOnline || !lidarFresh);
  });
  drawLidarMap();
}

function drawLidarMap() {
  if (!lidarCanvas) return;
  const context = lidarCanvas.getContext('2d');
  if (!context) return;
  const bounds = lidarCanvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const width = bounds.width;
  const height = bounds.height;
  const targetWidth = Math.round(width * pixelRatio);
  const targetHeight = Math.round(height * pixelRatio);
  if (lidarCanvas.width !== targetWidth || lidarCanvas.height !== targetHeight) {
    lidarCanvas.width = targetWidth;
    lidarCanvas.height = targetHeight;
  }
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);

  const range = Number(mapRangeSelect.value || 3);
  const centerX = width / 2;
  const centerY = height / 2 + 8;
  const radius = Math.min(width, height) * 0.43;
  const scale = radius / range;
  const ringStep = range <= 1.5 ? 0.25 : range <= 3 ? 0.5 : range <= 6 ? 1 : 2;

  const background = context.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
  background.addColorStop(0, 'rgba(28, 70, 59, 0.20)');
  background.addColorStop(1, 'rgba(3, 12, 10, 0.82)');
  context.fillStyle = background;
  context.fillRect(0, 0, width, height);

  drawSafetyWedge(context, centerX, centerY, 0.75 * scale, 'rgba(255, 202, 106, 0.11)', 0);
  drawSafetyWedge(context, centerX, centerY, 0.32 * scale, 'rgba(255, 114, 114, 0.16)', 0);
  drawSafetyWedge(context, centerX, centerY, 0.65 * scale, 'rgba(255, 202, 106, 0.08)', 180);
  drawSafetyWedge(context, centerX, centerY, 0.30 * scale, 'rgba(255, 114, 114, 0.12)', 180);

  context.lineWidth = 1;
  context.strokeStyle = 'rgba(143, 168, 160, 0.23)';
  context.fillStyle = 'rgba(143, 168, 160, 0.70)';
  context.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  context.textAlign = 'left';
  context.textBaseline = 'bottom';
  for (let meters = ringStep; meters <= range + 0.0001; meters += ringStep) {
    context.beginPath();
    context.arc(centerX, centerY, meters * scale, 0, Math.PI * 2);
    context.stroke();
    context.fillText(`${meters.toFixed(meters < 1 ? 2 : 1)} m`, centerX + 5, centerY - meters * scale + 14);
  }
  context.beginPath();
  context.moveTo(centerX - radius, centerY);
  context.lineTo(centerX + radius, centerY);
  context.moveTo(centerX, centerY - radius);
  context.lineTo(centerX, centerY + radius);
  context.stroke();

  const scan = Array.isArray(latestLidar.scan) ? latestLidar.scan : [];
  for (const point of scan) {
    if (!Array.isArray(point) || point.length < 3) continue;
    const angle = Number(point[0]);
    const distance = Number(point[1]);
    const intensity = Number(point[2]);
    if (!Number.isFinite(angle) || !Number.isFinite(distance) || distance <= 0 || distance > range) continue;
    const radians = angle * Math.PI / 180;
    const x = centerX - Math.sin(radians) * distance * scale;
    const y = centerY - Math.cos(radians) * distance * scale;
    const alpha = Math.max(0.45, Math.min(1, intensity / 180));
    context.fillStyle = `rgba(102, 242, 188, ${alpha})`;
    context.beginPath();
    context.arc(x, y, pixelRatio > 1 ? 2.1 : 2.5, 0, Math.PI * 2);
    context.fill();
  }

  drawRobot(context, centerX, centerY);
  drawCommand(context, centerX, centerY, latestCommanded);

  el('map-point-count').textContent = `${scan.length} points · ${latestLidar.last_packet_age_ms ?? '—'} ms`;
  const overlay = el('map-overlay');
  const live = Boolean(latestLidar.fresh && scan.length);
  overlay.classList.toggle('hidden', live);
  overlay.textContent = latestLidar.error || (latestLidar.fresh ? 'Waiting for scan points' : 'LD19 scan offline');
}

function drawSafetyWedge(context, centerX, centerY, radius, fill, headingDegrees) {
  const halfAngle = 25 * Math.PI / 180;
  const heading = (headingDegrees - 90) * Math.PI / 180;
  context.beginPath();
  context.moveTo(centerX, centerY);
  context.arc(centerX, centerY, radius, heading - halfAngle, heading + halfAngle);
  context.closePath();
  context.fillStyle = fill;
  context.fill();
}

function drawRobot(context, centerX, centerY) {
  context.save();
  context.translate(centerX, centerY);
  context.fillStyle = '#d9fff1';
  context.strokeStyle = '#66f2bc';
  context.lineWidth = 2;
  context.beginPath();
  context.rect(-13, -18, 26, 36);
  context.fill();
  context.stroke();
  context.fillStyle = '#07100f';
  context.beginPath();
  context.moveTo(0, -13);
  context.lineTo(-6, -4);
  context.lineTo(6, -4);
  context.closePath();
  context.fill();
  context.restore();
}

function drawCommand(context, centerX, centerY, commanded) {
  const forward = Number(commanded[0] || 0);
  const left = Number(commanded[1] || 0);
  const yaw = Number(commanded[2] || 0);
  const magnitude = Math.hypot(forward, left);
  context.save();
  context.strokeStyle = '#ffca6a';
  context.fillStyle = '#ffca6a';
  context.lineWidth = 3;
  context.lineCap = 'round';
  if (magnitude > 0.001) {
    const length = Math.min(70, 28 + magnitude * 500);
    const dx = (-left / magnitude) * length;
    const dy = (-forward / magnitude) * length;
    drawArrow(context, centerX, centerY, centerX + dx, centerY + dy);
  }
  if (Math.abs(yaw) > 0.01) {
    const anticlockwise = yaw > 0;
    context.beginPath();
    context.arc(centerX, centerY, 29, anticlockwise ? 0.2 : Math.PI - 0.2,
      anticlockwise ? Math.PI * 1.55 : -Math.PI * 0.55, anticlockwise);
    context.stroke();
  }
  context.restore();
}

function drawArrow(context, x1, y1, x2, y2) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  context.beginPath();
  context.moveTo(x1, y1);
  context.lineTo(x2, y2);
  context.stroke();
  context.beginPath();
  context.moveTo(x2, y2);
  context.lineTo(x2 - 10 * Math.cos(angle - Math.PI / 6), y2 - 10 * Math.sin(angle - Math.PI / 6));
  context.lineTo(x2 - 10 * Math.cos(angle + Math.PI / 6), y2 - 10 * Math.sin(angle + Math.PI / 6));
  context.closePath();
  context.fill();
}

function formatDistance(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
  return `${Number(value).toFixed(2)} m`;
}

function updateArmAvailability() {
  armButton.disabled = !safetyCheck.checked || !rospugConnected || rospugArmed;
}

function updateMotionAvailability() {
  const enabled = rospugConnected && rospugArmed && rospugMode !== 'auto';
  driveButtons.forEach((button) => { button.disabled = !enabled; });
  motionControls.forEach((control) => { control.disabled = !enabled; });
  el('stop-button').disabled = !rospugConnected;
  if (!enabled && activeDirection) stopMotion(rospugConnected);
}

function sendMotionRefresh() {
  if (!activeDirection || !rospugConnected || !rospugArmed) return;
  ui.send_message('motion', { direction: activeDirection });
}

function clearMotionTimer() {
  if (motionTimer) clearInterval(motionTimer);
  motionTimer = null;
}

function stopMotion(sendStop) {
  const wasActive = Boolean(activeDirection);
  activeDirection = null;
  clearMotionTimer();
  if (sendStop || wasActive) ui.send_message('stop');
}

function updateServoResult(result) {
  const output = el('servo-result');
  if (result.ok) {
    output.textContent = `Servo ${result.id}: position ${result.position} (status ${result.status})`;
  } else {
    output.textContent = result.error || `Servo ${result.id}: controller status ${result.status}`;
  }
}

function setAction(message, ok) {
  actionMessage.textContent = message || '';
  actionMessage.className = `message ${ok ? '' : 'error'}`;
}
