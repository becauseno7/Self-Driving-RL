const $ = (selector) => document.querySelector(selector);

const ui = {
  canvas: $("#road"),
  shell: $("#canvas-shell"),
  load: $("#load-state"),
  runState: $("#run-state"),
  canvasSeed: $("#canvas-seed"),
  outcome: $("#outcome-toast"),
  play: $("#play"),
  playLabel: $("#play-label"),
  newSeed: $("#new-seed"),
  restart: $("#restart"),
  difficulty: $("#difficulty"),
  rate: $("#rate"),
  dynamic: $("#dynamic"),
  sensors: $("#sensors"),
  elapsed: $("#elapsed"),
  intent: $("#intent"),
  intentMark: $("#intent-mark"),
  intentReason: $("#intent-reason"),
  speedValue: $("#speed-value"),
  speedFill: $("#speed-fill"),
  targetPin: $("#target-pin"),
  targetSpeed: $("#target-speed"),
  ttc: $("#ttc"),
  rearTtc: $("#rear-ttc"),
  passes: $("#passes"),
  distance: $("#distance"),
  laneChanges: $("#lane-changes"),
  latency: $("#latency"),
  challenge: $("#challenge"),
  action: $("#action"),
  layers: $("#layers"),
  braking: $("#braking"),
  seedValue: $("#seed-value"),
  episodeValue: $("#episode-value"),
  trafficEvents: $("#traffic-events"),
  footerState: $("#footer-state"),
};

const ctx = ui.canvas.getContext("2d");
const controls = [
  ui.play,
  ui.newSeed,
  ui.restart,
  ui.difficulty,
  ui.rate,
  ui.dynamic,
  ui.sensors,
];
const state = {
  socket: null,
  reconnectTimer: null,
  reconnectAttempt: 0,
  meta: null,
  current: null,
  previous: null,
  receivedAt: 0,
  expectedInterval: 100,
  settings: {
    seed: null,
    difficulty: "hard",
    dynamic_traffic: true,
    rate: 1,
    paused: false,
  },
  episode: 0,
  sensors: false,
  connected: false,
  outcomeTimer: null,
};

const carColors = ["#9b5d55", "#557486", "#b19a61", "#7b746d", "#607b68", "#8c6c85"];
const intentLabels = {
  CRUISE: "Open-road cruise",
  FOLLOW: "Following traffic",
  SEARCH_PASS: "Seeking a pass",
  PASS_LEFT: "Passing left",
  PASS_RIGHT: "Passing right",
  RETURN: "Settling in lane",
  EMERGENCY: "Emergency response",
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function lerp(a, b, amount) {
  return a + (b - a) * amount;
}

function lerpWithJump(a, b, amount, jump = 80) {
  return Math.abs(b - a) > jump ? (amount < 0.5 ? a : b) : lerp(a, b, amount);
}

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60).toString().padStart(2, "0");
  const remainder = (safe % 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${remainder}`;
}

function normalizeIntent(value) {
  return String(value || "CRUISE").replace(/^DrivingIntent\./, "").toUpperCase();
}

function formatTtc(value) {
  return value >= 90 ? "Clear" : `${Number(value).toFixed(1)} s`;
}

function setControlsEnabled(enabled) {
  controls.forEach((control) => {
    control.disabled = !enabled;
  });
}

function setConnectionState(kind, copy) {
  ui.runState.className = kind;
  ui.runState.innerHTML = `<i></i> ${copy}`;
  ui.footerState.textContent = `Live inference service · ${copy.toLowerCase()}`;
}

function showLoading(title, copy, spinning = true) {
  ui.load.hidden = false;
  ui.load.querySelector(".loader").style.display = spinning ? "block" : "none";
  ui.load.querySelector("strong").textContent = title;
  ui.load.querySelector("small").textContent = copy;
}

function hideLoading() {
  ui.load.hidden = true;
}

function showOutcome(message) {
  window.clearTimeout(state.outcomeTimer);
  ui.outcome.textContent = message;
  ui.outcome.hidden = false;
  state.outcomeTimer = window.setTimeout(() => {
    ui.outcome.hidden = true;
  }, 5200);
}

function updatePauseUi() {
  const paused = state.settings.paused;
  ui.play.classList.toggle("is-paused", paused);
  ui.playLabel.textContent = paused ? "Resume" : "Pause";
  ui.play.setAttribute("aria-label", paused ? "Resume live simulation" : "Pause live simulation");
  if (state.connected) setConnectionState(paused ? "paused" : "live", paused ? "Simulation paused" : "Policy deciding live");
}

function syncSettings(settings) {
  if (!settings) return;
  state.settings = { ...state.settings, ...settings };
  ui.difficulty.value = state.settings.difficulty;
  ui.rate.value = String(state.settings.rate);
  ui.dynamic.checked = Boolean(state.settings.dynamic_traffic);
  ui.canvasSeed.textContent = `Traffic seed ${Number(state.settings.seed).toLocaleString()}`;
  ui.seedValue.textContent = Number(state.settings.seed).toLocaleString();
  state.expectedInterval = 1000 / ((state.meta?.decision_hz || 10) * state.settings.rate);
  updatePauseUi();
}

function sendControl(command, payload = {}) {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;
  state.socket.send(JSON.stringify({ type: "control", command, ...payload }));
}

function roundedRect(context, x, y, width, height, radius) {
  const safeRadius = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.roundRect(x, y, width, height, safeRadius);
}

function drawVehicle(x, y, width, height, color, braking, ego = false, style = 0) {
  ctx.save();
  ctx.translate(x, y);
  ctx.shadowColor = "rgba(0, 0, 0, 0.34)";
  ctx.shadowBlur = height * 0.13;
  ctx.shadowOffsetY = height * 0.07;
  roundedRect(ctx, -width / 2, -height / 2, width, height, width * 0.24);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.shadowColor = "transparent";

  const inset = width * 0.16;
  const glassTop = -height * (style % 2 ? 0.19 : 0.22);
  roundedRect(ctx, -width / 2 + inset, glassTop, width - inset * 2, height * 0.24, 3);
  ctx.fillStyle = ego ? "#c9d9d1" : "#c1cbc8";
  ctx.fill();
  roundedRect(ctx, -width / 2 + inset, height * 0.06, width - inset * 2, height * 0.2, 3);
  ctx.fillStyle = "#263330";
  ctx.fill();

  ctx.fillStyle = "#f3e7bd";
  ctx.fillRect(-width * 0.33, -height * 0.48, width * 0.18, height * 0.035);
  ctx.fillRect(width * 0.15, -height * 0.48, width * 0.18, height * 0.035);
  ctx.fillStyle = braking ? "#e56d5f" : "#8e3934";
  ctx.fillRect(-width * 0.33, height * 0.445, width * 0.18, height * 0.04);
  ctx.fillRect(width * 0.15, height * 0.445, width * 0.18, height * 0.04);

  if (ego) {
    ctx.strokeStyle = "rgba(255,255,255,0.82)";
    ctx.lineWidth = Math.max(1.2, width * 0.04);
    roundedRect(ctx, -width / 2, -height / 2, width, height, width * 0.24);
    ctx.stroke();
  }
  ctx.restore();
}

function interpolateFrame(previous, current, mix) {
  if (!previous) return current;
  const ego = current.e.map((value, index) => {
    if (index === 2) return value;
    return index === 0
      ? lerpWithJump(previous.e[index], value, mix, 60)
      : lerp(previous.e[index], value, mix);
  });
  const previousCars = new Map(previous.c.map((car) => [car[0], car]));
  const cars = current.c.map((car) => {
    const before = previousCars.get(car[0]);
    if (!before) return car;
    return [
      car[0],
      lerpWithJump(before[1], car[1], mix),
      lerp(before[2], car[2], mix),
      lerp(before[3], car[3], mix),
      car[4],
      car[5],
      car[6],
    ];
  });
  return { e: ego, c: cars, s: current.s, t: current.t };
}

function drawSensors(layout, data) {
  const { roadLeft, laneWidth, egoY, metresToPixels } = layout;
  ctx.save();
  ctx.setLineDash([5, 6]);
  ctx.lineWidth = 1.2;
  data.s.forEach((sensor, lane) => {
    const x = roadLeft + laneWidth * (lane + 0.5);
    const ahead = Math.min(sensor[0], 70) * metresToPixels;
    const behind = Math.min(sensor[2], 42) * metresToPixels;
    ctx.strokeStyle = sensor[0] < 18 ? "rgba(224, 155, 91, .9)" : "rgba(175, 207, 195, .62)";
    ctx.beginPath();
    ctx.moveTo(x, egoY - 5);
    ctx.lineTo(x, egoY - ahead);
    ctx.stroke();
    ctx.strokeStyle = "rgba(175, 207, 195, .38)";
    ctx.beginPath();
    ctx.moveTo(x, egoY + 5);
    ctx.lineTo(x, egoY + behind);
    ctx.stroke();
    ctx.fillStyle = "rgba(239, 234, 208, .85)";
    ctx.beginPath();
    ctx.arc(x, egoY - ahead, 2.8, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

function drawRoad(data) {
  const width = ui.canvas.width;
  const height = ui.canvas.height;
  const mobile = width / height < 1.25;
  const lanes = state.meta?.road?.lanes || 4;
  const roadWidth = mobile ? width * 0.84 : Math.min(width * 0.69, height * 0.81);
  const roadLeft = (width - roadWidth) / 2;
  const laneWidth = roadWidth / lanes;
  const egoY = height * (mobile ? 0.72 : 0.73);
  const metresToPixels = height / (mobile ? 92 : 104);
  const roadScroll = (data.e[0] * metresToPixels) % (height * 0.12);
  const shoulder = Math.max(8, laneWidth * 0.08);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#738477";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "rgba(37, 55, 47, .16)";
  for (let y = -60 + roadScroll; y < height + 60; y += 64) {
    ctx.fillRect(0, y, roadLeft - shoulder * 1.5, 1);
    ctx.fillRect(roadLeft + roadWidth + shoulder * 1.5, y, width, 1);
  }
  ctx.fillStyle = "#c7c4b9";
  ctx.fillRect(roadLeft - shoulder, 0, roadWidth + shoulder * 2, height);
  ctx.fillStyle = "#313735";
  ctx.fillRect(roadLeft, 0, roadWidth, height);
  ctx.fillStyle = "rgba(255,255,255,.56)";
  ctx.fillRect(roadLeft + 2, 0, 2, height);
  ctx.fillRect(roadLeft + roadWidth - 4, 0, 2, height);

  const dashLength = height * 0.052;
  const dashGap = height * 0.048;
  const period = dashLength + dashGap;
  for (let lane = 1; lane < lanes; lane += 1) {
    const x = roadLeft + lane * laneWidth;
    for (let y = -period + (roadScroll % period); y < height + period; y += period) {
      ctx.fillStyle = "rgba(239, 238, 224, .68)";
      ctx.fillRect(x - 1.2, y, 2.4, dashLength);
    }
  }

  if (state.sensors) drawSensors({ roadLeft, laneWidth, egoY, metresToPixels }, data);

  const carWidth = Math.min(laneWidth * 0.4, 45);
  const carHeight = carWidth * 2.22;
  data.c
    .map((car) => ({ car, y: egoY - car[1] * metresToPixels }))
    .filter(({ y }) => y > -carHeight && y < height + carHeight)
    .sort((a, b) => a.y - b.y)
    .forEach(({ car, y }) => {
      const x = roadLeft + laneWidth * (car[2] + 0.5);
      drawVehicle(x, y, carWidth, carHeight, carColors[car[5] % carColors.length], car[4], false, car[6]);
    });

  const egoX = roadLeft + laneWidth * (data.e[1] + 0.5);
  drawVehicle(egoX, egoY, carWidth * 1.05, carHeight * 1.04, "#27594a", data.e[7] > 0.08, true, 0);
  const targetX = roadLeft + laneWidth * (data.e[2] + 0.5);
  if (Math.abs(data.e[1] - data.e[2]) > 0.06) {
    ctx.save();
    ctx.strokeStyle = "rgba(230, 188, 114, .84)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(egoX, egoY - carHeight * 0.65);
    ctx.quadraticCurveTo((egoX + targetX) / 2, egoY - carHeight, targetX, egoY - carHeight * 1.28);
    ctx.stroke();
    ctx.restore();
  }

  ctx.save();
  ctx.fillStyle = "rgba(255, 254, 249, .78)";
  ctx.font = `${Math.max(10, width * 0.009)}px ui-monospace, monospace`;
  ctx.textAlign = "center";
  ctx.fillText("DIRECTION OF TRAVEL", width / 2, height - 17);
  ctx.restore();
}

function renderTelemetry(data) {
  const telemetry = data.t;
  const intent = normalizeIntent(telemetry.intent);
  const speedKmh = data.e[3] * 3.6;
  const targetKmh = telemetry.desired_speed * 3.6;
  const netPasses = telemetry.overtakes - telemetry.passed_by_traffic;
  ui.intent.textContent = intentLabels[intent] || intent.toLowerCase().replaceAll("_", " ");
  ui.intentMark.textContent = intent.includes("LEFT") ? "↖" : intent.includes("RIGHT") ? "↗" : intent === "EMERGENCY" ? "!" : "↑";
  ui.intentReason.textContent = telemetry.reason;
  ui.speedValue.textContent = speedKmh.toFixed(0);
  ui.speedFill.style.width = `${clamp((speedKmh / 122) * 100, 0, 100)}%`;
  ui.targetPin.style.left = `${clamp((targetKmh / 122) * 100, 0, 100)}%`;
  ui.targetSpeed.textContent = `Target ${targetKmh.toFixed(0)}`;
  ui.ttc.textContent = formatTtc(telemetry.ttc);
  ui.rearTtc.textContent = formatTtc(telemetry.rear_ttc);
  ui.passes.textContent = netPasses > 0 ? `+${netPasses}` : String(netPasses);
  ui.distance.textContent = `${(data.e[0] / 1000).toFixed(2)} km`;
  ui.laneChanges.textContent = String(telemetry.lane_changes);
  ui.latency.textContent = `${telemetry.inference_ms.toFixed(1)} ms`;
  ui.challenge.textContent = telemetry.challenge;
  ui.action.textContent = `Control: ${telemetry.action.replaceAll("+", " + ")}`;
  ui.layers.textContent = `Learned layer: ${telemetry.preference_decision}`;
  ui.braking.textContent = `Speed controller: ${telemetry.braking_mode}`;
  ui.elapsed.textContent = formatTime(telemetry.elapsed_seconds);
  ui.trafficEvents.textContent = `${telemetry.traffic_lane_changes} lane changes · ${telemetry.near_misses} near misses`;
  ui.canvas.setAttribute(
    "aria-label",
    `Live drive at ${telemetry.elapsed_seconds.toFixed(1)} seconds. ${speedKmh.toFixed(0)} kilometres per hour, ${ui.intent.textContent}.`,
  );
}

function resizeCanvas() {
  const rect = ui.shell.getBoundingClientRect();
  const density = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * density));
  const height = Math.max(1, Math.round(rect.height * density));
  if (ui.canvas.width !== width || ui.canvas.height !== height) {
    ui.canvas.width = width;
    ui.canvas.height = height;
  }
}

function render(now) {
  if (!state.current) return;
  resizeCanvas();
  const mix = state.settings.paused
    ? 1
    : clamp((now - state.receivedAt) / Math.max(state.expectedInterval, 1), 0, 1);
  const data = interpolateFrame(state.previous, state.current, mix);
  drawRoad(data);
  renderTelemetry(data);
}

function animationFrame(now) {
  render(now);
  requestAnimationFrame(animationFrame);
}

function handleMessage(message) {
  if (message.type === "loading") {
    showLoading("Waking the live driver", message.message || "Loading the frozen policy on CPU…");
    setConnectionState("connecting", "Loading policy");
    return;
  }
  if (message.type === "hello") {
    state.meta = message;
    syncSettings(message.settings);
    return;
  }
  if (message.type === "status") {
    syncSettings(message.settings);
    return;
  }
  if (message.type === "state") {
    const episodeChanged = state.episode !== 0 && message.episode !== state.episode;
    state.previous = episodeChanged ? null : state.current;
    state.current = message.frame;
    state.receivedAt = performance.now();
    state.episode = message.episode;
    ui.episodeValue.textContent = String(message.episode);
    syncSettings(message.settings);
    state.connected = true;
    state.reconnectAttempt = 0;
    setControlsEnabled(true);
    hideLoading();
    updatePauseUi();
    return;
  }
  if (message.type === "outcome") {
    const detail = message.outcome === "crashed"
      ? `Crash after ${message.elapsed_seconds.toFixed(1)} s · restarting on seed ${Number(message.next_seed).toLocaleString()}`
      : `Run ended after ${message.elapsed_seconds.toFixed(1)} s · restarting`;
    showOutcome(detail);
    return;
  }
  if (message.type === "error") {
    showLoading("Live driver unavailable", message.message || "Please try again shortly.", false);
    setConnectionState("error", message.code === "busy" ? "Demo busy" : "Driver stopped");
    setControlsEnabled(false);
  }
}

function connect() {
  window.clearTimeout(state.reconnectTimer);
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${window.location.host}/ws`);
  state.socket = socket;
  state.connected = false;
  setControlsEnabled(false);
  setConnectionState("connecting", "Connecting");
  showLoading("Waking the live driver", "Connecting to a fresh policy session…");

  socket.addEventListener("open", () => {
    setConnectionState("connecting", "Loading policy");
  });
  socket.addEventListener("message", (event) => {
    try {
      handleMessage(JSON.parse(event.data));
    } catch (error) {
      showLoading("Unexpected live data", error.message, false);
    }
  });
  socket.addEventListener("close", () => {
    if (state.socket !== socket) return;
    state.connected = false;
    setControlsEnabled(false);
    setConnectionState("connecting", "Reconnecting");
    showLoading("Reconnecting", "The live session ended. Starting another…");
    const delay = Math.min(10000, 900 * 2 ** state.reconnectAttempt);
    state.reconnectAttempt += 1;
    state.reconnectTimer = window.setTimeout(connect, delay);
  });
  socket.addEventListener("error", () => {
    setConnectionState("error", "Connection interrupted");
  });
}

ui.play.addEventListener("click", () => {
  state.settings.paused = !state.settings.paused;
  sendControl(state.settings.paused ? "pause" : "resume");
  updatePauseUi();
});

ui.newSeed.addEventListener("click", () => {
  sendControl("reset", {
    seed: null,
    difficulty: ui.difficulty.value,
    dynamic_traffic: ui.dynamic.checked,
  });
  showOutcome("Generating a fresh deterministic traffic seed…");
});

ui.restart.addEventListener("click", () => {
  sendControl("reset", {
    seed: state.settings.seed,
    difficulty: ui.difficulty.value,
    dynamic_traffic: ui.dynamic.checked,
  });
  showOutcome(`Restarting seed ${Number(state.settings.seed).toLocaleString()}…`);
});

ui.difficulty.addEventListener("change", () => {
  sendControl("reset", {
    seed: state.settings.seed,
    difficulty: ui.difficulty.value,
    dynamic_traffic: ui.dynamic.checked,
  });
});

ui.dynamic.addEventListener("change", () => {
  sendControl("reset", {
    seed: state.settings.seed,
    difficulty: ui.difficulty.value,
    dynamic_traffic: ui.dynamic.checked,
  });
});

ui.rate.addEventListener("change", () => {
  state.settings.rate = Number(ui.rate.value);
  state.expectedInterval = 1000 / ((state.meta?.decision_hz || 10) * state.settings.rate);
  sendControl("rate", { value: state.settings.rate });
});

ui.sensors.addEventListener("click", () => {
  state.sensors = !state.sensors;
  ui.sensors.setAttribute("aria-pressed", String(state.sensors));
});

window.addEventListener("keydown", (event) => {
  if (event.code === "Space" && !["INPUT", "SELECT", "BUTTON"].includes(document.activeElement.tagName)) {
    event.preventDefault();
    ui.play.click();
  }
});

window.addEventListener("beforeunload", () => {
  window.clearTimeout(state.reconnectTimer);
  state.socket?.close();
});

new ResizeObserver(() => render(performance.now())).observe(ui.shell);
requestAnimationFrame(animationFrame);
connect();
