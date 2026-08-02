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
  pyodide: null,
  baseSession: null,
  overrideSession: null,
  ready: false,
  stepping: false,
  pendingReset: null,
  restartAt: null,
  meta: {
    decision_hz: 10,
    road: { lanes: 4, lane_width_m: 3.7, car_length_m: 4.6, car_width_m: 1.9 },
  },
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
  outcomeTimer: null,
};

const PYODIDE_VERSION = "314.0.3";
const ONNX_VERSION = "1.27.0";
const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const ONNX_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ONNX_VERSION}/dist/`;
const SOURCE_BASE = "https://raw.githubusercontent.com/becauseno7/Self-Driving-RL/v1.0.0/src/self_driving_rl";
const LOCAL_MODELS = ["127.0.0.1", "localhost"].includes(window.location.hostname);
const MODEL_BASE = LOCAL_MODELS
  ? "/artifacts/browser-policy-v1"
  : "https://huggingface.co/slicedonions/self-driving-rl-v1/resolve/main";
const MODEL_FILES = {
  base: {
    url: `${MODEL_BASE}/v5-qrdqn.onnx`,
    sha256: "970C9A7F725E2921228EEA22977B358C8BC242D58815DD933945CEE71B51C26E",
  },
  override: {
    url: `${MODEL_BASE}/v6-rlaif-override.onnx`,
    sha256: "D699D6845AEC7A642FBEBC81701580DCEEC3AE7A18DAB81A1B3A1A64ED97BA97",
  },
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
  ui.footerState.textContent = `Local browser inference · ${copy.toLowerCase()}`;
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
  if (state.ready) setConnectionState(paused ? "paused" : "live", paused ? "Simulation paused" : "Policy deciding live");
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

function requestReset(seed) {
  state.pendingReset = {
    seed,
    difficulty: ui.difficulty.value,
    dynamic_traffic: ui.dynamic.checked,
  };
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

function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function argmax(values) {
  let bestIndex = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[bestIndex]) bestIndex = index;
  }
  return bestIndex;
}

function softmax(values) {
  const maximum = Math.max(...values);
  const exponentials = Array.from(values, (value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

async function sha256Hex(buffer) {
  const hash = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(hash), (byte) => byte.toString(16).padStart(2, "0"))
    .join("")
    .toUpperCase();
}

async function verifiedModel({ url, sha256 }, label) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${label} download failed (${response.status})`);
  const buffer = await response.arrayBuffer();
  const actualHash = await sha256Hex(buffer);
  if (actualHash !== sha256) throw new Error(`${label} failed its integrity check`);
  return buffer;
}

async function loadBrowserModels() {
  window.ort.env.wasm.wasmPaths = ONNX_BASE;
  window.ort.env.wasm.numThreads = 1;
  const [baseBytes, overrideBytes] = await Promise.all([
    verifiedModel(MODEL_FILES.base, "V5 policy"),
    verifiedModel(MODEL_FILES.override, "V6 preference layer"),
  ]);
  const sessionOptions = {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  };
  [state.baseSession, state.overrideSession] = await Promise.all([
    window.ort.InferenceSession.create(baseBytes, sessionOptions),
    window.ort.InferenceSession.create(overrideBytes, sessionOptions),
  ]);
}

async function loadPythonRuntime() {
  state.pyodide = await window.loadPyodide({ indexURL: PYODIDE_BASE });
  await state.pyodide.loadPackage("numpy");
  const sourceUrls = [
    `${SOURCE_BASE}/game_env.py`,
    `${SOURCE_BASE}/longitudinal.py`,
    "./python/browser_runtime.py",
  ];
  const responses = await Promise.all(sourceUrls.map((url) => fetch(url)));
  const failedIndex = responses.findIndex((response) => !response.ok);
  if (failedIndex >= 0) {
    throw new Error(`Simulator source download failed (${responses[failedIndex].status})`);
  }
  const [gameEnvSource, longitudinalSource, runtimeSource] = await Promise.all(
    responses.map((response) => response.text()),
  );
  state.pyodide.globals.set("game_env_source", gameEnvSource);
  state.pyodide.globals.set("longitudinal_source", longitudinalSource);
  await state.pyodide.runPythonAsync(runtimeSource);
}

function randomSeed() {
  const values = new Uint32Array(1);
  crypto.getRandomValues(values);
  return 100000 + (values[0] % 900000);
}

function applyRuntimeState(message) {
  const episodeChanged = state.episode !== 0 && message.episode !== state.episode;
  state.previous = episodeChanged ? null : state.current;
  state.current = message.frame;
  state.receivedAt = performance.now();
  state.episode = message.episode;
  state.settings.seed = message.seed;
  ui.episodeValue.textContent = String(message.episode);
  syncSettings(state.settings);
  setControlsEnabled(true);
  hideLoading();
  updatePauseUi();
  if (message.done) {
    const nextSeed = Number(message.seed) + 1;
    const elapsed = message.frame.t.elapsed_seconds.toFixed(1);
    const label = message.outcome === "crashed" ? "Crash" : "Run ended";
    showOutcome(`${label} after ${elapsed} s · restarting on seed ${nextSeed.toLocaleString()}`);
    state.restartAt = performance.now() + 1500;
  }
}

function resetSimulation({ seed, difficulty, dynamic_traffic: dynamicTraffic }) {
  const selectedSeed = seed == null ? randomSeed() : Number(seed);
  state.settings = {
    ...state.settings,
    seed: selectedSeed,
    difficulty,
    dynamic_traffic: Boolean(dynamicTraffic),
  };
  state.pyodide.globals.set("reset_seed", selectedSeed);
  state.pyodide.globals.set("reset_difficulty", difficulty);
  state.pyodide.globals.set("reset_dynamic", Boolean(dynamicTraffic));
  const result = state.pyodide.runPython(
    "runtime.reset(int(reset_seed), str(reset_difficulty), bool(reset_dynamic))",
  );
  state.restartAt = null;
  applyRuntimeState(JSON.parse(result));
}

async function inferenceStep() {
  const prepared = JSON.parse(state.pyodide.runPython("runtime.prepare_json()"));
  const startedAt = performance.now();
  const baseInput = new window.ort.Tensor(
    "float32",
    Float32Array.from(prepared.observation),
    [1, 33],
  );
  const baseOutputs = await state.baseSession.run({ observation: baseInput });
  const baseAction = argmax(baseOutputs.q_values.data);
  const overrideInput = new window.ort.Tensor(
    "float32",
    Float32Array.from(prepared.override_observation),
    [1, 35],
  );
  const baseActionInput = new window.ort.Tensor(
    "int64",
    BigInt64Array.from([BigInt(baseAction)]),
    [1],
  );
  const overrideOutputs = await state.overrideSession.run({
    observation: overrideInput,
    base_action: baseActionInput,
  });
  const probabilities = softmax(overrideOutputs.kind_logits.data);
  const proposedAction = argmax(overrideOutputs.action_logits.data);
  const inferenceMs = performance.now() - startedAt;
  state.pyodide.globals.set("step_base_action", baseAction);
  state.pyodide.globals.set("step_probabilities_json", JSON.stringify(probabilities));
  state.pyodide.globals.set("step_proposed_action", proposedAction);
  state.pyodide.globals.set("step_inference_ms", inferenceMs);
  const result = state.pyodide.runPython(
    "runtime.step(int(step_base_action), json.loads(step_probabilities_json), " +
      "int(step_proposed_action), float(step_inference_ms))",
  );
  applyRuntimeState(JSON.parse(result));
}

async function policyLoop() {
  while (state.ready) {
    if (state.pendingReset) {
      const reset = state.pendingReset;
      state.pendingReset = null;
      resetSimulation(reset);
      continue;
    }
    if (state.restartAt && performance.now() >= state.restartAt) {
      resetSimulation({
        seed: Number(state.settings.seed) + 1,
        difficulty: state.settings.difficulty,
        dynamic_traffic: state.settings.dynamic_traffic,
      });
      continue;
    }
    if (state.settings.paused || state.restartAt) {
      await sleep(40);
      continue;
    }
    const startedAt = performance.now();
    await inferenceStep();
    const interval = 1000 / ((state.meta?.decision_hz || 10) * state.settings.rate);
    await sleep(Math.max(0, interval - (performance.now() - startedAt)));
  }
}

async function boot() {
  setControlsEnabled(false);
  setConnectionState("connecting", "Loading policy");
  showLoading("Loading the live driver", "Starting Python and verifying the learned policy…");
  try {
    await Promise.all([loadPythonRuntime(), loadBrowserModels()]);
    resetSimulation({
      seed: randomSeed(),
      difficulty: state.settings.difficulty,
      dynamic_traffic: state.settings.dynamic_traffic,
    });
    state.ready = true;
    updatePauseUi();
    void policyLoop();
  } catch (error) {
    console.error(error);
    showLoading("Live driver unavailable", error.message || "Initialization failed.", false);
    setConnectionState("error", "Driver stopped");
    setControlsEnabled(false);
  }
}

ui.play.addEventListener("click", () => {
  state.settings.paused = !state.settings.paused;
  updatePauseUi();
});

ui.newSeed.addEventListener("click", () => {
  requestReset(null);
  showOutcome("Generating a fresh deterministic traffic seed…");
});

ui.restart.addEventListener("click", () => {
  requestReset(state.settings.seed);
  showOutcome(`Restarting seed ${Number(state.settings.seed).toLocaleString()}…`);
});

ui.difficulty.addEventListener("change", () => {
  requestReset(state.settings.seed);
});

ui.dynamic.addEventListener("change", () => {
  requestReset(state.settings.seed);
});

ui.rate.addEventListener("change", () => {
  state.settings.rate = Number(ui.rate.value);
  state.expectedInterval = 1000 / ((state.meta?.decision_hz || 10) * state.settings.rate);
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

new ResizeObserver(() => render(performance.now())).observe(ui.shell);
requestAnimationFrame(animationFrame);
void boot();
