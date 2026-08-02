const $ = (selector) => document.querySelector(selector);

const ui = {
  canvas: $("#road"),
  shell: $("#canvas-shell"),
  load: $("#load-state"),
  runState: $("#run-state"),
  canvasSeed: $("#canvas-seed"),
  play: $("#play"),
  playLabel: $("#play-label"),
  scrub: $("#scrub"),
  elapsed: $("#elapsed"),
  duration: $("#duration"),
  speedSelect: $("#speed"),
  seedSelect: $("#seed"),
  restart: $("#restart"),
  sensors: $("#sensors"),
  intent: $("#intent"),
  intentMark: $("#intent-mark"),
  intentReason: $("#intent-reason"),
  speedValue: $("#speed-value"),
  speedFill: $("#speed-fill"),
  targetPin: $("#target-pin"),
  targetSpeed: $("#target-speed"),
  ttc: $("#ttc"),
  passes: $("#passes"),
  distance: $("#distance"),
  return: $("#return"),
  challenge: $("#challenge"),
  challengeFill: $("#challenge-fill"),
  action: $("#action"),
  outcome: $("#outcome"),
  summaryDistance: $("#summary-distance"),
  summaryCopy: $("#summary-copy"),
  sourceCommit: $("#source-commit"),
  baseHash: $("#base-hash"),
  overrideHash: $("#override-hash"),
  schema: $("#schema"),
};

const ctx = ui.canvas.getContext("2d");
const state = {
  manifest: null,
  replay: null,
  playhead: 0,
  duration: 0,
  playing: true,
  rate: 1,
  sensors: false,
  previousTime: null,
  loadToken: 0,
};

const carColors = ["#9b5d55", "#557486", "#b19a61", "#7b746d", "#607b68", "#8c6c85"];
const actionGlyphs = { LEFT: "←", RIGHT: "→", BRAKE: "↓", GAS: "↑", HOLD: "•" };
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

function lerpAngleSafe(a, b, amount, jump = 80) {
  return Math.abs(b - a) > jump ? (amount < 0.5 ? a : b) : lerp(a, b, amount);
}

function formatTime(seconds) {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60).toString().padStart(2, "0");
  const remainder = (safe % 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${remainder}`;
}

function normalizeIntent(value) {
  return String(value || "CRUISE").replace(/^DrivingIntent\./, "").toUpperCase();
}

function shortHash(hash) {
  return hash ? `${hash.slice(0, 12)}…${hash.slice(-8)}` : "—";
}

function currentSample() {
  if (!state.replay) return null;
  const hz = state.replay.meta.sample_hz;
  const position = clamp(state.playhead * hz, 0, state.replay.frames.length - 1);
  const index = Math.floor(position);
  return {
    a: state.replay.frames[index],
    b: state.replay.frames[Math.min(index + 1, state.replay.frames.length - 1)],
    mix: position - index,
    index,
  };
}

function interpolated(sample) {
  const { a, b, mix } = sample;
  const ego = a.e.map((value, index) =>
    index === 2 ? (mix < 0.5 ? value : b.e[index]) : lerp(value, b.e[index], mix),
  );
  const nextCars = new Map(b.c.map((car) => [car[0], car]));
  const cars = a.c.map((car) => {
    const next = nextCars.get(car[0]) || car;
    return [
      car[0],
      lerpAngleSafe(car[1], next[1], mix),
      lerp(car[2], next[2], mix),
      lerp(car[3], next[3], mix),
      mix < 0.5 ? car[4] : next[4],
      mix < 0.5 ? car[5] : next[5],
      mix < 0.5 ? car[6] : next[6],
    ];
  });
  return { ego, cars, sensors: mix < 0.5 ? a.s : b.s, telemetry: mix < 0.5 ? a.x : b.x };
}

function roundedRect(context, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.roundRect(x, y, width, height, r);
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
  roundedRect(ctx, -width / 2 + inset, height * 0.06, width - inset * 2, height * 0.20, 3);
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

function drawSensors(layout, data) {
  const { roadLeft, laneWidth, egoY, metresToPixels } = layout;
  ctx.save();
  ctx.setLineDash([5, 6]);
  ctx.lineWidth = 1.2;
  data.sensors.forEach((sensor, lane) => {
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
  const lanes = state.replay.meta.road.lanes;
  const roadWidth = mobile ? width * 0.84 : Math.min(width * 0.69, height * 0.81);
  const roadLeft = (width - roadWidth) / 2;
  const laneWidth = roadWidth / lanes;
  const egoY = height * (mobile ? 0.72 : 0.73);
  const metresToPixels = height / (mobile ? 92 : 104);
  const roadScroll = (data.ego[0] * metresToPixels) % (height * 0.12);
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

  if (state.sensors) {
    drawSensors({ roadLeft, laneWidth, egoY, metresToPixels }, data);
  }

  const carWidth = Math.min(laneWidth * 0.40, 45);
  const carHeight = carWidth * 2.22;
  const visibleCars = data.cars
    .map((car) => ({ car, y: egoY - car[1] * metresToPixels }))
    .filter(({ y }) => y > -carHeight && y < height + carHeight)
    .sort((a, b) => a.y - b.y);

  visibleCars.forEach(({ car, y }) => {
    const x = roadLeft + laneWidth * (car[2] + 0.5);
    drawVehicle(x, y, carWidth, carHeight, carColors[car[5] % carColors.length], car[4], false, car[6]);
  });

  const egoX = roadLeft + laneWidth * (data.ego[1] + 0.5);
  drawVehicle(egoX, egoY, carWidth * 1.05, carHeight * 1.04, "#27594a", data.ego[7] > 0.08, true, 0);

  const targetX = roadLeft + laneWidth * (data.ego[2] + 0.5);
  if (Math.abs(data.ego[1] - data.ego[2]) > 0.06) {
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
  const x = data.telemetry;
  const intent = normalizeIntent(x[2]);
  const speedKmh = data.ego[3] * 3.6;
  const targetKmh = data.ego[4] * 3.6;
  const ttc = x[11];
  const actionName = state.replay.action_names[String(x[0])] || "HOLD";
  const netPasses = x[7] - x[8];
  const progress = state.duration ? state.playhead / state.duration : 0;

  ui.intent.textContent = intentLabels[intent] || intent.toLowerCase().replaceAll("_", " ");
  ui.intentMark.textContent = intent.includes("LEFT") ? "↖" : intent.includes("RIGHT") ? "↗" : intent === "EMERGENCY" ? "!" : "↑";
  ui.intentReason.textContent = x[4];
  ui.speedValue.textContent = speedKmh.toFixed(0);
  ui.speedFill.style.width = `${clamp(speedKmh / 122 * 100, 0, 100)}%`;
  ui.targetPin.style.left = `${clamp(targetKmh / 122 * 100, 0, 100)}%`;
  ui.targetSpeed.textContent = `Target ${targetKmh.toFixed(0)}`;
  ui.ttc.textContent = ttc >= 90 ? "Clear" : `${ttc.toFixed(1)} s`;
  ui.passes.textContent = netPasses > 0 ? `+${netPasses}` : String(netPasses);
  ui.distance.textContent = `${(data.ego[0] / 1000).toFixed(2)} km`;
  ui.return.textContent = `${x[1] >= 0 ? "+" : ""}${x[1].toFixed(1)}`;
  ui.challenge.textContent = x[5];
  ui.challengeFill.style.width = `${progress * 100}%`;
  ui.action.textContent = `Control: ${actionName.replaceAll("+", " + ")}`;
  ui.elapsed.textContent = formatTime(state.playhead);
  ui.scrub.value = state.playhead.toFixed(2);
  ui.canvas.setAttribute(
    "aria-label",
    `Recorded drive at ${state.playhead.toFixed(1)} seconds. ${speedKmh.toFixed(0)} kilometres per hour, ${ui.intent.textContent}.`,
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

function render() {
  const sample = currentSample();
  if (!sample) return;
  resizeCanvas();
  const data = interpolated(sample);
  drawRoad(data);
  renderTelemetry(data);
}

function updatePlayState() {
  const atEnd = state.playhead >= state.duration - 0.001;
  const paused = !state.playing;
  ui.play.classList.toggle("is-paused", paused);
  ui.playLabel.textContent = paused ? (atEnd ? "Replay" : "Resume") : "Pause";
  ui.play.setAttribute("aria-label", paused ? (atEnd ? "Replay route" : "Resume replay") : "Pause replay");
  ui.runState.classList.toggle("paused", paused);
  ui.runState.innerHTML = `<i></i> ${atEnd ? "Route complete" : paused ? "Playback paused" : "Recorded run playing"}`;
}

function setPlaying(playing) {
  if (playing && state.playhead >= state.duration - 0.001) state.playhead = 0;
  state.playing = playing;
  state.previousTime = null;
  updatePlayState();
  render();
}

function frame(time) {
  if (state.replay) {
    if (state.previousTime === null) state.previousTime = time;
    const delta = Math.min((time - state.previousTime) / 1000, 0.1);
    state.previousTime = time;
    if (state.playing) {
      state.playhead = Math.min(state.duration, state.playhead + delta * state.rate);
      if (state.playhead >= state.duration) {
        state.playing = false;
        updatePlayState();
      }
    }
    render();
  }
  requestAnimationFrame(frame);
}

function populateSummary(replay) {
  const final = replay.final;
  ui.outcome.textContent = final.outcome;
  ui.summaryDistance.textContent = `${final.distance_km.toFixed(2)} km · ${final.elapsed_seconds.toFixed(1)} s`;
  ui.summaryCopy.textContent = `${final.overtakes} overtakes, ${final.lane_changes} lane changes, ${final.near_misses} near misses.`;
  ui.sourceCommit.textContent = replay.meta.source_commit;
  ui.baseHash.textContent = replay.meta.models.base.sha256;
  ui.overrideHash.textContent = replay.meta.models.override.sha256;
  ui.schema.textContent = replay.schema;
  ui.baseHash.title = replay.meta.models.base.sha256;
  ui.overrideHash.title = replay.meta.models.override.sha256;
}

async function loadReplay(entry, autoplay = true) {
  const token = ++state.loadToken;
  ui.load.hidden = false;
  ui.load.querySelector("strong").textContent = "Opening the road record";
  ui.load.querySelector("small").textContent = `Loading seed ${entry.seed.toLocaleString()}…`;
  try {
    const response = await fetch(`./data/${entry.file}`);
    if (!response.ok) throw new Error(`Replay request failed (${response.status})`);
    const replay = await response.json();
    if (token !== state.loadToken) return;
    if (replay.schema !== state.manifest.replay_schema || !Array.isArray(replay.frames)) {
      throw new Error("The replay file does not match this viewer.");
    }
    state.replay = replay;
    state.duration = (replay.frames.length - 1) / replay.meta.sample_hz;
    state.playhead = 0;
    state.playing = autoplay;
    state.previousTime = null;
    ui.scrub.max = state.duration.toFixed(2);
    ui.duration.textContent = formatTime(state.duration);
    ui.canvasSeed.textContent = `Traffic seed ${replay.meta.seed.toLocaleString()}`;
    populateSummary(replay);
    updatePlayState();
    render();
    ui.load.hidden = true;
  } catch (error) {
    if (token !== state.loadToken) return;
    state.playing = false;
    ui.load.hidden = false;
    ui.load.querySelector(".loader").style.display = "none";
    ui.load.querySelector("strong").textContent = "Replay unavailable";
    ui.load.querySelector("small").textContent = error.message;
    ui.runState.innerHTML = "<i></i> Could not load recording";
  }
}

async function boot() {
  try {
    const response = await fetch("./data/manifest.json");
    if (!response.ok) throw new Error(`Manifest request failed (${response.status})`);
    state.manifest = await response.json();
    if (!Array.isArray(state.manifest.replays) || state.manifest.replays.length === 0) {
      throw new Error("No recorded routes are listed.");
    }
    state.manifest.replays.forEach((entry, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `Seed ${entry.seed.toLocaleString()} · ${entry.final.outcome}`;
      ui.seedSelect.append(option);
    });
    await loadReplay(state.manifest.replays[0]);
  } catch (error) {
    ui.load.querySelector(".loader").style.display = "none";
    ui.load.querySelector("strong").textContent = "Replay unavailable";
    ui.load.querySelector("small").textContent = error.message;
  }
}

ui.play.addEventListener("click", () => setPlaying(!state.playing));
ui.restart.addEventListener("click", () => {
  state.playhead = 0;
  setPlaying(true);
});
ui.speedSelect.addEventListener("change", () => {
  state.rate = Number(ui.speedSelect.value);
});
ui.seedSelect.addEventListener("change", () => {
  const entry = state.manifest.replays[Number(ui.seedSelect.value)];
  loadReplay(entry);
});
ui.sensors.addEventListener("click", () => {
  state.sensors = !state.sensors;
  ui.sensors.setAttribute("aria-pressed", String(state.sensors));
  render();
});
ui.scrub.addEventListener("input", () => {
  state.playhead = Number(ui.scrub.value);
  state.previousTime = null;
  updatePlayState();
  render();
});
window.addEventListener("keydown", (event) => {
  if (event.code === "Space" && !["INPUT", "SELECT", "BUTTON"].includes(document.activeElement.tagName)) {
    event.preventDefault();
    setPlaying(!state.playing);
  }
});
new ResizeObserver(render).observe(ui.shell);

requestAnimationFrame(frame);
boot();
