/* V2X-TwinStream front end: one master clock drives three synchronized feeds (physical POV, live telemetry, defender walkthrough)
   plus the decision-gate modal. Telemetry is fetched once per decision path as frame-aligned columns (server-side SyncEngine), so what you
   see at video frame k is exactly the value the API returns for /frame/k. */
"use strict";
const $ = (id) => document.getElementById(id);
const C = { blue: "#0072B2", orange: "#E69F00", red: "#D55E00", green: "#009E73", purple: "#8a5cc2", grey: "#6b7480", ink: "#1b1f24", line: "#d9dee4" };

const S = { sc: null, tl: null, path: null, decisions: [], t: 0, playing: false, last: 0, nFrames: 0, gateOpen: false, ended: false, lidar: {}, lidarBusy: {} };

async function api(url, opt) { const r = await fetch(url, opt); if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.statusText); } return r.json(); }
const dq = () => encodeURIComponent(S.decisions.join(","));

/* ------------------------------------------------------------------ track loading */
async function boot() {
  const tracks = await api("/api/tracks");
  $("trackSel").innerHTML = tracks.map((t) => `<option value="${t.id}">${t.title}</option>`).join("");
  $("trackSel").onchange = () => loadTrack($("trackSel").value);
  const want = new URLSearchParams(location.search).get("track");
  if (want) $("trackSel").value = want;
  await loadTrack($("trackSel").value);
  const q = new URLSearchParams(location.search);                      // deep-link hooks (used by the screenshot script)
  if (q.get("d")) { S.decisions = q.get("d").split(","); await reloadTimeline(); }
  if (q.get("t")) S.t = +q.get("t");
  if (q.get("play")) { S.playing = true; S.last = performance.now(); $("btnPlay").textContent = "Pause"; }
  requestAnimationFrame(tick);
}

async function loadTrack(id) {
  S.playing = false; S.ended = false; S.decisions = []; S.t = 0; $("btnPlay").textContent = "Play"; closeModal();
  S.sc = await api(`/api/tracks/${id}`); S.nFrames = Math.floor(S.sc.duration_ms * S.sc.fps / 1000);
  $("scrub").max = S.sc.duration_ms; $("score").textContent = "0"; $("feedback").innerHTML = "";
  $("ctfPanel").hidden = S.sc.domain !== "ctf"; $("ctfVideo").hidden = S.sc.domain !== "ctf"; $("pov").hidden = S.sc.domain === "ctf";
  if (S.sc.domain === "ctf") $("ctfVideo").src = "/api/ctf/video";
  $("povNote").textContent = ({ cav: "top-down, real NGSIM trajectories", ev: "charger bank, true vs reported current", av: "LiDAR bird's-eye view (KITTI)", ctf: "incident footage" })[S.sc.domain];
  await reloadTimeline();
}

async function reloadTimeline() {
  try {
    [S.tl, S.path] = await Promise.all([api(`/api/tracks/${S.sc.id}/timeline?decisions=${dq()}`), api(`/api/tracks/${S.sc.id}/path?decisions=${dq()}`)]);
    $("digest").textContent = "digest " + S.tl.digest.slice(0, 12);
    $("score").textContent = S.path.score;
  } catch (e) { S.tl = null; showModal("Unavailable", "This track cannot run yet", `<p>${e.message}</p>`, []); return; }
  renderWalkthrough();
}

/* ------------------------------------------------------------------ master clock */
function gate() { const n = S.sc.nodes.find((x) => x.id === S.path.at); return n && n.gate ? n.gate : null; }
const frameAt = (t) => Math.max(0, Math.min(S.nFrames, Math.floor(t * S.sc.fps / 1000)));

function tick(now) {
  if (S.playing && S.tl) {
    const dt = (now - S.last) * parseFloat($("rate").value); S.t += dt;
    const g = gate();
    if (g && S.t >= g.t_ms) { S.t = g.t_ms; S.playing = false; $("btnPlay").textContent = "Play"; openGate(g); }
    else if (S.t >= S.sc.duration_ms) { S.t = S.sc.duration_ms; S.playing = false; $("btnPlay").textContent = "Replay"; S.ended = true; openDebrief(); }
  }
  S.last = now; draw(); requestAnimationFrame(tick);
}

$("btnPlay").onclick = () => {
  if (!S.tl) return;
  if (S.ended || S.t >= S.sc.duration_ms) { S.t = 0; S.ended = false; }
  const g = gate(); if (g && S.t >= g.t_ms) { openGate(g); return; }
  S.playing = !S.playing; $("btnPlay").textContent = S.playing ? "Pause" : "Play"; S.last = performance.now();
  const v = $("ctfVideo"); if (S.sc.domain === "ctf") S.playing ? v.play().catch(() => {}) : v.pause();
};
$("scrub").oninput = (e) => { let t = +e.target.value; const g = gate(); if (g) t = Math.min(t, g.t_ms); S.t = t; S.ended = false; };

/* ------------------------------------------------------------------ decision gate / debrief */
function openGate(g) {
  if (S.gateOpen) return; S.gateOpen = true;
  const btns = g.options.map((o) => `<button class="opt" data-id="${o.id}">${o.label}</button>`).join("");
  showModal("Decision gate", g.question, `<div id="opts">${btns}</div>`);
  document.querySelectorAll("#opts .opt").forEach((b) => (b.onclick = () => decide(b.dataset.id)));
}

async function decide(id) {
  S.decisions.push(id); closeModal(); S.gateOpen = false;
  await reloadTimeline(); const fb = S.path.feedback[S.path.feedback.length - 1];
  if (fb) $("feedback").insertAdjacentHTML("beforeend", `<div class="fb ${fb.score > 0 ? "good" : fb.score < 0 ? "bad" : ""}"><b>${fb.score > 0 ? "+" : ""}${fb.score}</b> ${fb.feedback}</div>`);
  S.playing = true; S.last = performance.now(); $("btnPlay").textContent = "Pause"; if (S.sc.domain === "ctf") $("ctfVideo").play().catch(() => {});
}

const fmt = (v, d = 2) => (v === null || v === undefined ? "none" : typeof v === "number" ? v.toFixed(d) : String(v));
function openDebrief() {
  const o = S.path.outcome, rows = [];
  const add = (k, v) => rows.push(`<div><small>${k}</small><b>${v}</b></div>`);
  add("Detection latency", o.latency_ms == null ? "never responded" : (o.latency_ms / 1000).toFixed(1) + " s");
  if ("min_ttc_s" in o) add("Min TTC", o.min_ttc_s.toFixed(2) + " s"); if ("min_gap_m" in o) add("Min gap", o.min_gap_m.toFixed(1) + " m"); if ("collision" in o) add("Collision", o.collision ? "YES" : "no");
  if ("voltage_dev_pu" in o) { add("Voltage deviation", o.voltage_dev_pu.toFixed(3) + " pu"); add("Max frequency deviation", o.max_f_dev_hz.toFixed(2) + " Hz"); add("Overcurrent", Math.round(o.overcurrent_charger_s) + " charger-s"); }
  if (S.sc.domain === "ctf") add("Provenance", o.verified_first ? "verified first" : "not verified");
  const nd = S.sc.nodes.find((n) => n.id === S.path.at);
  showModal("Debrief", nd ? nd.title : "Outcome", `<p>${nd ? nd.narration : ""}</p><div class="metrics">${rows.join("")}</div><p>Score: <b>${S.path.score}</b></p>
    <button id="btnRewind">Rewind to the decision and try another branch</button>`);
  $("btnRewind").onclick = async () => { S.decisions.pop(); closeModal(); S.gateOpen = false; $("feedback").innerHTML = ""; await reloadTimeline(); const g = gate(); S.t = g ? Math.max(0, g.t_ms - 1) : 0; S.ended = false; $("btnPlay").textContent = "Play"; };
}

function showModal(kind, title, body) { $("modalKind").textContent = kind; $("modalTitle").textContent = title; $("modalBody").innerHTML = body; $("modal").hidden = false; }
function closeModal() { $("modal").hidden = true; S.gateOpen = false; }

/* ------------------------------------------------------------------ defender walkthrough */
function renderWalkthrough() {
  const nodes = S.path.nodes.map((id) => S.sc.nodes.find((n) => n.id === id)); const cur = nodes[nodes.length - 1];
  $("nodeTitle").textContent = nodes.map((n) => n.title).join("  ›  "); $("narration").textContent = cur.narration;
  const steps = []; nodes.forEach((n) => (n.defender || []).forEach((s) => steps.push(s))); steps.sort((a, b) => a.t_ms - b.t_ms);
  S.steps = steps; $("steps").innerHTML = steps.map((s, i) => `<li data-i="${i}">${(s.t_ms / 1000).toFixed(1)} s &mdash; ${s.text}</li>`).join("");
}
function updateSteps() {
  let last = -1; S.steps.forEach((s, i) => { if (s.t_ms <= S.t) last = i; });
  document.querySelectorAll("#steps li").forEach((li, i) => { li.className = i <= last ? "on" + (i === last ? " now" : "") : ""; });
}

/* ------------------------------------------------------------------ telemetry */
const col = (st, c) => S.tl.streams[st][c];
const at = (st, c) => col(st, c)[Math.min(frameAt(S.t), col(st, c).length - 1)];
const CHARTS = {
  cav: () => {
    const ll = S.path.outcome.lead_len_m;
    const claimedGap = col("leader_claimed", "y_m").map((y, i) => y - col("follower", "y_m")[i] - ll);
    return { readouts: [["True gap", at("follower", "gap_m").toFixed(1) + " m", at("follower", "gap_m") < 3], ["TTC", Math.min(99, at("follower", "ttc_s")).toFixed(1) + " s", at("follower", "ttc_s") < 2], ["Follower speed", at("follower", "speed_mps").toFixed(1) + " m/s"], ["Verifier", at("verifier", "alert") > 0 ? "ALERT" : "quiet", at("verifier", "alert") > 0]],
      charts: [{ t: "Gap to leader (m): true vs claimed in BSM", s: [[col("follower", "gap_m"), C.blue, "true (ranging)"], [claimedGap, C.red, "claimed (BSM)"]], ymin: -5 },
               { t: "Speed (m/s)", s: [[col("leader_true", "speed_mps"), C.grey, "leader"], [col("follower", "speed_mps"), C.blue, "follower"]] },
               { t: "Physics residuals", s: [[col("verifier", "kin_resid"), C.orange, "kinematic"], [col("verifier", "gap_resid"), C.purple, "V2X vs ranging gap"]], ymax: 8 }] };
  },
  ev: () => ({ readouts: [["Feeder V", at("feeder", "V_pu").toFixed(3) + " pu", at("feeder", "V_pu") < 0.97], ["Frequency", at("feeder", "f_hz").toFixed(2) + " Hz", Math.abs(at("feeder", "f_hz") - 50) > 1], ["Load (true)", at("feeder", "P_kw").toFixed(1) + " kW"], ["Load (reported)", at("feeder", "P_reported_kw").toFixed(1) + " kW"], ["Verifier", at("verifier", "alert") > 0 ? "ALERT" : "quiet", at("verifier", "alert") > 0]],
    charts: [{ t: "Feeder load (kW): true vs MeterValues", s: [[col("feeder", "P_kw"), C.red, "true"], [col("feeder", "P_reported_kw"), C.blue, "reported"]] },
             { t: "Feeder voltage (pu)", s: [[col("feeder", "V_pu"), C.blue, "V"]] }, { t: "Island frequency (Hz)", s: [[col("feeder", "f_hz"), C.green, "f"]] }] }),
  av: () => ({ readouts: [["Ego speed", at("ego", "speed_mps").toFixed(1) + " m/s"], ["Brake request", at("can", "accel_req").toFixed(1) + " m/s²", at("can", "accel_req") < -2], ["Follower TTC", Math.min(99, at("follower", "ttc_s")).toFixed(1) + " s", at("follower", "ttc_s") < 3], ["Verifier", at("verifier", "alert") > 0 ? "ALERT" : "quiet", at("verifier", "alert") > 0]],
    charts: [{ t: "Acceleration (m/s²): planned vs requested on CAN", s: [[col("can", "accel_plan"), C.grey, "plan"], [col("can", "accel_req"), C.red, "bus request"]] },
             { t: "Ego speed (m/s)", s: [[col("can", "speed_plan_mps"), C.grey, "no attack"], [col("ego", "speed_mps"), C.blue, "actual"]] }, { t: "Follower TTC (s)", s: [[col("follower", "ttc_s"), C.purple, "TTC"]], ymax: 15 }] }),
  ctf: () => ({ readouts: [["Charger current", at("session", "current_a").toFixed(1) + " A", at("session", "current_a") > at("session", "pilot_limit_a") * 0.99 && S.t > 15000], ["Pilot limit", at("session", "pilot_limit_a").toFixed(0) + " A"], ["Feeder V", at("feeder", "V_pu").toFixed(3) + " pu"]],
    charts: [{ t: "Session current (A) vs pilot limit", s: [[col("session", "current_a"), C.blue, "current"], [col("session", "pilot_limit_a"), C.red, "pilot limit"]] }, { t: "Session power (kW)", s: [[col("session", "power_kw"), C.green, "P"]] }] }),
};

function drawCharts(cfg) {
  const cv = $("charts"), g = cv.getContext("2d"), W = cv.width, H = cv.height; g.clearRect(0, 0, W, H); g.font = "19px Segoe UI, sans-serif";
  const n = cfg.charts.length, h = H / n, x0 = 66, x1 = W - 10, N = S.nFrames + 1, k = frameAt(S.t);
  cfg.charts.forEach((c, i) => {
    const y0 = i * h + 34, y1 = (i + 1) * h - 16; let lo = c.ymin ?? Infinity, hi = c.ymax ?? -Infinity;
    c.s.forEach(([a]) => { for (let j = 0; j < a.length; j += 3) { if (c.ymin === undefined) lo = Math.min(lo, a[j]); if (c.ymax === undefined) hi = Math.max(hi, a[j]); } });
    if (hi - lo < 1e-9) { hi = lo + 1; } const X = (j) => x0 + (x1 - x0) * j / (N - 1), Y = (v) => y1 - (y1 - y0) * (Math.min(v, hi) - lo) / (hi - lo);
    g.fillStyle = C.ink; g.fillText(c.t, x0, y0 - 12); g.strokeStyle = C.line; g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    g.fillStyle = C.grey; const dg = (hi - lo) < 0.5 ? 3 : (hi - lo) < 5 ? 2 : hi > 100 ? 0 : 1; g.fillText(hi.toFixed(dg), 2, y0 + 14); g.fillText(lo.toFixed(dg), 2, y1);
    c.s.forEach(([a, colr, name], si) => { g.strokeStyle = colr; g.lineWidth = 1.6; g.beginPath(); for (let j = 0; j < a.length; j++) { j ? g.lineTo(X(j), Y(a[j])) : g.moveTo(X(j), Y(a[j])); } g.stroke(); g.fillStyle = colr; g.fillText("■ " + name, x1 - 210, y0 + 20 + si * 22); });
    const o = S.path.outcome; [[o.attack_ms ?? (o.attack_s ? o.attack_s * 1000 / o.time_scale : null), C.red, "attack"], [S.sc.id.startsWith("track4") ? o.detect_ms : (o.physics_alert_ms ?? (o.physics_alert_s ? o.physics_alert_s * 1000 / o.time_scale : null)), C.orange, S.sc.id.startsWith("track4") ? "IDS alert" : "alert"]].forEach(([t, cl, lab]) => {
      if (t == null) return; const j = Math.floor(t * S.sc.fps / 1000); g.strokeStyle = cl; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(X(j), y0); g.lineTo(X(j), y1); g.stroke(); g.setLineDash([]); g.fillStyle = cl; g.fillText(lab, X(j) + 3, y1 - 3); });
    g.strokeStyle = C.ink; g.lineWidth = 1; g.beginPath(); g.moveTo(X(k), y0); g.lineTo(X(k), y1); g.stroke();
  });
  $("readouts").innerHTML = cfg.readouts.map(([a, b, bad]) => `<div class="readout ${bad ? "alarm" : ""}"><small>${a}</small><b>${b}</b></div>`).join("");
}

/* ------------------------------------------------------------------ physical POV */
function drawCav(g, W, H) {
  const fy = at("follower", "y_m"), ty = at("leader_true", "y_m"), cy = at("leader_claimed", "y_m"), ll = S.path.outcome.lead_len_m, sc = W / 120, ox = 120;
  const X = (y) => ox + (y - fy) * sc, laneY = H / 2;
  g.fillStyle = "#e9edf1"; g.fillRect(0, laneY - 40, W, 80); g.strokeStyle = "#fff"; g.setLineDash([22, 18]); g.lineWidth = 3; g.beginPath(); g.moveTo(0, laneY); g.lineTo(W, laneY); g.stroke(); g.setLineDash([]);
  const car = (y, len, color, label, dashed, yOff, below) => { const x = X(y) - len * sc, w = len * sc, hh = 26; g.fillStyle = dashed ? "rgba(213,94,0,.15)" : color; g.strokeStyle = color; g.lineWidth = 2; if (dashed) g.setLineDash([6, 4]); g.fillRect(x, laneY + yOff - hh / 2, w, hh); g.strokeRect(x, laneY + yOff - hh / 2, w, hh); g.setLineDash([]); g.fillStyle = C.ink; g.font = "20px Segoe UI"; g.fillText(label, x, laneY + yOff + (below ? 44 : -22)); };
  car(fy, 4.5, C.blue, "follower (CACC)", false, 0); car(ty, ll, C.grey, "leader (true)", false, 0, true);
  if (Math.abs(cy - ty) > 0.5) car(cy, ll, C.red, "leader as claimed in BSM", true, -52);
  const gap = at("follower", "gap_m"); g.fillStyle = gap < 3 ? C.red : C.ink; g.font = "bold 24px Segoe UI"; g.fillText(`true gap ${gap.toFixed(1)} m`, 20, 30);
  if (gap <= 0) { g.fillStyle = C.red; g.font = "bold 44px Segoe UI"; g.fillText("COLLISION", W / 2 - 70, 70); }
}
function drawEv(g, W, H) {
  const n = Object.keys(S.tl.streams.chargers).filter((k) => /^I_\d+$/.test(k)).length, rated = at("chargers", "rated_a"), bw = (W - 60) / n; g.font = "18px Segoe UI";
  const Y = (a) => H - 50 - (H - 110) * Math.min(a, 40) / 40; g.strokeStyle = C.red; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(30, Y(rated)); g.lineTo(W - 30, Y(rated)); g.stroke(); g.setLineDash([]); g.fillStyle = C.red; g.fillText(`rated ${rated.toFixed(0)} A`, 34, Y(rated) - 4);
  for (let i = 1; i <= n; i++) { const x = 30 + (i - 1) * bw + 6, I = at("chargers", "I_" + i), R = at("chargers", "Irep_" + i), over = I > rated * 1.05;
    g.fillStyle = over ? C.red : C.blue; g.fillRect(x, Y(I), bw - 16, H - 50 - Y(I)); g.strokeStyle = C.ink; g.lineWidth = 2; g.strokeRect(x - 2, Y(R), bw - 12, H - 50 - Y(R)); g.fillStyle = C.ink; g.fillText("EVSE " + i, x, H - 32); g.fillText(I.toFixed(0) + " A", x, Y(I) - 5); }
  g.fillStyle = C.ink; g.font = "18px Segoe UI"; g.fillText("filled = true current, outline = MeterValues-reported current", 30, 20);
}
async function fetchLidar(k) { if (S.lidar[k] || S.lidarBusy[k]) return; S.lidarBusy[k] = 1; try { S.lidar[k] = await api(`/api/tracks/track3_av_can/lidar/${k}`); } catch (e) { S.lidar[k] = { err: e.message }; } }
function drawAv(g, W, H) {
  const k = Math.min(Math.floor(S.t / 100), 107); fetchLidar(k); const L = S.lidar[k] || S.lidar[k - 1] || S.lidar[Math.max(0, k - 2)]; const s = H / 60, cx = W / 2, cy = H - 40;
  g.fillStyle = "#101820"; g.fillRect(0, 0, W, H);
  if (L && L.err) { g.fillStyle = "#fff"; g.font = "20px Segoe UI"; g.fillText("LiDAR unavailable: " + L.err, 20, 40); }
  if (L && L.points) { g.fillStyle = "#7fd3ff"; L.points.forEach(([x, y]) => g.fillRect(cx - y * s, cy - x * s, 2, 2)); g.strokeStyle = C.orange; g.lineWidth = 2; (L.boxes || []).forEach((b) => g.strokeRect(cx - b.cy * s - 14, cy - b.cx * s - 14, 28, 28)); }
  g.fillStyle = C.green; g.fillRect(cx - 9, cy - 4.5 * s * 0.5, 18, 4.5 * s); g.fillStyle = "#fff"; g.font = "18px Segoe UI"; g.fillText("ego", cx + 14, cy);
  const gap = at("follower", "gap_m"); g.fillStyle = C.orange; g.fillRect(cx - 9, cy + 6, 18, 4.5 * s * 0.6); g.fillText(`follower ${gap.toFixed(1)} m behind`, cx + 14, cy + 40);
  g.fillStyle = at("can", "accel_req") < -2 ? "#ff8a5c" : "#fff"; g.font = "bold 24px Segoe UI"; g.fillText(`CAN brake request ${at("can", "accel_req").toFixed(1)} m/s²`, 16, 28);
}
function syncVideo() { const v = $("ctfVideo"); if (!v.duration) return; const want = (S.t / 1000) % v.duration; if (!S.playing && Math.abs(v.currentTime - want) > 0.05) v.currentTime = want; }

function draw() {
  if (!S.tl) return;
  const cv = $("pov"), g = cv.getContext("2d"), W = cv.width, H = cv.height; g.clearRect(0, 0, W, H); g.lineWidth = 1;
  const d = S.sc.domain; if (d === "cav") drawCav(g, W, H); else if (d === "ev") drawEv(g, W, H); else if (d === "av") drawAv(g, W, H); else syncVideo();
  drawCharts(CHARTS[d]()); updateSteps();
  const k = frameAt(S.t); $("clock").textContent = `${(S.t / 1000).toFixed(3)} s · frame ${k} · ${S.tl.t_ms[Math.min(k, S.tl.t_ms.length - 1)]} ms`; $("scrub").value = S.t;
}

/* ------------------------------------------------------------------ CTF console */
$("btnExtract").onclick = async () => { $("extracted").textContent = "decoding..."; const r = await api("/api/ctf/extract", { method: "POST" }); $("extracted").textContent = r.decode.ok ? r.decode.digest_hex : "decode failed"; $("hashIn").value = r.decode.ok ? r.decode.digest_hex : ""; };
$("btnVerify").onclick = async () => { const r = await api("/api/ctf/verify", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ hash: $("hashIn").value }) }); $("ctfMsg").textContent = r.message; };
$("btnFlag").onclick = async () => { const r = await api("/api/ctf/submit", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ flag: $("flagIn").value }) }); $("ctfMsg").textContent = r.message; };

boot();
window.__twin = S;   // exposed for headless tests
