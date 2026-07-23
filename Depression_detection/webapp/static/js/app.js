/* ═══════════════════════════════════════════════════════════
   NeuroSense — Dashboard JavaScript (Chart.js + Plotly)
════════════════════════════════════════════════════════════ */

const API = "";   // same origin

/* ── Chart.js global defaults ─────────────────────────────── */
Chart.defaults.color = "#8892b0";
Chart.defaults.borderColor = "rgba(255,255,255,.08)";
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.plugins.legend.labels.boxWidth = 12;

const PALETTE = ["#6c63ff", "#00d4aa", "#ff6b6b", "#ffd93d", "#4da6ff", "#b17cf4", "#ff9f43", "#2ed573"];
const _charts = {};   // canvas id → Chart instance

function destroyChart(id) {
  if (_charts[id]) { _charts[id].destroy(); delete _charts[id]; }
}
function mkChart(id, cfg) {
  destroyChart(id);
  const ctx = document.getElementById(id);
  if (!ctx) return null;
  _charts[id] = new Chart(ctx, cfg);
  return _charts[id];
}

/* ── Sidebar / tabs ──────────────────────────────────────── */
const TITLES = {
  dashboard: "Dashboard", eda: "Data Analysis",
  regression: "Regression Models", classification: "Classification",
  predict: "Live Prediction", explainability: "Explainability"
};

document.querySelectorAll(".nav-item").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    const tab = btn.dataset.tab;
    document.getElementById(`tab-${tab}`).classList.add("active");
    document.getElementById("pageTitle").textContent = TITLES[tab];
    if (tab === "dashboard") loadDashboard();
    if (tab === "eda") loadEDA();
    if (tab === "regression") loadRegression();
    if (tab === "classification") loadClassification();
    if (tab === "explainability") { /* triggered by button */ }
  });
});

function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("open");
}

/* ── API health check ────────────────────────────────────── */
async function checkHealth() {
  const dot = document.getElementById("apiStatus");
  const txt = document.getElementById("apiStatusText");
  try {
    const r = await fetch(`${API}/api/health`);
    const j = await r.json();
    dot.className = "status-dot ok";
    txt.textContent = "API Online";
  } catch {
    dot.className = "status-dot err";
    txt.textContent = "API Offline";
  }
}

/* ═══════════════════════════════════════════════════════════
   DASHBOARD
════════════════════════════════════════════════════════════ */
async function loadDashboard() {
  try {
    const [eda, best] = await Promise.all([
      fetch(`${API}/api/analysis/eda`).then(r => r.json()),
      fetch(`${API}/api/analysis/best-model`).then(r => r.json()),
    ]);

    const ds = eda.dataset_summary;
    document.getElementById("kpiTotal").textContent = ds.total_samples;
    document.getElementById("kpiDep").textContent =
      ds.depressed_train + ds.depressed_dev;
    document.getElementById("kpiNot").textContent =
      ds.normal_train + ds.normal_dev;
    document.getElementById("kpiBestClf").textContent = best.best_classification.name.split(" ")[0];
    document.getElementById("kpiBestReg").textContent = best.best_regression.name.split(" ")[0];

    /* Class doughnut */
    const cd = eda.class_distribution;
    const totDep = cd.train_counts[1] + cd.dev_counts[1];
    const totNot = cd.train_counts[0] + cd.dev_counts[0];
    mkChart("dashClassChart", {
      type: "doughnut",
      data: {
        labels: ["Non-Depressed", "Depressed"],
        datasets: [{
          data: [totNot, totDep],
          backgroundColor: ["#00d4aa", "#ff6b6b"],
          borderWidth: 0, hoverOffset: 8
        }]
      },
      options: { plugins: { legend: { position: "bottom" } }, cutout: "65%" }
    });

    /* PCA scatter */
    const pca = eda.pca;
    Plotly.newPlot("dashPcaChart", [
      {
        x: pca.x_not, y: pca.y_not, mode: "markers", name: "Non-Depressed",
        marker: { color: "#00d4aa", size: 7, opacity: .7 }
      },
      {
        x: pca.x_dep, y: pca.y_dep, mode: "markers", name: "Depressed",
        marker: { color: "#ff6b6b", size: 7, opacity: .7 }
      },
    ], {
      plot_bgcolor: "transparent", paper_bgcolor: "transparent",
      font: { color: "#8892b0", size: 11 },
      margin: { t: 10, b: 40, l: 40, r: 10 },
      xaxis: {
        title: `PC1 (${(pca.variance_explained[0] * 100).toFixed(1)}%)`,
        gridcolor: "rgba(255,255,255,.07)"
      },
      yaxis: {
        title: `PC2 (${(pca.variance_explained[1] * 100).toFixed(1)}%)`,
        gridcolor: "rgba(255,255,255,.07)"
      },
      legend: { orientation: "h", y: -0.2 },
    }, { responsive: true, displayModeBar: false });

  } catch (e) { console.error("Dashboard load error:", e); }
}

/* ═══════════════════════════════════════════════════════════
   EDA
════════════════════════════════════════════════════════════ */
async function loadEDA() {
  const loading = document.getElementById("edaLoading");
  const content = document.getElementById("edaContent");
  loading.style.display = "flex"; content.style.display = "none";

  try {
    const eda = await fetch(`${API}/api/analysis/eda`).then(r => r.json());
    loading.style.display = "none"; content.style.display = "block";

    /* Modality stats table */
    const ms = eda.modality_stats;
    const dims = eda.dataset_summary.feature_dims;
    const tbody = document.getElementById("modalityStatsBody");
    tbody.innerHTML = "";
    for (const [mod, st] of Object.entries(ms)) {
      const r = document.createElement("tr");
      const dim = dims[mod] || "—";
      r.innerHTML = `<td>${mod.charAt(0).toUpperCase() + mod.slice(1)}</td>
        <td>${dim}</td><td>${st.mean}</td><td>${st.std}</td>
        <td>${st.min}</td><td>${st.max}</td>`;
      tbody.appendChild(r);
    }

    /* Class distribution bar */
    const cd = eda.class_distribution;
    mkChart("edaClassChart", {
      type: "bar",
      data: {
        labels: ["Non-Depressed", "Depressed"],
        datasets: [
          { label: "Train", data: cd.train_counts, backgroundColor: "#6c63ff", borderRadius: 6 },
          { label: "Dev", data: cd.dev_counts, backgroundColor: "#00d4aa", borderRadius: 6 },
        ]
      },
      options: {
        plugins: { legend: { position: "bottom" } },
        scales: { y: { beginAtZero: true, grid: { color: "rgba(255,255,255,.07)" } } }
      }
    });

    /* Feature norm histograms */
    const fn = eda.feature_norms;
    const binData = (arr, bins = 20) => {
      const mn = Math.min(...arr), mx = Math.max(...arr);
      const step = (mx - mn) / bins;
      const counts = new Array(bins).fill(0);
      const labels = [];
      for (let b = 0; b < bins; b++) labels.push((mn + b * step).toFixed(2));
      arr.forEach(v => {
        let b = Math.floor((v - mn) / step);
        if (b === bins) b = bins - 1;
        counts[b]++;
      });
      return { labels, counts };
    };
    ["audio", "image", "text"].forEach((m, idx) => {
      const { labels, counts } = binData(fn[m]);
      mkChart(`eda${m.charAt(0).toUpperCase() + m.slice(1)}Chart`, {
        type: "bar",
        data: {
          labels, datasets: [{
            label: `${m} norms`, data: counts,
            backgroundColor: PALETTE[idx] + "99", borderColor: PALETTE[idx], borderWidth: 1
          }]
        },
        options: {
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { display: false }, ticks: { maxTicksLimit: 6 } },
            y: { beginAtZero: true, grid: { color: "rgba(255,255,255,.07)" } }
          }
        }
      });
    });

    /* PCA */
    const pca = eda.pca;
    Plotly.newPlot("edaPcaChart", [
      {
        x: pca.x_not, y: pca.y_not, mode: "markers", name: "Non-Depressed",
        marker: { color: "#00d4aa", size: 8, opacity: .75 }
      },
      {
        x: pca.x_dep, y: pca.y_dep, mode: "markers", name: "Depressed",
        marker: { color: "#ff6b6b", size: 8, opacity: .75 }
      },
    ], {
      plot_bgcolor: "transparent", paper_bgcolor: "transparent",
      font: { color: "#8892b0", size: 11 }, margin: { t: 10, b: 50, l: 50, r: 10 },
      xaxis: { title: `PC1 (${(pca.variance_explained[0] * 100).toFixed(1)}%)`, gridcolor: "rgba(255,255,255,.06)" },
      yaxis: { title: `PC2 (${(pca.variance_explained[1] * 100).toFixed(1)}%)`, gridcolor: "rgba(255,255,255,.06)" },
      legend: { orientation: "h", y: -0.18 },
    }, { responsive: true, displayModeBar: false });

  } catch (e) {
    loading.style.display = "none";
    console.error("EDA load error:", e);
  }
}

/* ═══════════════════════════════════════════════════════════
   REGRESSION
════════════════════════════════════════════════════════════ */
async function loadRegression() {
  const loading = document.getElementById("regLoading");
  const content = document.getElementById("regContent");
  loading.style.display = "flex"; content.style.display = "none";

  try {
    const data = await fetch(`${API}/api/analysis/regression`).then(r => r.json());
    loading.style.display = "none"; content.style.display = "block";

    const models = data.models;
    const best = data.best_model;
    const names = Object.keys(models);

    /* Banner */
    document.getElementById("regBestBanner").innerHTML =
      `🏆 Best Regression Model: <strong>${best}</strong> &nbsp;|&nbsp; R² = ${models[best].r2}`;

    /* R2 bar chart */
    mkChart("regR2Chart", {
      type: "bar",
      data: {
        labels: names,
        datasets: [{
          label: "R² Score", data: names.map(n => models[n].r2),
          backgroundColor: names.map((n, i) => n === best ? "#6c63ff" : PALETTE[i % PALETTE.length] + "99"),
          borderRadius: 6
        }]
      },
      options: {
        indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: {
          x: {
            min: Math.min(...names.map(n => models[n].r2)) - 0.05, max: Math.max(...names.map(n => models[n].r2)) + 0.05,
            grid: { color: "rgba(255,255,255,.07)" }
          },
          y: { grid: { display: false } }
        }
      }
    });

    /* Error chart */
    mkChart("regErrorChart", {
      type: "bar",
      data: {
        labels: names,
        datasets: [
          { label: "MAE", data: names.map(n => models[n].mae), backgroundColor: "#ff6b6b99", borderRadius: 4 },
          { label: "RMSE", data: names.map(n => models[n].rmse), backgroundColor: "#ffd93d99", borderRadius: 4 },
        ]
      },
      options: {
        plugins: { legend: { position: "bottom" } },
        scales: {
          y: { beginAtZero: true, grid: { color: "rgba(255,255,255,.07)" } },
          x: { grid: { display: false } }
        }
      }
    });

    /* Table */
    const tbody = document.getElementById("regTableBody");
    tbody.innerHTML = "";
    names.forEach(n => {
      const m = models[n];
      const row = document.createElement("tr");
      if (n === best) row.classList.add("best-row");
      row.innerHTML = `<td>${n}${n === best ? " 🏆" : ""}</td>
        <td>${m.r2}</td><td>${m.mae}</td><td>${m.mse}</td><td>${m.rmse}</td>
        <td>${(m.modal_importance.Audio * 100).toFixed(1)}%</td>
        <td>${(m.modal_importance.Image * 100).toFixed(1)}%</td>
        <td>${(m.modal_importance.Text * 100).toFixed(1)}%</td>`;
      tbody.appendChild(row);
    });

    /* Actual vs Predicted (best model) */
    const bm = models[best];
    const rng = [Math.min(...bm.y_true) - 1, Math.max(...bm.y_true) + 1];
    Plotly.newPlot("regActPredChart", [
      {
        x: bm.y_true, y: bm.y_pred, mode: "markers", name: "Samples",
        marker: {
          color: "#6c63ff", size: 9, opacity: .75,
          line: { color: "#fff", width: 0.8 }
        }
      },
      { x: rng, y: rng, mode: "lines", name: "Perfect", line: { color: "#00d4aa", dash: "dash", width: 1.5 } },
    ], [{
      plot_bgcolor: "transparent", paper_bgcolor: "transparent",
      font: { color: "#8892b0", size: 12 }, margin: { t: 10, b: 50, l: 50, r: 10 },
      xaxis: { title: "Actual PHQ-8", gridcolor: "rgba(255,255,255,.06)" },
      yaxis: { title: "Predicted PHQ-8", gridcolor: "rgba(255,255,255,.06)" },
      legend: { orientation: "h", y: -0.2 },
    }], { responsive: true, displayModeBar: false });

    /* Residual plot */
    mkChart("regResidChart", {
      type: "bar",
      data: {
        labels: bm.y_true.map((_, i) => `S${i + 1}`),
        datasets: [{
          label: "Residual",
          data: bm.residuals,
          backgroundColor: bm.residuals.map(r => r >= 0 ? "rgba(108,99,255,.7)" : "rgba(255,107,107,.7)"),
          borderRadius: 3,
        }]
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          y: {
            grid: { color: "rgba(255,255,255,.07)" },
            title: { display: true, text: "Residual (actual - predicted)" }
          },
          x: { grid: { display: false }, ticks: { maxTicksLimit: 20 } }
        }
      }
    });

  } catch (e) {
    loading.style.display = "none";
    console.error("Regression load error:", e);
  }
}

/* ═══════════════════════════════════════════════════════════
   CLASSIFICATION
════════════════════════════════════════════════════════════ */
async function loadClassification() {
  const loading = document.getElementById("clfLoading");
  const content = document.getElementById("clfContent");
  loading.style.display = "flex"; content.style.display = "none";

  try {
    const data = await fetch(`${API}/api/analysis/classification`).then(r => r.json());
    loading.style.display = "none"; content.style.display = "block";

    const models = data.models;
    const best = data.best_model;
    const names = Object.keys(models);

    document.getElementById("clfBestBanner").innerHTML =
      `🏆 Best Classifier: <strong>${best}</strong> &nbsp;|&nbsp;` +
      ` F1 = ${data.best_f1} &nbsp;|&nbsp; AUC = ${data.best_auc}`;

    /* F1 bar */
    mkChart("clfF1Chart", {
      type: "bar",
      data: {
        labels: names, datasets: [{
          label: "F1 Score",
          data: names.map(n => models[n].f1),
          backgroundColor: names.map((n, i) => n === best ? "#6c63ff" : PALETTE[i % PALETTE.length] + "99"),
          borderRadius: 6
        }]
      },
      options: {
        indexAxis: "y", plugins: { legend: { display: false } },
        scales: { x: { min: 0, max: 1, grid: { color: "rgba(255,255,255,.07)" } }, y: { grid: { display: false } } }
      }
    });

    /* AUC bar */
    mkChart("clfAucChart", {
      type: "bar",
      data: {
        labels: names, datasets: [{
          label: "AUC-ROC",
          data: names.map(n => models[n].auc_roc),
          backgroundColor: names.map((n, i) => n === best ? "#00d4aa" : PALETTE[i % PALETTE.length] + "99"),
          borderRadius: 6
        }]
      },
      options: {
        indexAxis: "y", plugins: { legend: { display: false } },
        scales: { x: { min: 0, max: 1, grid: { color: "rgba(255,255,255,.07)" } }, y: { grid: { display: false } } }
      }
    });

    /* Metrics table */
    const tbody = document.getElementById("clfTableBody");
    tbody.innerHTML = "";
    names.forEach(n => {
      const m = models[n];
      const row = document.createElement("tr");
      if (n === best) row.classList.add("best-row");
      row.innerHTML = `<td>${n}${n === best ? " 🏆" : ""}</td>
        <td>${(m.accuracy * 100).toFixed(1)}%</td>
        <td>${(m.precision * 100).toFixed(1)}%</td>
        <td>${(m.recall * 100).toFixed(1)}%</td>
        <td>${m.f1}</td><td>${m.auc_roc}</td>`;
      tbody.appendChild(row);
    });

    /* ROC curves */
    const rocDatasets = names.map((n, i) => ({
      label: `${n} (AUC=${models[n].auc_roc})`,
      data: models[n].roc_fpr.map((x, j) => ({ x, y: models[n].roc_tpr[j] })),
      borderColor: PALETTE[i % PALETTE.length],
      backgroundColor: "transparent",
      borderWidth: 2, pointRadius: 0, tension: 0.3
    }));
    rocDatasets.push({
      label: "Random",
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: "#4a5568", borderDash: [6, 3], borderWidth: 1.5,
      backgroundColor: "transparent", pointRadius: 0
    });
    mkChart("clfRocChart", {
      type: "line",
      data: { datasets: rocDatasets },
      options: {
        scales: {
          x: { type: "linear", min: 0, max: 1, title: { display: true, text: "FPR" }, grid: { color: "rgba(255,255,255,.06)" } },
          y: { min: 0, max: 1, title: { display: true, text: "TPR" }, grid: { color: "rgba(255,255,255,.06)" } }
        },
        plugins: { legend: { position: "bottom" } }
      }
    });

    /* Confusion matrix (Plotly heatmap) */
    const cm = models[best].confusion_matrix;
    Plotly.newPlot("clfCmChart", [{
      z: cm, x: ["Pred: Not", "Pred: Dep"], y: ["True: Not", "True: Dep"],
      type: "heatmap", colorscale: [["0", "#111420"], ["1", "#6c63ff"]],
      text: cm.map(row => row.map(v => `${v}`)),
      texttemplate: "%{text}", textfont: { size: 22, color: "white" }
    }], {
      plot_bgcolor: "transparent", paper_bgcolor: "transparent",
      font: { color: "#8892b0", size: 12 }, margin: { t: 10, b: 50, l: 70, r: 10 },
      xaxis: { gridcolor: "transparent" }, yaxis: { gridcolor: "transparent" },
    }, { responsive: true, displayModeBar: false });

    /* Modal importance doughnut */
    const mi = models[best].modal_importance;
    mkChart("clfModalChart", {
      type: "doughnut",
      data: {
        labels: ["Audio", "Image", "Text (BERT)"],
        datasets: [{
          data: [mi.Audio, mi.Image, mi.Text],
          backgroundColor: ["#6c63ff", "#ff6b6b", "#00d4aa"], borderWidth: 0, hoverOffset: 6
        }]
      },
      options: { plugins: { legend: { position: "bottom" } }, cutout: "60%" }
    });

  } catch (e) {
    loading.style.display = "none";
    console.error("Classification load error:", e);
  }
}

/* ═════════════════════════════════════════════════════════
   LIVE PREDICTION — MODALITY SWITCHER
═════════════════════════════════════════════════════════ */
let _currentModality = null;
let _audioFile = null;
let _audioFileMulti = null;
let _imageFileSingle = null;
let _imageFileMulti = null;
let _capturedImageBlob = null;  // from camera

// Voice recording state
let _mediaRecorder = null;
let _recordChunks = [];
let _recordStream = null;
let _recordInterval = null;
let _recordSeconds = 0;
// Camera state
let _cameraStream = null;

/* ─ selectModality ─────────────────────────────────────────── */
function selectModality(modality) {
  _currentModality = modality;
  document.querySelectorAll(".mod-btn").forEach(b => b.classList.remove("active"));
  const btn = document.getElementById(`modBtn-${modality}`);
  if (btn) btn.classList.add("active");
  document.querySelectorAll(".modality-panel").forEach(p => p.classList.remove("active"));
  const panel = document.getElementById(`panel-${modality}`);
  if (panel) panel.classList.add("active");
  document.getElementById("noModalityMsg").style.display = "none";
  document.getElementById("predictBtnRow").style.display = "flex";
  document.getElementById("resultCard").style.display = "none";
  stopCamera(); stopRecording();
}

/* ─ File handlers ─────────────────────────────────────────────── */
function handleAudioFile(input) {
  _audioFile = input.files[0];
  if (!_audioFile) return;
  _showChip("audioChip", `🎵 ${_audioFile.name}`);
  _styleZone("audioZone", "#6c63ff");
  _showAudioPlayer(_audioFile);
  // Reset transcript
  document.getElementById("transcriptBox").className = "transcript-box";
}
function handleImageFileSingle(input) {
  _imageFileSingle = input.files[0];
  _capturedImageBlob = null;
  if (!_imageFileSingle) return;
  _showChip("imageChip-single", `🖼️ ${_imageFileSingle.name}`);
  _styleZone("imageZone-single", "#ff6b6b");
  _showImagePreview(_imageFileSingle);
}
function handleAudioFileMulti(input) {
  _audioFileMulti = input.files[0];
  if (!_audioFileMulti) return;
  _showChip("audioChip-multi", `🎵 ${_audioFileMulti.name}`);
  _styleZone("audioZone-multi", "#6c63ff");
}
function handleImageFileMulti(input) {
  _imageFileMulti = input.files[0];
  if (!_imageFileMulti) return;
  _showChip("imageChip-multi", `🖼️ ${_imageFileMulti.name}`);
  _styleZone("imageZone-multi", "#ff6b6b");
}

/* ─ Drag-drop handlers ────────────────────────────────────────── */
function handleAudioDrop(e) {
  e.preventDefault();
  document.getElementById("audioZone").classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (!file) return;
  _audioFile = file;
  _showChip("audioChip", `🎵 ${file.name}`);
  _styleZone("audioZone", "#6c63ff");
  _showAudioPlayer(file);
}
function handleImageDrop(e) {
  e.preventDefault();
  document.getElementById("imageZone-single").classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (!file) return;
  _imageFileSingle = file;
  _capturedImageBlob = null;
  _showChip("imageChip-single", `🖼️ ${file.name}`);
  _styleZone("imageZone-single", "#ff6b6b");
  _showImagePreview(file);
}
function handleAudioDropMulti(e) {
  e.preventDefault();
  document.getElementById("audioZone-multi").classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (!file) return;
  _audioFileMulti = file;
  _showChip("audioChip-multi", `🎵 ${file.name}`);
  _styleZone("audioZone-multi", "#6c63ff");
}
function handleImageDropMulti(e) {
  e.preventDefault();
  document.getElementById("imageZone-multi").classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (!file) return;
  _imageFileMulti = file;
  _showChip("imageChip-multi", `🖼️ ${file.name}`);
  _styleZone("imageZone-multi", "#ff6b6b");
}

/* ─ UI helpers ──────────────────────────────────────────────────── */
function _showChip(id, text) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.classList.remove("hidden");
}
function _styleZone(id, color) {
  const el = document.getElementById(id);
  if (el) el.style.borderColor = color;
}
function _showAudioPlayer(file) {
  const player = document.getElementById("audioPlayer");
  if (!player) return;
  player.src = URL.createObjectURL(file);
  player.style.display = "block";
}
function _showImagePreview(file) {
  const img = document.getElementById("capturedPhoto");
  if (!img) return;
  img.src = URL.createObjectURL(file);
  img.style.display = "block";
}

/* ─ Voice recording ────────────────────────────────────────────── */
async function toggleRecording() {
  if (_mediaRecorder && _mediaRecorder.state === "recording") {
    stopRecording();
  } else {
    startRecording();
  }
}

async function startRecording() {
  try {
    _recordStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    alert("Microphone access denied. Please allow microphone permissions.");
    return;
  }

  _recordChunks = [];
  _mediaRecorder = new MediaRecorder(_recordStream, { mimeType: "audio/webm" });
  _mediaRecorder.ondataavailable = e => { if (e.data.size > 0) _recordChunks.push(e.data); };
  _mediaRecorder.onstop = _onRecordingStop;
  _mediaRecorder.start(100);

  // UI
  const btn = document.getElementById("recordBtn");
  if (btn) { btn.classList.add("recording"); }
  const txt = document.getElementById("recordBtnText");
  if (txt) txt.textContent = "Stop Recording";
  const viz = document.getElementById("audioViz");
  if (viz) viz.classList.add("show");

  // Timer
  _recordSeconds = 0;
  _recordInterval = setInterval(() => {
    _recordSeconds++;
    const timer = document.getElementById("recordTimer");
    if (timer) {
      const m = String(Math.floor(_recordSeconds / 60)).padStart(2, "0");
      const s = String(_recordSeconds % 60).padStart(2, "0");
      timer.textContent = `🔴 ${m}:${s}`;
    }
    // Draw waveform
    _drawWaveformDots();
  }, 1000);
}

function stopRecording() {
  if (!_mediaRecorder || _mediaRecorder.state === "inactive") return;
  _mediaRecorder.stop();
  if (_recordStream) _recordStream.getTracks().forEach(t => t.stop());
  clearInterval(_recordInterval);
  const btn = document.getElementById("recordBtn");
  if (btn) btn.classList.remove("recording");
  const txt = document.getElementById("recordBtnText");
  if (txt) txt.textContent = "Record Voice";
  const timer = document.getElementById("recordTimer");
  if (timer) timer.textContent = "";
  const viz = document.getElementById("audioViz");
  if (viz) viz.classList.remove("show");
}

function _onRecordingStop() {
  const blob = new Blob(_recordChunks, { type: "audio/webm" });
  const filename = `recording_${Date.now()}.webm`;
  _audioFile = new File([blob], filename, { type: "audio/webm" });
  _showChip("audioChip", `🎵 ${filename} (recorded)`);
  _styleZone("audioZone", "#6c63ff");
  const player = document.getElementById("audioPlayer");
  if (player) { player.src = URL.createObjectURL(blob); player.style.display = "block"; }
  document.getElementById("audioZoneText").textContent = `✓ Recorded: ${filename}`;
  document.getElementById("transcriptBox").className = "transcript-box";
}

function _drawWaveformDots() {
  const canvas = document.getElementById("audioVizCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#6c63ff";
  for (let i = 0; i < 40; i++) {
    const h = Math.random() * canvas.height * 0.8;
    ctx.fillRect(i * (canvas.width / 40), (canvas.height - h) / 2, 4, h);
  }
}

/* ─ Camera capture ─────────────────────────────────────────────── */
async function toggleCamera() {
  if (_cameraStream) {
    stopCamera();
  } else {
    await openCamera();
  }
}

async function openCamera() {
  try {
    _cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    const video = document.getElementById("cameraPreview");
    if (video) { video.srcObject = _cameraStream; video.style.display = "block"; }
    const btn = document.getElementById("cameraBtnText");
    if (btn) btn.textContent = "Close Camera";
    const cap = document.getElementById("captureBtn");
    if (cap) cap.style.display = "inline-flex";
  } catch (e) {
    alert("Camera access denied. Please allow camera permissions.");
  }
}

function stopCamera() {
  if (_cameraStream) {
    _cameraStream.getTracks().forEach(t => t.stop());
    _cameraStream = null;
  }
  const video = document.getElementById("cameraPreview");
  if (video) video.style.display = "none";
  const btn = document.getElementById("cameraBtnText");
  if (btn) btn.textContent = "Open Camera";
  const cap = document.getElementById("captureBtn");
  if (cap) cap.style.display = "none";
}

function capturePhoto() {
  const video = document.getElementById("cameraPreview");
  if (!video) return;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob(blob => {
    _capturedImageBlob = blob;
    _imageFileSingle = new File([blob], `camera_${Date.now()}.jpg`, { type: "image/jpeg" });
    const img = document.getElementById("capturedPhoto");
    if (img) { img.src = URL.createObjectURL(blob); img.style.display = "block"; }
    _showChip("imageChip-single", `📸 Camera photo captured`);
    _styleZone("imageZone-single", "#00d4aa");
    stopCamera();
  }, "image/jpeg", 0.92);
}

/* ─ Clear ───────────────────────────────────────────────────────── */
function clearPrediction() {
  _audioFile = _audioFileMulti = _imageFileSingle = _imageFileMulti = _capturedImageBlob = null;
  stopRecording(); stopCamera();
  ["audioChip", "audioChip-multi", "imageChip-single", "imageChip-multi"].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.textContent = ""; el.classList.add("hidden"); }
  });
  ["audioInput", "audioInput-multi", "imageInput-single", "imageInput-multi"].forEach(id => {
    const el = document.getElementById(id); if (el) el.value = "";
  });
  ["textInput-text", "textInput-multi"].forEach(id => {
    const el = document.getElementById(id); if (el) el.value = "";
  });
  const player = document.getElementById("audioPlayer");
  if (player) { player.src = ""; player.style.display = "none"; }
  const capImg = document.getElementById("capturedPhoto");
  if (capImg) capImg.style.display = "none";
  const tb = document.getElementById("transcriptBox");
  if (tb) tb.className = "transcript-box";
  document.getElementById("resultCard").style.display = "none";
}

/* ── Main prediction dispatcher ─────────────────────────── */
async function runPrediction() {
  if (!_currentModality) {
    alert("Please select a modality first (Text, Audio, Image, or Multimodal).");
    return;
  }

  const btn = document.getElementById("predictBtn");
  btn.disabled = true;
  btn.textContent = "⏳ Predicting…";

  try {
    let data;
    switch (_currentModality) {
      case "text": data = await predictText(); break;
      case "audio": data = await predictAudio(); break;
      case "image": data = await predictImage(); break;
      case "multimodal": data = await predictMultimodal(); break;
      default:
        throw new Error("Unknown modality: " + _currentModality);
    }
    renderResult(data);
  } catch (e) {
    alert("Prediction failed: " + e.message);
    console.error(e);
  } finally {
    btn.disabled = false;
    btn.textContent = "🤖 Predict";
  }
}

/* ── Text-only prediction ────────────────────────────────── */
async function predictText() {
  const text = document.getElementById("textInput-text").value.trim();
  if (!text) throw new Error("Please enter some text before predicting.");

  const fd = new FormData();
  fd.append("text", text);

  const res = await fetch(`${API}/api/predict/text`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Server error (${res.status}): ${err}`);
  }
  return res.json();
}

/* ── Audio-only prediction ───────────────────────────────────────── */
async function predictAudio() {
  if (!_audioFile) throw new Error("Please upload or record an audio file first.");

  const fd = new FormData();
  fd.append("audio_file", _audioFile);
  fd.append("return_transcript", "true");   // ask backend to transcribe

  const res = await fetch(`${API}/api/predict/audio`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Server error (${res.status}): ${err}`);
  }
  const data = await res.json();

  // Show transcript if available
  if (data.transcript) {
    const box = document.getElementById("transcriptBox");
    const text = document.getElementById("transcriptText");
    if (box && text) {
      text.textContent = data.transcript;
      box.className = "transcript-box show";
    }
  }
  return data;
}

/* ── Image-only prediction ───────────────────────────────── */
async function predictImage() {
  if (!_imageFileSingle) throw new Error("Please upload a face image first.");

  const fd = new FormData();
  fd.append("image_file", _imageFileSingle);

  const res = await fetch(`${API}/api/predict/image`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Server error (${res.status}): ${err}`);
  }
  return res.json();
}

/* ── Multimodal prediction ───────────────────────────────── */
async function predictMultimodal() {
  if (!_audioFileMulti)
    throw new Error("Audio file is required for multimodal prediction.");
  const text = document.getElementById("textInput-multi").value.trim();
  if (!text)
    throw new Error("Transcript text is required for multimodal prediction.");

  const fd = new FormData();
  fd.append("audio_file", _audioFileMulti);
  if (_imageFileMulti) fd.append("image_file", _imageFileMulti);
  fd.append("text", text);

  const res = await fetch(`${API}/api/predict/multimodal`, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Server error (${res.status}): ${err}`);
  }
  return res.json();
}

/* ── Demo prediction ─────────────────────────────────────── */
async function runDemo() {
  const btn = document.getElementById("predictBtn");
  btn.disabled = true;
  btn.textContent = "⏳ Running demo…";
  try {
    const data = await fetch(`${API}/api/predict/demo`, { method: "POST" }).then(r => r.json());
    /* Populate text fields for visual context */
    const demoText = "I feel completely hopeless and exhausted. I've lost interest in everything I used to enjoy.";
    const ta1 = document.getElementById("textInput-text");
    const ta2 = document.getElementById("textInput-multi");
    if (ta1) ta1.value = demoText;
    if (ta2) ta2.value = demoText;
    renderResult(data);
  } catch (e) {
    alert("Demo failed: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "🤖 Predict";
  }
}

/* ── Render result ───────────────────────────────────────── */
function renderResult(data) {
  const card = document.getElementById("resultCard");
  card.style.display = "block";
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });

  /* Label — handle both response formats */
  const isDep = data.label_code === 1 || data.prediction === "Depressed";
  const lbl = document.getElementById("resultLabel");
  lbl.textContent = (isDep ? "🔴 Depressed" : "🟢 Non-Depressed");
  lbl.className = "result-label " + (isDep ? "dep" : "not");

  /* Risk — derive from phq_score_estimate if risk_level not provided */
  const risk = document.getElementById("resultRisk");
  let riskLevel = data.risk_level;
  if (!riskLevel && data.phq_score_estimate != null) {
    const phq = data.phq_score_estimate;
    riskLevel = phq >= 15 ? "High" : (phq >= 10 ? "Moderate" : "Low");
  }
  risk.textContent = (riskLevel || "—") + " Risk";
  risk.className = "result-risk " + (riskLevel || "").toLowerCase();

  /* Probability bars */
  const pDep = parseFloat(data.prob_depressed) || 0;
  const pNot = parseFloat(data.prob_normal) || 0;
  document.getElementById("probDepBar").style.width = (pDep * 100) + "%";
  document.getElementById("probNotBar").style.width = (pNot * 100) + "%";
  document.getElementById("probDepPct").textContent = (pDep * 100).toFixed(1) + "%";
  document.getElementById("probNotPct").textContent = (pNot * 100).toFixed(1) + "%";

  /* ── Modality Attention / Contribution chart ────────────── */
  const aw = data.attention_weights || { Audio: 0.33, Image: 0.33, Text: 0.34 };
  const modality = (data.modality || _currentModality || "multimodal").toLowerCase();
  const attnSection = document.querySelector(".attn-section h4");
  destroyChart("attnChart");

  if (modality === "multimodal") {
    /* Multimodal: show real attention-weight doughnut */
    if (attnSection) attnSection.textContent = "Modality Attention Weights";
    mkChart("attnChart", {
      type: "doughnut",
      data: {
        labels: ["Audio", "Image", "Text (BERT)"],
        datasets: [{
          data: [aw.Audio, aw.Image, aw.Text],
          backgroundColor: ["#6c63ff", "#ff6b6b", "#00d4aa"],
          borderWidth: 0, hoverOffset: 8
        }]
      },
      options: { plugins: { legend: { position: "right" } }, cutout: "58%" }
    });
  } else {
    /* Unimodal: show a simple horizontal bar for active modality + greyed others */
    if (attnSection) attnSection.textContent = "Active Modality Contribution";
    const MODAL_COLORS = { audio: "#6c63ff", image: "#ff6b6b", text: "#00d4aa" };
    const labels = ["Audio", "Image", "Text"];
    const vals = labels.map(l => aw[l] || 0);
    const bgs = labels.map(l => {
      const key = l.toLowerCase();
      return key === modality ? MODAL_COLORS[key] : "rgba(255,255,255,0.08)";
    });
    mkChart("attnChart", {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Contribution",
          data: vals,
          backgroundColor: bgs,
          borderRadius: 8
        }]
      },
      options: {
        indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: {
          x: { min: 0, max: 1, grid: { color: "rgba(255,255,255,.06)" } },
          y: { grid: { display: false } }
        }
      }
    });
  }

  /* Meta & explanation */
  const modUsed = data.model_used || data.method || "—";
  const domMod = data.dominant_modality || modality;
  document.getElementById("resultMeta").textContent =
    `Model: ${modUsed}  |  Modality: ${modality}  |  Dominant: ${domMod}`;

  let explanationText = data.explanation || data.note || "";
  if (!explanationText && data.phq_score_estimate != null) {
    explanationText = `PHQ Score Estimate: ${data.phq_score_estimate} | Confidence: ${(data.confidence * 100).toFixed(1)}%`;
  }
  document.getElementById("explanationBox").textContent = explanationText;

  /* Audio transcript visibility */
  const resTr = document.getElementById("resultTranscript");
  if (resTr) {
    if (data.transcript) {
      resTr.innerHTML = `<strong>📝 Audio Transcript:</strong><br/>"${data.transcript}"`;
      resTr.style.display = "block";
    } else {
      resTr.style.display = "none";
    }
  }
}


/* ═══════════════════════════════════════════════════════════
   SHAP
════════════════════════════════════════════════════════════ */
async function loadSHAP() {
  const loading = document.getElementById("shapLoading");
  const content = document.getElementById("shapContent");
  loading.style.display = "flex"; content.style.display = "none";

  try {
    const data = await fetch(`${API}/api/analysis/shap`).then(r => r.json());
    if (data.error) {
      loading.style.display = "none";
      alert("SHAP error: " + data.error);
      return;
    }
    loading.style.display = "none"; content.style.display = "block";

    /* Modal bars */
    const ms = data.modality_shap;
    mkChart("shapModalChart", {
      type: "bar",
      data: {
        labels: ["Audio", "Image", "Text (BERT)"],
        datasets: [{
          label: "Mean |SHAP|",
          data: [ms.Audio, ms.Image, ms.Text],
          backgroundColor: ["#6c63ff", "#ff6b6b", "#00d4aa"],
          borderRadius: 8
        }]
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, grid: { color: "rgba(255,255,255,.06)" } },
          x: { grid: { display: false } }
        }
      }
    });

    /* Pct doughnut */
    const pct = data.modality_pct;
    mkChart("shapPctChart", {
      type: "pie",
      data: {
        labels: [`Audio (${pct.Audio}%)`, `Image (${pct.Image}%)`, `Text (${pct.Text}%)`],
        datasets: [{
          data: [pct.Audio, pct.Image, pct.Text],
          backgroundColor: ["#6c63ff", "#ff6b6b", "#00d4aa"], borderWidth: 0
        }]
      },
      options: { plugins: { legend: { position: "bottom" } } }
    });

    /* Top-20 BERT */
    const b20 = data.top20_bert;
    mkChart("shapBertChart", {
      type: "bar",
      data: {
        labels: b20.labels,
        datasets: [{
          label: "Mean |SHAP|", data: b20.values,
          backgroundColor: b20.values.map((v, i) => i < 5 ? "#6c63ff" : "#6c63ff66"),
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: "y", plugins: { legend: { display: false } },
        scales: { x: { grid: { color: "rgba(255,255,255,.06)" } }, y: { grid: { display: false }, ticks: { font: { size: 11 } } } }
      }
    });

    /* Waterfall */
    const wf = data.waterfall_sample;
    Plotly.newPlot("shapWaterfallChart", [{
      type: "waterfall", orientation: "h",
      y: ["Audio (MFCC)", "Image (CLNF/HOG)", "Text (BERT)"],
      x: [wf.audio, wf.image, wf.text],
      connector: { line: { color: "rgba(255,255,255,.1)" } },
      decreasing: { marker: { color: "#00d4aa" } },
      increasing: { marker: { color: "#ff6b6b" } },
      totals: { marker: { color: "#6c63ff" } },
      text: [`${(wf.audio * 100).toFixed(2)}%`, `${(wf.image * 100).toFixed(2)}%`, `${(wf.text * 100).toFixed(2)}%`],
      textposition: "outside"
    }], [{
      title: {
        text: `Waterfall — ${wf.label} (Participant ${wf.index + 1})`,
        font: { color: "#8892b0", size: 13 }
      },
      plot_bgcolor: "transparent", paper_bgcolor: "transparent",
      font: { color: "#8892b0", size: 12 },
      margin: { t: 40, b: 20, l: 130, r: 60 },
      xaxis: { gridcolor: "rgba(255,255,255,.06)", title: "Cumulative |SHAP|" },
      yaxis: { gridcolor: "transparent" },
    }], { responsive: true, displayModeBar: false });

    /* Per-sample grouped bar */
    const ps = data.per_sample;
    const idxLabels = ps.labels.map((l, i) => `S${i + 1}(${l === 1 ? "D" : "N"})`);
    mkChart("shapPerSampleChart", {
      type: "bar",
      data: {
        labels: idxLabels, datasets: [
          { label: "Audio", data: ps.audio, backgroundColor: "#6c63ff99", borderRadius: 3 },
          { label: "Image", data: ps.image, backgroundColor: "#ff6b6b99", borderRadius: 3 },
          { label: "Text", data: ps.text, backgroundColor: "#00d4aa99", borderRadius: 3 },
        ]
      },
      options: {
        plugins: { legend: { position: "bottom" } },
        scales: {
          x: { grid: { display: false } },
          y: {
            beginAtZero: true, grid: { color: "rgba(255,255,255,.06)" },
            title: { display: true, text: "Mean |SHAP|" }
          }
        }
      }
    });

  } catch (e) {
    loading.style.display = "none";
    console.error("SHAP load error:", e);
  }
}

/* ── DRAG & DROP ──────────────────────────────────────────── */
const dropZones = [
  { zoneId: "audioZone", fileVar: () => _audioFile, setter: (f) => { _audioFile = f; setChip("audioChip", "🎵", f.name, "audioZone", "#6c63ff"); } },
  { zoneId: "audioZone-multi", fileVar: () => _audioFileMulti, setter: (f) => { _audioFileMulti = f; setChip("audioChip-multi", "🎵", f.name, "audioZone-multi", "#6c63ff"); } },
  { zoneId: "imageZone-single", fileVar: () => _imageFileSingle, setter: (f) => { _imageFileSingle = f; setChip("imageChip-single", "🖼️", f.name, "imageZone-single", "#ff6b6b"); } },
  { zoneId: "imageZone-multi", fileVar: () => _imageFileMulti, setter: (f) => { _imageFileMulti = f; setChip("imageChip-multi", "🖼️", f.name, "imageZone-multi", "#ff6b6b"); } },
];

function setChip(chipId, icon, name, zoneId, color) {
  const chip = document.getElementById(chipId);
  if (chip) { chip.textContent = `${icon} ${name}`; chip.classList.remove("hidden"); }
  const zone = document.getElementById(zoneId);
  if (zone) zone.style.borderColor = color;
}

dropZones.forEach(({ zoneId, setter }) => {
  const zone = document.getElementById(zoneId);
  if (!zone) return;
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) setter(file);
  });
});

/* ── INIT ─────────────────────────────────────────────────── */
checkHealth();
// loadDashboard(); // Defer to manual refresh or first tab click to prevent startup block
