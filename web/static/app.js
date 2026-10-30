const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const statusMeta = document.getElementById("status-meta");
const samplesCard = document.getElementById("samples-card");
const sampleChips = document.getElementById("sample-chips");
const predictCard = document.getElementById("predict-card");
const imageIdInput = document.getElementById("image-id");
const btnPredict = document.getElementById("btn-predict");
const errorEl = document.getElementById("error-text");
const resultCard = document.getElementById("result-card");
const resultHeadline = document.getElementById("result-headline");
const resultTruth = document.getElementById("result-truth");
const topList = document.getElementById("top-list");

const SESSION_KEY = "bird-species-v1";
let lastResult = null;
let pollTimer = null;

function persistState() {
  AppSession.save(SESSION_KEY, {
    imageId: imageIdInput.value,
    lastResult,
  });
}

function restoreState() {
  const saved = AppSession.load(SESSION_KEY);
  if (!saved) return;
  if (saved.imageId) imageIdInput.value = saved.imageId;
  if (saved.lastResult) applyResult(saved.lastResult);
}

AppSession.wireHelpSave(SESSION_KEY, () => ({
  imageId: imageIdInput.value,
  lastResult,
}));

function setError(message) {
  errorEl.hidden = !message;
  errorEl.textContent = message || "";
}

function applyStatus(data) {
  statusDot.classList.remove("ready", "training", "error");
  if (data.error) {
    statusDot.classList.add("error");
    statusText.textContent = "Model failed to load.";
    statusMeta.hidden = false;
    statusMeta.textContent = data.error;
    return;
  }
  if (!data.ready) {
    statusDot.classList.add("training");
    statusText.textContent = data.training ? "Training random forest…" : "Starting…";
    statusMeta.hidden = true;
    return;
  }
  statusDot.classList.add("ready");
  statusText.textContent = "Model ready";
  statusMeta.hidden = false;
  statusMeta.textContent = `Holdout accuracy ${(data.holdout_accuracy * 100).toFixed(1)}% · CV ${(data.cv_mean * 100).toFixed(1)}% (±${(data.cv_std * 100).toFixed(1)}%) · ${data.species_count} species`;

  samplesCard.hidden = false;
  predictCard.hidden = false;
  sampleChips.innerHTML = "";
  for (const sample of data.samples || []) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip";
    btn.textContent = `#${sample.image_id} · ${sample.species.replace(/^\d+\./, "")}`;
    btn.addEventListener("click", () => {
      imageIdInput.value = String(sample.image_id);
      runPredict();
    });
    sampleChips.appendChild(btn);
  }
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function applyResult(data) {
  lastResult = data;
  resultCard.hidden = false;
  const verdict = data.correct ? "Correct species" : "Mismatch";
  resultHeadline.textContent = `${verdict}: ${data.predicted_species.replace(/^\d+\./, "")}`;
  resultHeadline.classList.toggle("correct", data.correct);
  resultHeadline.classList.toggle("wrong", !data.correct);
  resultTruth.textContent = `True label: ${data.true_species.replace(/^\d+\./, "")} (image ${data.image_id})`;

  topList.innerHTML = "";
  for (const row of data.top_predictions) {
    const li = document.createElement("li");
    const name = row.species.replace(/^\d+\./, "");
    const pct = (row.probability * 100).toFixed(1);
    li.innerHTML = `<span>${name}</span><div class="bar-track"><div class="bar" style="width:${pct}%"></div></div><span>${pct}%</span>`;
    topList.appendChild(li);
  }
  persistState();
}

async function fetchStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  applyStatus(data);
}

async function runPredict() {
  setError("");
  const imageId = Number(imageIdInput.value);
  if (!imageId) {
    setError("Enter an image ID.");
    return;
  }
  btnPredict.disabled = true;
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: imageId }),
    });
    const data = await res.json();
    if (!res.ok) {
      setError(data.detail || "Prediction failed.");
      return;
    }
    applyResult(data);
  } catch {
    setError("Network error.");
  } finally {
    btnPredict.disabled = false;
  }
}

btnPredict.addEventListener("click", runPredict);
restoreState();
pollTimer = setInterval(fetchStatus, 1500);
fetchStatus();
