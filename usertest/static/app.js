const state = {
  trials: [],
  index: 0,
  selectedScore: null,
  selectedNeeded: null,
  participantId: "",
  sessionId: "",
  trialStartedAt: 0,
};

const elements = {
  participantId: document.querySelector("#participantId"),
  startButton: document.querySelector("#startButton"),
  testPanel: document.querySelector("#testPanel"),
  donePanel: document.querySelector("#donePanel"),
  summaryPanel: document.querySelector("#summaryPanel"),
  progressText: document.querySelector("#progressText"),
  progressFill: document.querySelector("#progressFill"),
  trialTitle: document.querySelector("#trialTitle"),
  cleanImage: document.querySelector("#cleanImage"),
  dirtyImage: document.querySelector("#dirtyImage"),
  scoreButtons: Array.from(document.querySelectorAll(".score-button")),
  needButtons: Array.from(document.querySelectorAll(".need-button")),
  ratingForm: document.querySelector("#ratingForm"),
  submitButton: document.querySelector("#submitButton"),
  statusText: document.querySelector("#statusText"),
  restartButton: document.querySelector("#restartButton"),
  summaryButton: document.querySelector("#summaryButton"),
  thresholdBox: document.querySelector("#thresholdBox"),
  summaryBody: document.querySelector("#summaryBody"),
};

function createSessionId() {
  if (crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function shuffle(items) {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

function setPanel(panel) {
  [elements.testPanel, elements.donePanel, elements.summaryPanel].forEach((item) => {
    item.classList.add("hidden");
  });

  if (panel) {
    panel.classList.remove("hidden");
  }
}

async function fetchTrials() {
  const response = await fetch("/api/trials");
  if (!response.ok) {
    throw new Error("이미지 목록을 불러오지 못했습니다.");
  }

  const payload = await response.json();
  return payload.trials;
}

function resetSelections() {
  state.selectedScore = null;
  state.selectedNeeded = null;

  elements.scoreButtons.forEach((button) => button.classList.remove("selected"));
  elements.needButtons.forEach((button) => button.classList.remove("selected"));
  elements.submitButton.disabled = true;
  elements.statusText.textContent = "점수와 청소 여부를 선택하세요.";
}

function updateSubmitState() {
  const ready = state.selectedScore !== null && state.selectedNeeded !== null;
  elements.submitButton.disabled = !ready;
  elements.statusText.textContent = ready
    ? "선택 완료. 저장 후 다음으로 이동할 수 있습니다."
    : "점수와 청소 여부를 선택하세요.";
}

function renderTrial() {
  const trial = state.trials[state.index];

  if (!trial) {
    setPanel(elements.donePanel);
    return;
  }

  resetSelections();
  state.trialStartedAt = performance.now();

  const progress = state.index / state.trials.length;
  elements.progressText.textContent = `${state.index + 1} / ${state.trials.length}`;
  elements.progressFill.style.width = `${progress * 100}%`;
  elements.trialTitle.textContent = `평가 이미지 ${trial.image_id}`;
  elements.cleanImage.src = trial.clean_url;
  elements.dirtyImage.src = trial.dirty_url;
}

async function startTest() {
  elements.startButton.disabled = true;
  elements.startButton.textContent = "불러오는 중...";

  try {
    state.participantId = elements.participantId.value.trim();
    state.sessionId = createSessionId();
    state.index = 0;
    state.trials = shuffle(await fetchTrials());

    if (state.trials.length === 0) {
      throw new Error("평가할 이미지 쌍이 없습니다.");
    }

    setPanel(elements.testPanel);
    renderTrial();
  } catch (error) {
    elements.statusText.textContent = error.message;
    alert(error.message);
  } finally {
    elements.startButton.disabled = false;
    elements.startButton.textContent = "테스트 시작";
  }
}

async function submitCurrentTrial(event) {
  event.preventDefault();

  const trial = state.trials[state.index];
  if (!trial || state.selectedScore === null || state.selectedNeeded === null) {
    return;
  }

  elements.submitButton.disabled = true;
  elements.statusText.textContent = "저장 중...";

  const payload = {
    session_id: state.sessionId,
    participant_id: state.participantId || null,
    image_id: trial.image_id,
    group_id: trial.group_id,
    dirty_variant: trial.dirty_variant,
    clean_image: trial.clean_image,
    dirty_image: trial.dirty_image,
    cleanliness_score: state.selectedScore,
    cleaning_needed: state.selectedNeeded,
    response_ms: Math.round(performance.now() - state.trialStartedAt),
    trial_index: state.index,
    trial_count: state.trials.length,
  };

  const response = await fetch("/api/responses", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    elements.statusText.textContent = "저장 실패. 다시 시도하세요.";
    elements.submitButton.disabled = false;
    return;
  }

  state.index += 1;
  if (state.index >= state.trials.length) {
    elements.progressFill.style.width = "100%";
    setPanel(elements.donePanel);
    return;
  }

  renderTrial();
}

async function renderSummary() {
  const response = await fetch("/api/summary");
  if (!response.ok) {
    throw new Error("요약을 불러오지 못했습니다.");
  }

  const summary = await response.json();
  setPanel(elements.summaryPanel);

  if (summary.threshold) {
    elements.thresholdBox.textContent =
      `청소 필요 예측 threshold: 청결도 ${summary.threshold.threshold}점 이하 ` +
      `(accuracy ${(summary.threshold.accuracy * 100).toFixed(1)}%, ` +
      `${summary.threshold.correct}/${summary.threshold.total})`;
  } else {
    elements.thresholdBox.textContent = "아직 threshold를 계산할 응답이 없습니다.";
  }

  elements.summaryBody.innerHTML = "";
  for (const row of summary.images) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.image_id}</td>
      <td>${row.responses}</td>
      <td>${Number(row.median_cleanliness).toFixed(2)}</td>
      <td>${Number(row.stddev_cleanliness).toFixed(2)}</td>
      <td>${(row.cleaning_needed_ratio * 100).toFixed(1)}%</td>
    `;
    elements.summaryBody.appendChild(tr);
  }
}

elements.startButton.addEventListener("click", startTest);
elements.ratingForm.addEventListener("submit", submitCurrentTrial);

elements.scoreButtons.forEach((button) => {
  button.addEventListener("click", () => {
    state.selectedScore = Number(button.dataset.score);
    elements.scoreButtons.forEach((item) => item.classList.remove("selected"));
    button.classList.add("selected");
    updateSubmitState();
  });
});

elements.needButtons.forEach((button) => {
  button.addEventListener("click", () => {
    state.selectedNeeded = button.dataset.needed === "true";
    elements.needButtons.forEach((item) => item.classList.remove("selected"));
    button.classList.add("selected");
    updateSubmitState();
  });
});

elements.restartButton.addEventListener("click", () => {
  state.index = 0;
  state.trials = shuffle(state.trials);
  setPanel(elements.testPanel);
  renderTrial();
});

elements.summaryButton.addEventListener("click", () => {
  renderSummary().catch((error) => alert(error.message));
});

window.addEventListener("keydown", (event) => {
  if (event.key >= "1" && event.key <= "5") {
    const button = elements.scoreButtons.find((item) => item.dataset.score === event.key);
    button?.click();
  }

  if (event.key.toLowerCase() === "n") {
    elements.needButtons.find((item) => item.dataset.needed === "true")?.click();
  }

  if (event.key.toLowerCase() === "u") {
    elements.needButtons.find((item) => item.dataset.needed === "false")?.click();
  }
});
