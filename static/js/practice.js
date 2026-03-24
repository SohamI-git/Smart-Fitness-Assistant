// static/js/practice.js

// ─── State ────────────────────────────────────────────────────────────────────
let stream            = null;
let isRunning         = false;
let intervalId        = null;
let poseStartTime     = null;
let currentPose       = null;
let holdSeconds       = 0;
let timerInterval     = null;
let poseIsCorrect     = false;
let isProcessing      = false;
let lastReferencePose = null;
let validHoldStart    = null;
let validHoldSeconds  = 0;

const MIN_HOLD_SECS = 3;
const MIN_SCORE     = 60;

const webcam    = document.getElementById("webcam");
const processed = document.getElementById("processed-frame");

// ─── Voice engine ─────────────────────────────────────────────────────────────
const Voice = {
  enabled:       true,
  speed:         0.9,
  volume:        0.8,
  mode:          "full",        // full | corrections | pose
  lastSpoken:    "",
  lastPose:      "",
  lastCorrections: [],
  speakTimeout:  null,
  speaking:      false,
  correctCount:  0,             // tracks consecutive correct frames

  // Encouraging phrases for correct poses
  encouragements: [
    "Perfect! Hold this pose.",
    "Great form! Keep holding.",
    "Excellent! You're doing great.",
    "Beautiful pose! Stay there.",
    "Well done! Maintain this position.",
  ],

  speak(text, priority = false) {
    if (!this.enabled || !text) return;
    if (text === this.lastSpoken && !priority) return;

    // Cancel current speech if priority (e.g. new pose detected)
    if (priority) window.speechSynthesis.cancel();

    // Debounce — don't speak same thing within 4 seconds
    if (this.speakTimeout) clearTimeout(this.speakTimeout);

    this.speakTimeout = setTimeout(() => {
      const utterance        = new SpeechSynthesisUtterance(text);
      utterance.rate         = this.speed;
      utterance.volume       = this.volume;
      utterance.pitch        = 1.0;
      utterance.lang         = "en-US";
      utterance.onstart      = () => { this.speaking = true; };
      utterance.onend        = () => { this.speaking = false; };
      utterance.onerror      = () => { this.speaking = false; };

      window.speechSynthesis.speak(utterance);
      this.lastSpoken = text;
    }, 300);
  },

  announceNewPose(poseName) {
    if (poseName === this.lastPose) return;
    this.lastPose    = poseName;
    this.correctCount = 0;
    this.lastCorrections = [];

    if (this.mode === "corrections") return;
    this.speak(`${poseName}`, true);
  },

  announceCorrections(corrections, isCorrect, score) {
    if (!this.enabled) return;

    if (isCorrect) {
      this.correctCount++;
      // Only say encouraging message every 10 correct frames (~5 seconds)
      if (this.correctCount % 10 === 1) {
        const msg = this.encouragements[
          Math.floor(Math.random() * this.encouragements.length)
        ];
        if (this.mode !== "pose") this.speak(msg);
      }
      return;
    }

    this.correctCount = 0;
    if (this.mode === "pose") return;
    if (!corrections || corrections.length === 0) return;

    // Only speak if corrections changed
    const corrKey = corrections.join("|");
    if (corrKey === this.lastCorrections.join("|")) return;
    this.lastCorrections = corrections;

    // Speak first correction only — reading all would be too long
    this.speak(corrections[0]);
  },

  announcePoseNotMatching(targetPose) {
    const msg = `This does not look like ${targetPose}. Check the reference image.`;
    if (msg !== this.lastSpoken) this.speak(msg);
  },

  announceHoldProgress(secondsRemaining) {
    if (this.mode === "pose") return;
    if (secondsRemaining === 2) this.speak("Almost there, keep holding.");
    if (secondsRemaining === 0) this.speak("Pose logged!");
  },

  stop() {
    window.speechSynthesis.cancel();
    this.lastSpoken      = "";
    this.lastPose        = "";
    this.lastCorrections = [];
    this.correctCount    = 0;
  }
};

// ─── Voice control handlers ───────────────────────────────────────────────────
function toggleVoice(enabled) {
  Voice.enabled = enabled;
  const label   = document.getElementById("voice-status-label");
  label.textContent = enabled ? "On" : "Off";
  label.style.color = enabled ? "#4caf8a" : "#888";
  if (!enabled) Voice.stop();
}

document.getElementById("voice-speed").addEventListener("input", function() {
  Voice.speed = parseFloat(this.value);
  document.getElementById("voice-speed-label").textContent = this.value + "x";
});

document.getElementById("voice-volume").addEventListener("input", function() {
  Voice.volume = parseFloat(this.value);
  document.getElementById("voice-volume-label").textContent =
    Math.round(this.value * 100) + "%";
});

document.getElementById("feedback-mode").addEventListener("change", function() {
  Voice.mode = this.value;
});

// ─── Hold timer ───────────────────────────────────────────────────────────────
function startHoldTimer() {
  timerInterval = setInterval(() => {
    if (!currentPose || !poseStartTime) return;
    holdSeconds = Math.floor((Date.now() - poseStartTime) / 1000);

    if (validHoldStart) {
      const prev         = validHoldSeconds;
      validHoldSeconds   = Math.floor((Date.now() - validHoldStart) / 1000);
      const remaining    = Math.max(0, MIN_HOLD_SECS - validHoldSeconds);

      // Voice hold countdown
      if (prev !== validHoldSeconds) {
        Voice.announceHoldProgress(remaining);
      }
    }

    updateTimerDisplay();
  }, 1000);
}

function updateTimerDisplay() {
  const timerEl  = document.getElementById("pose-timer");
  const statusEl = document.getElementById("calorie-status");
  timerEl.textContent = `Hold time: ${holdSeconds}s`;

  if (!poseIsCorrect) return;

  const remaining = Math.max(0, MIN_HOLD_SECS - validHoldSeconds);
  if (remaining > 0) {
    statusEl.textContent = `Hold for ${remaining} more second${remaining !== 1 ? "s" : ""} to start tracking`;
    statusEl.style.color = "#f0a500";
  } else {
    statusEl.textContent = "✓ Calories being tracked";
    statusEl.style.color = "#4caf8a";
  }
}

// ─── Camera ──────────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 }
    });
    webcam.srcObject        = stream;
    webcam.style.display    = "block";
    processed.style.display = "block";

    document.getElementById("btn-start").disabled = true;
    document.getElementById("btn-stop").disabled  = false;

    isRunning  = true;
    intervalId = setInterval(captureAndProcess, 500);
    startHoldTimer();

    Voice.speak("Camera started. Ready to detect your pose.", true);

  } catch (err) {
    alert("Camera error: " + err.message);
  }
}

function stopCamera() {
  isRunning = false;
  clearInterval(intervalId);
  clearInterval(timerInterval);
  if (stream) stream.getTracks().forEach(t => t.stop());
  webcam.style.display    = "none";
  processed.style.display = "none";
  document.getElementById("btn-start").disabled = false;
  document.getElementById("btn-stop").disabled  = true;
  hideReferencePanel();
  Voice.stop();
}

// ─── Target pose changed ──────────────────────────────────────────────────────
function onTargetPoseChanged() {
  const target = document.getElementById("target-pose").value;
  currentPose      = null;
  holdSeconds      = 0;
  validHoldSeconds = 0;
  validHoldStart   = null;
  poseIsCorrect    = false;
  Voice.lastPose   = "";
  Voice.lastCorrections = [];

  document.getElementById("pose-timer").textContent     = "Hold time: 0s";
  document.getElementById("calorie-status").textContent = "";
  hideReferencePanel();

  if (target) {
    Voice.speak(`Target set to ${target}. Try to perform this pose.`, true);
  } else {
    Voice.speak("Auto detect mode. Perform any yoga pose.", true);
  }
}

// ─── Capture and process ──────────────────────────────────────────────────────
async function captureAndProcess() {
  if (!isRunning || isProcessing) return;
  isProcessing = true;

  try {
    const canvas  = document.createElement("canvas");
    canvas.width  = 320;
    canvas.height = 240;
    canvas.getContext("2d").drawImage(webcam, 0, 0, 320, 240);
    const imageData  = canvas.toDataURL("image/jpeg", 0.6);
    const weight     = document.getElementById("weight").value;
    const targetPose = document.getElementById("target-pose").value;

    const resp = await fetch("/api/process_frame", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ image: imageData, weight, target_pose: targetPose })
    });
    const data = await resp.json();
    updateUI(data);
  } catch (err) {
    console.error("Processing error:", err);
  } finally {
    isProcessing = false;
  }
}

// ─── Update UI ────────────────────────────────────────────────────────────────
function updateUI(data) {
  if (!data.pose_detected) {
    document.getElementById("pose-name").textContent = "No pose detected";
    document.getElementById("corrections-list").innerHTML =
      makeInfoBox("#888", "#222", "#555", "Step into frame");
    resetValidHold();
    hideReferencePanel();
    return;
  }

  if (data.frame) processed.src = data.frame;

  const targetPose = document.getElementById("target-pose").value;

  if (data.mode === "target" && targetPose) {
    // ── TARGET MODE ──────────────────────────────────────────────────────────
    document.getElementById("pose-name").textContent = toTitleCase(targetPose);

    const matchBadge = data.predicted_match
      ? `<span style="color:#4caf8a;font-size:0.85rem">✓ Pose recognised</span>`
      : `<span style="color:#f0a500;font-size:0.85rem">≈ Model sees: ${toTitleCase(data.predicted_pose)}</span>`;
    document.getElementById("top3-list").innerHTML =
      `<div style="margin-top:4px">${matchBadge}</div>
       <div style="color:#888;font-size:0.8rem;margin-top:4px">
         Confidence: ${data.confidence}%</div>`;

    const score = data.score || 0;
    document.getElementById("confidence-bar").style.width      = score + "%";
    document.getElementById("confidence-bar").style.background =
      score >= 75 ? "#4caf8a" : score >= 50 ? "#f0a500" : "#e05252";
    document.getElementById("confidence-text").textContent = score.toFixed(0) + "%";

    const scoreEl = document.getElementById("score-display");
    scoreEl.textContent = score.toFixed(1) + " / 100";
    scoreEl.style.color = score >= 75 ? "#4caf8a" : score >= 50 ? "#f0a500" : "#e05252";

    const corrEl = document.getElementById("corrections-list");
    let html = "";
    if (data.feedback) {
      const fc = data.is_correct ? "#4caf8a" : "#f0a500";
      const fb = data.is_correct ? "#1a2f1a"  : "#2f2a1a";
      const fl = data.is_correct ? "#4caf8a"  : "#f0a500";
      html += `<div style="color:${fc};background:${fb};border-left:3px solid ${fl};
        padding:5px 8px;border-radius:0 6px 6px 0;margin-bottom:6px;
        font-weight:500">${data.feedback}</div>`;
    }
    if (data.corrections && data.corrections.length > 0) {
      html += data.corrections.map(c =>
        `<div style="padding:4px 8px;margin:3px 0;background:#2d1f1f;
          border-left:3px solid #e05252;border-radius:0 6px 6px 0;
          color:#ffaaaa;font-size:0.88rem">${c}</div>`
      ).join("");
    } else if (data.has_reference && data.is_correct) {
      html += makeInfoBox("#4caf8a","#1a2f1a","#4caf8a","Great form! Hold the pose.");
    } else if (!data.has_reference) {
      html += makeInfoBox("#888","#222","#555","No angle reference for this pose yet.");
    }
    corrEl.innerHTML = html;

    // Voice feedback for target mode
    if (!data.is_performing) {
      Voice.announcePoseNotMatching(targetPose);
    } else {
      Voice.announceCorrections(data.corrections, data.is_correct, score);
    }

    poseIsCorrect = data.is_correct || false;
    updateValidHold(poseIsCorrect, score);

    if (!data.is_performing || !data.is_correct) {
      showReferencePanel(targetPose, data.corrections || []);
      updateCalorieStatus(false, "target_wrong");
    } else {
      hideReferencePanel();
      updateCalorieStatus(true,
        validHoldSeconds >= MIN_HOLD_SECS ? "ready" : "holding");
    }

  } else {
    // ── FREE DETECTION MODE ──────────────────────────────────────────────────
    const poseName = data.predicted_pose || "Unknown";
    document.getElementById("pose-name").textContent = toTitleCase(poseName);

    const conf = data.confidence || 0;
    document.getElementById("confidence-bar").style.width      = conf + "%";
    document.getElementById("confidence-bar").style.background = "#7c6af5";
    document.getElementById("confidence-text").textContent     = conf + "%";

    if (data.top3) {
      document.getElementById("top3-list").innerHTML = data.top3
        .map((p, i) => `<div>${i+1}. ${toTitleCase(p.pose)} — ${p.confidence}%</div>`)
        .join("");
    }

    const score   = data.score || 0;
    const scoreEl = document.getElementById("score-display");
    scoreEl.textContent = score + " / 100";
    scoreEl.style.color = score >= 75 ? "#4caf8a" : score >= 50 ? "#f0a500" : "#e05252";

    const corrEl = document.getElementById("corrections-list");
    if (data.corrections && data.corrections.length > 0) {
      corrEl.innerHTML = data.corrections.map(c =>
        `<div style="padding:4px 8px;margin:3px 0;background:#2d1f1f;
          border-left:3px solid #e05252;border-radius:0 6px 6px 0;
          color:#ffaaaa">${c}</div>`
      ).join("");
    } else if (data.has_reference) {
      corrEl.innerHTML = makeInfoBox("#4caf8a","#1a2f1a","#4caf8a","Great form! Keep holding.");
    } else {
      corrEl.innerHTML = makeInfoBox("#888","#222","#555","No reference data for this pose.");
    }

    hideReferencePanel();

    // Voice feedback for free mode
    Voice.announceNewPose(poseName);
    Voice.announceCorrections(data.corrections, data.is_correct, score);

    const canTrack = data.can_track || false;
    poseIsCorrect  = canTrack;
    updateValidHold(canTrack, score);
    updateCalorieStatus(canTrack,
      canTrack ? (validHoldSeconds >= MIN_HOLD_SECS ? "ready" : "holding")
               : "low_confidence"
    );
  }

  document.getElementById("calories-display").textContent =
    (data.calories_total || 0).toFixed(3) + " kcal";

  const activePose = targetPose || data.predicted_pose;
  if (activePose !== currentPose) {
    currentPose   = activePose;
    poseStartTime = Date.now();
    holdSeconds   = 0;
    resetValidHold();
  }
}

// ─── Valid hold ───────────────────────────────────────────────────────────────
function updateValidHold(isValid, score) {
  if (isValid && score >= MIN_SCORE) {
    if (!validHoldStart) validHoldStart = Date.now();
    validHoldSeconds = Math.floor((Date.now() - validHoldStart) / 1000);
  } else {
    resetValidHold();
  }
}

function resetValidHold() {
  validHoldStart   = null;
  validHoldSeconds = 0;
}

// ─── Calorie status ───────────────────────────────────────────────────────────
function updateCalorieStatus(isActive, reason) {
  const el        = document.getElementById("calorie-status");
  const remaining = Math.max(0, MIN_HOLD_SECS - validHoldSeconds);

  if (!isActive) {
    el.textContent = reason === "target_wrong"
      ? "✗ Correct your pose to start tracking"
      : "✗ Pose not clear enough — adjust position";
    el.style.color = "#e05252";
  } else if (reason === "holding") {
    el.textContent = `Hold for ${remaining} more second${remaining !== 1 ? "s" : ""}…`;
    el.style.color = "#f0a500";
  } else {
    el.textContent = "✓ Calories being tracked";
    el.style.color = "#4caf8a";
  }
}

// ─── Reference panel ─────────────────────────────────────────────────────────
async function showReferencePanel(poseName, corrections) {
  const panel = document.getElementById("reference-panel");
  panel.style.display = "block";

  document.getElementById("reference-title").textContent =
    `Reference: ${toTitleCase(poseName)}`;
  document.getElementById("reference-badge").textContent = "Pose not matching";
  document.getElementById("reference-badge").className   = "badge-wrong";

  if (poseName !== lastReferencePose) {
    lastReferencePose = poseName;
    const img = document.getElementById("reference-img");
    const ph  = document.getElementById("reference-placeholder");
    img.style.display = "none";
    ph.style.display  = "block";
    ph.textContent    = "Loading...";

    try {
      const resp = await fetch(
        `/api/reference_image?pose=${encodeURIComponent(poseName)}`
      );
      const data = await resp.json();
      if (data.exists) {
        img.onload  = () => { img.style.display = "block"; ph.style.display = "none"; };
        img.onerror = () => { ph.textContent = "Image failed to load"; };
        img.src = data.url + "?t=" + Date.now();
      } else {
        ph.textContent = "No reference image available";
      }
    } catch (e) {
      ph.textContent = "Could not load reference image";
    }
  }

  const tipsEl = document.getElementById("reference-tips");
  if (corrections && corrections.length > 0) {
    tipsEl.innerHTML = `
      <h4>What to fix</h4>
      <ul>${corrections.map(c => `<li>${c}</li>`).join("")}</ul>
      <p style="color:#888;font-size:0.78rem;margin-top:8px">
        Match the reference image to score points.
      </p>`;
  } else {
    tipsEl.innerHTML = `
      <h4>Try to match</h4>
      <p style="color:#aaa;font-size:0.85rem;line-height:1.6">
        Align your body with the reference image shown.
      </p>`;
  }
}

function hideReferencePanel() {
  document.getElementById("reference-panel").style.display = "none";
  lastReferencePose = null;
}

// ─── Log pose ─────────────────────────────────────────────────────────────────
async function logCurrentPose() {
  if (!currentPose) { alert("No pose detected yet."); return; }

  const targetPose = document.getElementById("target-pose").value;
  if (targetPose && !poseIsCorrect) {
    alert("Pose does not match the target. Correct your form first.");
    return;
  }
  if (validHoldSeconds < MIN_HOLD_SECS) {
    Voice.speak(`Hold for ${MIN_HOLD_SECS - validHoldSeconds} more seconds.`, true);
    alert(`Hold the correct pose for at least ${MIN_HOLD_SECS} seconds. Currently: ${validHoldSeconds}s`);
    return;
  }

  const weight = document.getElementById("weight").value;
  const resp   = await fetch("/api/log_pose", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({
      pose_name:    currentPose,
      duration_sec: holdSeconds,
      is_correct:   poseIsCorrect && validHoldSeconds >= MIN_HOLD_SECS,
      weight:       weight,
      score:        parseFloat(document.getElementById("score-display")
                               .textContent) || 0,
    })
  });

  const data = await resp.json();
  updateSessionLog(data.session_log);
  document.getElementById("calories-display").textContent =
    data.calories_total.toFixed(3) + " kcal";

  Voice.speak(`Pose logged. ${data.calories_this.toFixed(3)} kilocalories earned.`, true);

  poseStartTime = Date.now();
  holdSeconds   = 0;
  resetValidHold();
}

// ─── Reset ────────────────────────────────────────────────────────────────────
async function resetSession() {
  await fetch("/api/reset_session", { method: "POST" });
  document.getElementById("calories-display").textContent  = "0.000 kcal";
  document.getElementById("session-log").innerHTML         = "No poses logged yet.";
  document.getElementById("pose-timer").textContent        = "Hold time: 0s";
  document.getElementById("calorie-status").textContent    = "";
  hideReferencePanel();
  currentPose      = null;
  holdSeconds      = 0;
  poseIsCorrect    = false;
  Voice.stop();
  Voice.speak("Session reset.", true);
  resetValidHold();
}

// ─── Session log ──────────────────────────────────────────────────────────────
function updateSessionLog(log) {
  if (!log || log.length === 0) return;
  const html = log.slice().reverse().map(e =>
    `<div class="log-entry">
      <span>${toTitleCase(e.pose)}</span>
      <span>${e.duration_sec}s</span>
      <span class="${e.is_correct ? 'correct' : 'incorrect'}">
        ${e.is_correct ? "✓" : "✗"}
      </span>
      <span>${e.calories} kcal</span>
    </div>`
  ).join("");
  document.getElementById("session-log").innerHTML = html;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function makeInfoBox(color, bg, border, text) {
  return `<div style="color:${color};background:${bg};
    border-left:3px solid ${border};padding:5px 8px;
    border-radius:0 6px 6px 0">${text}</div>`;
}

function toTitleCase(str) {
  if (!str) return "";
  return str.replace(/\b\w/g, c => c.toUpperCase());
}