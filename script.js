const SESSION_KEY = "facelink_session";
const MODELS_URL =
  "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model";

let modelsLoaded = false;
let capturedDescriptor = null;
let authMode = "login";
let authStream = null;
let scanStream = null;
let autoScanTimer = null;

// Store users locally in browser
let localUsers = [];
function loadLocalUsers() {
  try {
    localUsers = JSON.parse(localStorage.getItem("facelink_users")) || [];
  } catch {
    localUsers = [];
  }
}
function saveLocalUsers() {
  localStorage.setItem("facelink_users", JSON.stringify(localUsers));
}

// ── Session (localStorage — მხოლოდ ლოგინ სესიისთვის) ──────
function getSession() {
  try {
    return JSON.parse(localStorage.getItem(SESSION_KEY));
  } catch {
    return null;
  }
}
function setSession(u) {
  localStorage.setItem(SESSION_KEY, JSON.stringify(u));
}
function clearSession() {
  localStorage.removeItem(SESSION_KEY);
}

// ── Face matching ──────────────────────────────────────────
function euclideanDist(d1, d2) {
  let s = 0;
  for (let i = 0; i < d1.length; i++) s += (d1[i] - d2[i]) ** 2;
  return Math.sqrt(s);
}
function descToArr(d) {
  return Array.from(d);
}
function arrToDesc(a) {
  return new Float32Array(a);
}

async function findMatch(descriptor, threshold = 0.5) {
  let best = null,
    bestDist = Infinity;
  for (const u of localUsers) {
    const d = arrToDesc(u.descriptor);
    const dist = euclideanDist(descriptor, d);
    if (dist < bestDist) {
      bestDist = dist;
      best = u;
    }
  }
  if (best && bestDist < threshold)
    return { ...best, matchScore: Math.round((1 - bestDist) * 100) };
  return null;
}

// ── Models ────────────────────────────────────────────────
async function loadModels() {
  setStatus("loginStatus", "Loading AI models...", "info");
  try {
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODELS_URL),
      faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODELS_URL),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODELS_URL),
    ]);
    modelsLoaded = true;
    setStatus(
      "loginStatus",
      "Models ready. Position your face in the camera.",
      "success",
    );
    document.getElementById("loginBtn").disabled = false;
    startAutoScan();
  } catch (e) {
    setStatus(
      "loginStatus",
      "Failed to load models. Check your internet connection.",
      "error",
    );
    console.error(e);
  }
}

// ── Camera ────────────────────────────────────────────────
async function startCamera(videoEl) {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
    });
    videoEl.srcObject = stream;
    return stream;
  } catch (e) {
    console.error("Camera:", e);
    return null;
  }
}
function stopStream(s) {
  if (s) s.getTracks().forEach((t) => t.stop());
}

async function detectFaceInVideo(videoEl, canvasEl) {
  if (!modelsLoaded || videoEl.readyState < 2) return null;
  const opts = new faceapi.TinyFaceDetectorOptions({
    inputSize: 320,
    scoreThreshold: 0.4,
  });
  const result = await faceapi
    .detectSingleFace(videoEl, opts)
    .withFaceLandmarks(true)
    .withFaceDescriptor();
  if (result) {
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
    const ctx = canvasEl.getContext("2d");
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    const b = result.detection.box;
    ctx.strokeStyle = "#00f5c4";
    ctx.lineWidth = 2;
    ctx.shadowColor = "#00f5c4";
    ctx.shadowBlur = 10;
    ctx.strokeRect(b.x, b.y, b.width, b.height);
    ctx.shadowBlur = 0;
  }
  return result;
}

// ── Auto-scan ─────────────────────────────────────────────
function startAutoScan() {
  if (authMode !== "login") return;
  clearInterval(autoScanTimer);
  autoScanTimer = setInterval(async () => {
    if (authMode !== "login" || !modelsLoaded) return;
    const v = document.getElementById("authVideo");
    const c = document.getElementById("authCanvas");
    const r = await detectFaceInVideo(v, c);
    if (r) {
      setStatus("loginStatus", "Face detected. Comparing...", "info");
      const match = await findMatch(r.descriptor);
      if (match) {
        clearInterval(autoScanTimer);
        setSession(match);
        setStatus("loginStatus", `Welcome, ${match.name}!`, "success");
        setTimeout(() => {
          showPage("dashboard");
          renderDashboard();
        }, 1200);
      }
    }
  }, 1800);
}

// ── Login ─────────────────────────────────────────────────
async function tryLogin() {
  if (!modelsLoaded) return;
  clearInterval(autoScanTimer);
  setStatus("loginStatus", "Scanning face...", "info");
  const v = document.getElementById("authVideo");
  const c = document.getElementById("authCanvas");
  const r = await detectFaceInVideo(v, c);
  if (!r) {
    setStatus(
      "loginStatus",
      "Face not detected. Please get closer to the camera.",
      "error",
    );
    startAutoScan();
    return;
  }
  setStatus("loginStatus", "Comparing face...", "info");
  const match = await findMatch(r.descriptor);
  if (match) {
    setSession(match);
    setStatus("loginStatus", `Welcome, ${match.name}!`, "success");
    setTimeout(() => {
      showPage("dashboard");
      renderDashboard();
    }, 1000);
  } else {
    setStatus(
      "loginStatus",
      "Profile not found. Please register first.",
      "warn",
    );
    startAutoScan();
  }
}

// ── Register ──────────────────────────────────────────────
async function captureFaceForReg() {
  if (!modelsLoaded) {
    setStatus("captureStatus", "AI models still loading", "warn");
    return;
  }
  setStatus("captureStatus", "Scanning face...", "info");
  const v = document.getElementById("authVideo");
  const c = document.getElementById("authCanvas");
  const r = await detectFaceInVideo(v, c);
  if (!r) {
    setStatus("captureStatus", "Face not detected.", "error");
    return;
  }
  capturedDescriptor = r.descriptor;
  setStatus(
    "captureStatus",
    "Face captured! Please complete the form.",
    "success",
  );
  document.getElementById("regForm").classList.remove("hidden");
}

async function registerUser() {
  if (!capturedDescriptor) {
    alert("Please capture your face first!");
    return;
  }
  const name = document.getElementById("regName").value.trim();
  const surname = document.getElementById("regSurname").value.trim();
  if (!name || !surname) {
    alert("First name and last name are required!");
    return;
  }

  const regBtn = document.querySelector("#regForm .btn-primary");
  regBtn.disabled = true;
  regBtn.textContent = "Saving...";

  const user = {
    name,
    surname,
    instagram: document.getElementById("regInsta").value.trim(),
    facebook: document.getElementById("regFacebook").value.trim(),
    phone: document.getElementById("regPhone").value.trim(),
    email: document.getElementById("regEmail").value.trim(),
    descriptor: descToArr(capturedDescriptor),
    createdAt: new Date().toISOString(),
  };

  try {
    localUsers.push(user);
    saveLocalUsers();
    setSession({ ...user, id: Date.now().toString() });
    capturedDescriptor = null;
    showPage("dashboard");
    renderDashboard();
  } catch (e) {
    regBtn.disabled = false;
    regBtn.textContent = "Register";
    alert("Error saving profile: " + e.message);
    console.error(e);
  }
}

// ── Logout ────────────────────────────────────────────────
function logout() {
  clearSession();
  switchAuthTab("login");
  showPage("landing");
}

// ── Scan Camera ───────────────────────────────────────────
async function scanFromCamera() {
  if (!modelsLoaded) {
    setStatus("scanCamStatus", "AI models still loading", "warn");
    return;
  }
  setStatus("scanCamStatus", "Scanning face...", "info");
  const v = document.getElementById("scanVideo");
  const c = document.getElementById("scanCanvas");
  const r = await detectFaceInVideo(v, c);
  if (!r) {
    setStatus("scanCamStatus", "Face not detected.", "error");
    return;
  }
  setStatus("scanCamStatus", "Face detected!", "info");
  await showFaceInfo(r.descriptor);
}

// ── Scan Upload ───────────────────────────────────────────
async function handleUpload(event) {
  const file = event.target.files[0];
  if (!file || !modelsLoaded) return;
  showUploadStatus("Processing photo...", "info");
  const img = await faceapi.bufferToImage(file);
  const opts = new faceapi.TinyFaceDetectorOptions({
    inputSize: 416,
    scoreThreshold: 0.3,
  });
  const r = await faceapi
    .detectSingleFace(img, opts)
    .withFaceLandmarks(true)
    .withFaceDescriptor();
  if (!r) {
    showUploadStatus("No face detected in photo.", "error");
    return;
  }
  showUploadStatus("Face detected!", "success");
  await showFaceInfo(r.descriptor);
}
function showUploadStatus(msg, type) {
  const el = document.getElementById("uploadStatus");
  el.classList.remove("hidden");
  el.textContent = msg;
  el.className = `status ${type}`;
}

// ── Display Face Info ─────────────────────────────────────
async function showFaceInfo(descriptor) {
  const container = document.getElementById("scanResult");
  container.classList.remove("hidden");
  const match = await findMatch(descriptor);
  if (!match) {
    container.innerHTML = `
      <div class="glass-card text-center" style="max-width:420px;">
        <div style="font-size:3rem;margin-bottom:12px;">?</div>
        <h2>No Match</h2>
        <p style="color:var(--muted);">This face is not registered in the system</p>
      </div>`;
    return;
  }
  const initials = (match.name[0] || "") + (match.surname[0] || "");
  const socials = [
    match.instagram && {
      icon: "[I]",
      label: "Instagram",
      val: match.instagram,
      href: `https://instagram.com/${match.instagram.replace("@", "")}`,
    },
    match.facebook && {
      icon: "[F]",
      label: "Facebook",
      val: match.facebook,
      href: `https://facebook.com/${match.facebook}`,
    },
    match.phone && {
      icon: "[P]",
      label: "Phone",
      val: match.phone,
      href: `tel:${match.phone}`,
    },
    match.email && {
      icon: "[E]",
      label: "Email",
      val: match.email,
      href: `mailto:${match.email}`,
    },
  ].filter(Boolean);
  container.innerHTML = `
    <div class="profile-card">
      <div class="match-badge">Match: ${match.matchScore}%</div>
      <div class="profile-avatar">${initials}</div>
      <div class="profile-name">${match.name} ${match.surname}</div>
      <div class="section-title" style="margin-top:4px;">Contact Info</div>
      <div class="social-links">
        ${
          socials
            .map(
              (s) => `
          <a class="social-link" href="${s.href}" target="_blank" rel="noopener">
            <span class="social-icon">${s.icon}</span>
            <div>
              <div style="font-size:0.7rem;color:var(--muted);text-transform:uppercase;">${s.label}</div>
              <div>${s.val}</div>
            </div>
          </a>`,
            )
            .join("") || '<p style="color:var(--muted)">No contact info</p>'
        }
      </div>
    </div>`;
}

// ── Dashboard ─────────────────────────────────────────────
async function renderDashboard() {
  const session = getSession();
  const container = document.getElementById("dashProfile");
  if (!session) {
    container.innerHTML = `<div class="text-center" style="color:var(--muted);padding:20px;">Please login first</div>`;
    return;
  }
  const userCount = localUsers.length;
  const initials = (session.name[0] || "") + (session.surname[0] || "");
  const socials = [
    session.instagram && {
      icon: "[I]",
      label: "Instagram",
      val: session.instagram,
    },
    session.facebook && {
      icon: "[F]",
      label: "Facebook",
      val: session.facebook,
    },
    session.phone && {
      icon: "[P]",
      label: "Phone",
      val: session.phone,
    },
    session.email && { icon: "[E]", label: "Email", val: session.email },
  ].filter(Boolean);
  container.innerHTML = `
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px;">
      <div class="profile-avatar" style="margin-bottom:0;">${initials}</div>
      <div>
        <div class="profile-name">${session.name} ${session.surname}</div>
        <div style="color:var(--muted);font-size:0.85rem;">Locally stored profile</div>
      </div>
    </div>
    <div class="stats-row">
      <div class="stat-card"><div class="stat-val">${userCount}</div><div class="stat-lbl">Total profiles</div></div>
      <div class="stat-card"><div class="stat-val">${socials.length}</div><div class="stat-lbl">My contacts</div></div>
    </div>
    <div class="section-title">My Profile Data</div>
    <div class="social-links">
      ${
        socials
          .map(
            (s) => `
        <div class="social-link">
          <span class="social-icon">${s.icon}</span>
          <div>
            <div style="font-size:0.7rem;color:var(--muted);text-transform:uppercase;">${s.label}</div>
            <div>${s.val}</div>
          </div>
        </div>`,
          )
          .join("") || '<p style="color:var(--muted)">No contacts</p>'
      }
    </div>`;
}

// ── Navigation ────────────────────────────────────────────
async function showPage(page) {
  stopStream(authStream);
  authStream = null;
  stopStream(scanStream);
  scanStream = null;
  clearInterval(autoScanTimer);
  document.getElementById("scanResult").classList.add("hidden");
  document
    .querySelectorAll(".page")
    .forEach((p) => p.classList.remove("active"));
  document.getElementById("page-" + page).classList.add("active");
  if (page === "landing") {
    authStream = await startCamera(document.getElementById("authVideo"));
    if (modelsLoaded && authMode === "login") startAutoScan();
  }
  if (page === "scan") {
    scanStream = await startCamera(document.getElementById("scanVideo"));
    if (modelsLoaded)
      setStatus(
        "scanCamStatus",
        "✅ კამერა მზადაა — დააჭირე სკანირებას",
        "success",
      );
  }
  if (page === "dashboard") renderDashboard();
}

function switchAuthTab(mode) {
  authMode = mode;
  clearInterval(autoScanTimer);
  document
    .getElementById("tab-login")
    .classList.toggle("active", mode === "login");
  document
    .getElementById("tab-register")
    .classList.toggle("active", mode === "register");
  document
    .getElementById("loginPanel")
    .classList.toggle("hidden", mode !== "login");
  document
    .getElementById("registerPanel")
    .classList.toggle("hidden", mode !== "register");
  capturedDescriptor = null;
  document.getElementById("regForm").classList.add("hidden");
  if (mode === "login" && modelsLoaded) startAutoScan();
}
function switchScanTab(mode) {
  document
    .getElementById("tab-cam-scan")
    .classList.toggle("active", mode === "camera");
  document
    .getElementById("tab-upload-scan")
    .classList.toggle("active", mode === "upload");
  document
    .getElementById("scanCamPanel")
    .classList.toggle("hidden", mode !== "camera");
  document
    .getElementById("scanUploadPanel")
    .classList.toggle("hidden", mode !== "upload");
  document.getElementById("scanResult").classList.add("hidden");
}
function setStatus(id, msg, type) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = msg;
  el.className = `status ${type}`;
}

// ── Init ──────────────────────────────────────────────────
async function init() {
  loadLocalUsers();
  authStream = await startCamera(document.getElementById("authVideo"));
  const waitFor = (fn) =>
    new Promise((res) => {
      if (fn()) return res();
      const id = setInterval(() => {
        if (fn()) {
          clearInterval(id);
          res();
        }
      }, 100);
    });
  await waitFor(() => window.faceapi);
  await loadModels();
}

document.addEventListener("DOMContentLoaded", init);
