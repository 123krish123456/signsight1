/** Popup — backend connection + region selection entry point (PRD §6.3). */

const status = document.getElementById("status");

document.getElementById("connect").onclick = async () => {
  status.textContent = "Connecting…";
  await chrome.runtime.sendMessage({ type: "connect" });
  const health = await fetch("http://127.0.0.1:8000/health")
    .then((r) => r.json())
    .catch(() => null);
  status.textContent = health
    ? `Connected · vocab ${health.vocab} · model ${health.model}`
    : "Backend unreachable — is uvicorn running on :8000?";
};

document.getElementById("capture").onclick = () => {
  // M6: chooseDesktopMedia → offscreen document → region crop → MediaPipe.
  status.textContent = "Region capture arrives in M6.";
};
