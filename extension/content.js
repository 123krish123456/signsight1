/**
 * Transcript overlay (PRD §6.4). Fixed panel, draggable, last 4 lines, and a
 * permanent honesty label — machine output is never presented as verified.
 */

const MAX_LINES = 4;
const lines = [];

const panel = document.createElement("div");
panel.id = "signsight-overlay";
panel.style.cssText = `
  position: fixed; right: 16px; bottom: 16px; z-index: 2147483647;
  width: 340px; padding: 12px 14px; border-radius: 12px;
  background: rgba(15,23,42,.88); color: #e2e8f0; opacity: .95;
  font: 15px/1.45 system-ui, sans-serif; cursor: grab;
  box-shadow: 0 8px 24px rgba(0,0,0,.4);
`;
panel.innerHTML = `
  <div id="ss-lines" style="min-height:64px"></div>
  <div style="margin-top:8px;display:flex;align-items:center;gap:8px">
    <span style="font-size:11px;color:#94a3b8">SignSight — automated, may contain errors</span>
    <input id="ss-opacity" type="range" min="30" max="100" value="95" style="margin-left:auto;width:70px">
  </div>
`;
document.documentElement.appendChild(panel);

const linesEl = panel.querySelector("#ss-lines");
panel.querySelector("#ss-opacity").oninput = (e) => {
  panel.style.opacity = e.target.value / 100;
};

// drag by the panel body
let drag = null;
panel.onpointerdown = (e) => {
  if (e.target.id === "ss-opacity") return;
  drag = { x: e.clientX, y: e.clientY, r: panel.getBoundingClientRect() };
  panel.setPointerCapture(e.pointerId);
};
panel.onpointermove = (e) => {
  if (!drag) return;
  panel.style.left = `${drag.r.left + e.clientX - drag.x}px`;
  panel.style.top = `${drag.r.top + e.clientY - drag.y}px`;
  panel.style.right = panel.style.bottom = "auto";
};
panel.onpointerup = () => (drag = null);

function render() {
  linesEl.textContent = "";
  for (const l of lines.slice(-MAX_LINES)) {
    const p = document.createElement("p");
    p.style.cssText = "margin:0 0 4px";
    p.textContent = l;
    linesEl.appendChild(p);
  }
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "transcript") lines.push(msg.text);
  // below-threshold recognition shows as "…" — the user must be able to tell
  // "not understood" from "not signing" (PRD §4.5)
  else if (msg.type === "gloss" && msg.value === "UNKNOWN") lines.push("…");
  else if (msg.type === "status") lines.push(msg.connected ? "[connected]" : "[disconnected]");
  else return;
  render();
});
