/**
 * MV3 service worker — owns the WebSocket to the backend (PRD §6.3 step 5).
 * M0 scope: connection lifecycle + message relay. Capture wiring lands in M6.
 */

const WS_URL = "ws://127.0.0.1:8000/ws/stream";

let socket = null;
let seq = 0;

function connect() {
  if (socket && socket.readyState <= WebSocket.OPEN) return;
  socket = new WebSocket(WS_URL);
  socket.onopen = () => broadcast({ type: "status", connected: true });
  socket.onclose = () => {
    socket = null;
    broadcast({ type: "status", connected: false });
  };
  socket.onmessage = (m) => broadcast(JSON.parse(m.data));
}

/** To the overlay content script in whichever tab is active. */
function broadcast(payload) {
  chrome.tabs.query({ url: "https://meet.google.com/*" }, (tabs) => {
    for (const t of tabs) chrome.tabs.sendMessage(t.id, payload).catch(() => {});
  });
}

chrome.runtime.onMessage.addListener((msg, _sender, respond) => {
  if (msg.type === "connect") {
    connect();
    respond({ ok: true });
  } else if (msg.type === "frame") {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ ...msg, seq: seq++, t_client_ms: Date.now() }));
    }
    respond({ ok: true });
  } else if (msg.type === "disconnect") {
    socket?.close();
    respond({ ok: true });
  }
  return true;
});
