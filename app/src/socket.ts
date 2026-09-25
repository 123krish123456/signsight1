/** WebSocket client for /ws/stream (PRD §5.1). Landmarks only — never video (§3.1). */

// Follow the serving host, so the app works when opened from another machine on the
// same network rather than only from the one running the backend.
const WS_URL =
  import.meta.env.VITE_WS_URL ?? `ws://${location.hostname}:8000/ws/stream`;

export type ServerEvent =
  | { type: "state"; value: "IDLE" | "SIGNING" }
  | { type: "gloss"; value: string; confidence: number; segment_ms: number; segment?: string; via?: "model" | "memory" }
  | { type: "transcript"; text: string; is_final: boolean }
  | { type: "meter"; energy: number; enter: number; exit: number }
  | { type: "error"; code: string; message: string };

export class SignSocket {
  private ws: WebSocket | null = null;
  private seq = 0;
  readonly sessionId = crypto.randomUUID();

  constructor(private readonly onEvent: (e: ServerEvent) => void) {}

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(WS_URL);
      ws.onopen = () => resolve();
      ws.onerror = () => reject(new Error(`cannot reach backend at ${WS_URL}`));
      ws.onmessage = (m) => this.onEvent(JSON.parse(m.data));
      ws.onclose = () => (this.ws = null);
      this.ws = ws;
    });
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  sendFrame(landmarks: Float64Array, handsPresent: [boolean, boolean]): void {
    if (!this.connected) return;
    // bufferedAmount guard: if the socket is backing up, drop this frame rather
    // than queue stale landmarks the segmenter would mis-time (PRD §5.3).
    if (this.ws!.bufferedAmount > 1 << 20) return;
    this.ws!.send(
      JSON.stringify({
        type: "frame",
        session_id: this.sessionId,
        seq: this.seq++,
        t_client_ms: Date.now(),
        landmarks: Array.from(landmarks),
        hands_present: handsPresent,
      }),
    );
  }

  control(action: "start" | "stop" | "reset_buffer"): void {
    if (this.connected) this.ws!.send(JSON.stringify({ type: "control", action }));
  }

  close(): void {
    this.ws?.close();
    this.ws = null;
  }
}
