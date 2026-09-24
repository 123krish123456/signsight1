/** Speaker Mode (PRD §1.3). M1: webcam → landmarks → WebSocket, with a live state
 *  indicator and measured FPS. Transcript/TTS land in M5. */

import { useCallback, useEffect, useRef, useState } from "react";
import { LandmarkStream } from "./landmarks";
import { SignSocket, type ServerEvent } from "./socket";
import { FEATURE_DIM } from "./normalise";

type Status = "idle" | "loading" | "running" | "error";

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<LandmarkStream | null>(null);
  const socketRef = useRef<SignSocket | null>(null);
  const stopRef = useRef<(() => void) | null>(null);
  const fpsRef = useRef({ count: 0, since: performance.now() });

  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");
  const [signing, setSigning] = useState(false);
  const [fps, setFps] = useState(0);
  const [hands, setHands] = useState<[boolean, boolean]>([false, false]);
  const [glosses, setGlosses] = useState<string[]>([]);

  const onEvent = useCallback((e: ServerEvent) => {
    if (e.type === "state") setSigning(e.value === "SIGNING");
    // "…" means a sign was detected but not understood — the user must be able to
    // tell that apart from "not signing at all" (PRD §4.5).
    else if (e.type === "gloss")
      setGlosses((g) => [...g.slice(-11), e.value === "UNKNOWN" ? "…" : e.value]);
    else if (e.type === "error") setError(`${e.code}: ${e.message}`);
  }, []);

  const stop = useCallback(() => {
    stopRef.current?.();
    socketRef.current?.control("stop");
    socketRef.current?.close();
    streamRef.current?.close();
    const tracks = (videoRef.current?.srcObject as MediaStream | null)?.getTracks();
    tracks?.forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    stopRef.current = socketRef.current = streamRef.current = null;
    setStatus("idle");
    setSigning(false);
  }, []);

  const start = useCallback(async () => {
    setStatus("loading");
    setError("");
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        // See Recorder.tsx: hand detection falls off a cliff below 720p, and the
        // live path has to match the resolution the training clips were captured at.
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: 30 },
      });
      const video = videoRef.current!;
      video.srcObject = media;
      await video.play();

      const socket = new SignSocket(onEvent);
      await socket.connect();
      socket.control("start");
      socketRef.current = socket;

      const stream = new LandmarkStream(15);
      await stream.init();
      streamRef.current = stream;

      stopRef.current = stream.start(video, ({ vector, handsPresent }) => {
        socket.sendFrame(vector, handsPresent);
        setHands(handsPresent);
        const f = fpsRef.current;
        f.count++;
        const dt = performance.now() - f.since;
        if (dt >= 1000) {
          setFps(Math.round((f.count / dt) * 1000));
          f.count = 0;
          f.since = performance.now();
        }
      });
      setStatus("running");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
      stop();
    }
  }, [onEvent, stop]);

  useEffect(() => stop, [stop]);

  return (
    <main style={S.page}>
      <header style={S.header}>
        <h1 style={S.h1}>SignSight</h1>
        <span style={S.sub}>Speaker Mode · ISL v1 · {FEATURE_DIM}-d landmarks</span>
      </header>

      <div style={S.stage}>
        <video ref={videoRef} muted playsInline style={S.video} />
        <div style={{ ...S.badge, background: signing ? "#16a34a" : "#334155" }}>
          {signing ? "SIGNING" : "IDLE"}
        </div>
      </div>

      <div style={S.row}>
        <button onClick={status === "running" ? stop : start} disabled={status === "loading"} style={S.button}>
          {status === "loading" ? "Loading model…" : status === "running" ? "Stop" : "Start camera"}
        </button>
        <span style={S.stat}>{fps} FPS</span>
        <span style={S.stat}>L {hands[0] ? "●" : "○"}　R {hands[1] ? "●" : "○"}</span>
      </div>

      {error && <p style={S.error}>{error}</p>}

      <section style={S.panel}>
        <h2 style={S.h2}>Glosses</h2>
        <p style={S.glosses}>{glosses.length ? glosses.join(" · ") : "—"}</p>
        <p style={S.note}>
          SignSight is automated and may contain errors. &ldquo;…&rdquo; means a sign was
          detected but not recognised &mdash; recognition needs a trained model.
        </p>
      </section>
    </main>
  );
}

const S: Record<string, React.CSSProperties> = {
  page: { fontFamily: "system-ui, sans-serif", maxWidth: 720, margin: "0 auto", padding: 24, color: "#e2e8f0", background: "#0f172a", minHeight: "100vh" },
  header: { display: "flex", alignItems: "baseline", gap: 12, marginBottom: 16 },
  h1: { fontSize: 24, margin: 0 },
  sub: { fontSize: 13, color: "#94a3b8" },
  stage: { position: "relative", width: "100%", aspectRatio: "4/3", background: "#020617", borderRadius: 12, overflow: "hidden" },
  video: { width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" },
  badge: { position: "absolute", top: 12, left: 12, padding: "4px 10px", borderRadius: 999, fontSize: 12, fontWeight: 600, letterSpacing: 1 },
  row: { display: "flex", alignItems: "center", gap: 16, margin: "16px 0" },
  button: { padding: "10px 18px", borderRadius: 8, border: 0, background: "#2563eb", color: "white", fontSize: 15, cursor: "pointer" },
  stat: { fontSize: 13, color: "#94a3b8", fontVariantNumeric: "tabular-nums" },
  error: { color: "#f87171", fontSize: 14 },
  panel: { background: "#1e293b", borderRadius: 12, padding: 16 },
  h2: { fontSize: 14, margin: "0 0 8px", color: "#94a3b8", textTransform: "uppercase", letterSpacing: 1 },
  glosses: { fontSize: 18, margin: 0, minHeight: 24 },
  note: { fontSize: 12, color: "#64748b", margin: "12px 0 0" },
};
