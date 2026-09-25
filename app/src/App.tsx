/** Speaker Mode (PRD §1.3): webcam → landmarks → WebSocket → glosses, English and
 *  speech, plus the reference sheet of what the model can actually recognise. */

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
  const [lines, setLines] = useState<string[]>([]);
  const [speak, setSpeak] = useState(true);
  const speakRef = useRef(true);
  const [vocab, setVocab] = useState<{ gloss: string; pos: string }[]>([]);
  const [showSheet, setShowSheet] = useState(false);

  // The event handler is built once, so reading a state variable inside it would read
  // the value from that first render forever. The ref is what the handler actually
  // consults, which is why unticking Speak has to write to both.
  useEffect(() => { speakRef.current = speak; }, [speak]);

  // The vocabulary is data from the pack, never hardcoded here (PRD §4.6) — the sheet
  // has to stay correct when the pack changes without anyone remembering to edit it.
  useEffect(() => {
    fetch(`${import.meta.env.VITE_API ?? "http://127.0.0.1:8000"}/vocab`)
      .then((r) => (r.ok ? r.json() : null))
      .then((p) => p && setVocab(p.entries.map((e: { gloss: string; pos: string }) =>
        ({ gloss: e.gloss, pos: e.pos }))))
      .catch(() => { /* the sheet is optional; the app works without the backend list */ });
  }, []);

  const onEvent = useCallback((e: ServerEvent) => {
    if (e.type === "state") setSigning(e.value === "SIGNING");
    // "…" means a sign was detected but not understood — the user must be able to
    // tell that apart from "not signing at all" (PRD §4.5).
    else if (e.type === "gloss")
      setGlosses((g) => [...g.slice(-11), e.value === "UNKNOWN" ? "…" : e.value]);
    else if (e.type === "transcript") {
      setLines((l) => [...l.slice(-19), e.text]);
      if (speakRef.current && "speechSynthesis" in window) {
        // Cancel first: sentences can arrive faster than they are spoken, and a queue
        // that runs behind the signer is worse than dropping a line.
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(new SpeechSynthesisUtterance(e.text));
      }
    }
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
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
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
        <label style={S.check}>
          <input type="checkbox" checked={speak} onChange={(e) => setSpeak(e.target.checked)} />
          Speak
        </label>
      </div>

      {error && <p style={S.error}>{error}</p>}

      <section style={S.panel}>
        <h2 style={S.h2}>Transcript</h2>
        <div style={S.transcript}>
          {lines.length
            ? [...lines].reverse().map((line, i) => (
                <p key={lines.length - i} style={{ ...S.line, opacity: i ? 0.5 : 1 }}>
                  {line}
                </p>
              ))
            : <p style={S.placeholder}>Sign something. Finished sentences appear here.</p>}
        </div>

        <h2 style={{ ...S.h2, marginTop: 18 }}>Glosses</h2>
        <p style={S.glosses}>{glosses.length ? glosses.join(" · ") : "—"}</p>
        <p style={S.note}>
          SignSight is automated and may contain errors. &ldquo;…&rdquo; means a sign was
          detected but not recognised &mdash; that is different from not signing at all.
        </p>
      </section>

      <section style={{ ...S.panel, marginTop: 14 }}>
        <button style={S.disclosure} onClick={() => setShowSheet((v) => !v)}>
          {showSheet ? "▾" : "▸"} Signs it knows ({vocab.length})
        </button>
        {showSheet && (
          <div style={S.chips}>
            {vocab.map((v) => (
              <span key={v.gloss} style={S.chip} title={v.pos}>{v.gloss}</span>
            ))}
          </div>
        )}
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
  glosses: { fontSize: 18, margin: 0, minHeight: 24, color: "#94a3b8", fontVariantNumeric: "tabular-nums" },
  check: { display: "flex", alignItems: "center", gap: 6, fontSize: 13, color: "#94a3b8" },
  transcript: { maxHeight: 190, overflowY: "auto" },
  line: { fontSize: 22, lineHeight: 1.35, margin: "0 0 6px" },
  placeholder: { fontSize: 15, color: "#64748b", margin: 0 },
  disclosure: { background: "transparent", border: 0, color: "#94a3b8", fontSize: 14,
                cursor: "pointer", padding: 0 },
  chips: { display: "flex", flexWrap: "wrap", gap: 6, marginTop: 12 },
  chip: { fontSize: 12, padding: "4px 9px", borderRadius: 999, background: "#0f172a",
          border: "1px solid #334155", color: "#cbd5e1" },
  note: { fontSize: 12, color: "#64748b", margin: "12px 0 0" },
};
