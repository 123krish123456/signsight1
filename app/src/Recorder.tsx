/** Browser capture: reference clip on one side, your camera on the other.
 *
 *  Exists so recording does not require a Python toolchain — a teammate opens a URL,
 *  types their name and records. Clips post straight to the backend, which writes them
 *  into the same clips directory and manifest the desktop recorder uses.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  backendStore, pickLocalFolder, supportsLocalFolder,
  type ClipStore, type StoredClip,
} from "./storage";

/** Talk to the backend on whichever host served this page, not to localhost.
 *  A teammate opening http://192.168.1.42:5173/record from their own laptop must reach
 *  the backend on .42 as well — pointing at 127.0.0.1 would silently address their own
 *  machine, where nothing is running and no clips would ever be collected centrally. */
const API =
  import.meta.env.VITE_API_URL ?? `${location.protocol}//${location.hostname}:8000`;

/** The 24 signs, and where their reference clips live. Served by the dev server from
 *  assets/reference, so recording works with `npm run dev` alone — no Python. */
const SIGNS = [
  "HELLO", "THANK-YOU", "GOOD-MORNING", "HOW-ARE-YOU", "ALRIGHT", "PLEASED",
  "ME", "YOU", "HE", "SHE", "WE",
  "MOTHER", "FATHER", "FRIEND", "MAN", "WOMAN",
  "HAPPY", "SICK", "HEALTHY", "BIG", "SMALL", "COLD",
  "HOUSE", "SCHOOL",
];

/** One tab each. Picking from a list instead of typing means nobody records half a
 *  session as "Krish" and half as "krish" — signer id is the split key, and a typo
 *  invents a phantom person that quietly weakens the evaluation. */
const TEAM = ["eashan", "krish", "arpit"] as const;
const CLIP_MS = 3000;
const COUNTDOWN_MS = 1500;

interface Sign {
  gloss: string;
  pos: string;
  count: number;
  reference: string | null;
}

type Phase = "idle" | "counting" | "recording" | "saving";

export default function Recorder() {
  const camRef = useRef<HTMLVideoElement>(null);
  const refRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timersRef = useRef<number[]>([]);

  const [signer, setSigner] = useState(() => localStorage.getItem("signsight.signer") ?? "");
  const [store, setStore] = useState<ClipStore | null>(null);
  const [takes, setTakes] = useState<StoredClip[]>([]);
  const [started, setStarted] = useState(false);
  const [signs, setSigns] = useState<Sign[]>([]);
  const [current, setCurrent] = useState(0);
  const [phase, setPhase] = useState<Phase>("idle");
  const [countdown, setCountdown] = useState(0);
  const [target, setTarget] = useState(10);
  const [auto, setAuto] = useState(false);
  const [error, setError] = useState("");

  const sign = signs[current];
  const done = signs.reduce((n, s) => n + Math.min(s.count, target), 0);
  const goal = signs.length * target;

  const loadPlan = useCallback(async (where: ClipStore) => {
    const counted = await Promise.all(
      SIGNS.map(async (gloss) => ({
        gloss,
        pos: "",
        count: await where.count(gloss),
        reference: `/reference/${gloss}.mp4`,
      })),
    );
    setSigns(counted);
    const next = counted.findIndex((s) => s.count < target);
    setCurrent(next === -1 ? 0 : next);
  }, [target]);

  const begin = useCallback(async (where: ClipStore) => {
    setError("");
    const who = signer.trim().toLowerCase();
    if (!/^[a-z0-9_-]{1,32}$/.test(who)) {
      setError("Pick your tab first.");
      return;
    }
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }, audio: false,
      });
      streamRef.current = media;
      localStorage.setItem("signsight.signer", who);
      setStore(where);
      await loadPlan(where);
      setStarted(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [signer, loadPlan]);

  const startLocal = useCallback(async () => {
    const where = await pickLocalFolder();
    if (where) await begin(where);
  }, [begin]);

  const startBackend = useCallback(async () => {
    const who = signer.trim().toLowerCase();
    try {
      // fail here, with a clear message, rather than on the first save
      const r = await fetch(`${API}/health`);
      if (!r.ok) throw new Error(`backend said ${r.status}`);
      await begin(backendStore(API, who));
    } catch (e) {
      setError(`Backend unreachable at ${API}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [signer, begin]);

  /** Reload the takes for whichever sign is showing, freeing the previous object URLs.
   *  Without the release these leak a blob per clip every time you change sign. */
  const refreshTakes = useCallback(async (gloss: string | undefined) => {
    if (!store || !gloss) return;
    setTakes((old) => { store.release(old); return []; });
    try {
      setTakes(await store.list(gloss));
    } catch { /* folder not created until the first clip is saved */ }
  }, [store]);

  useEffect(() => { void refreshTakes(sign?.gloss); }, [sign?.gloss, refreshTakes]);
  useEffect(() => () => { store?.release(takes); }, [store, takes]);

  const discard = useCallback(async (name: string) => {
    if (!store || !sign) return;
    if (await store.remove(sign.gloss, name)) {
      bump(sign.gloss, -1);
      await refreshTakes(sign.gloss);
    }
  }, [store, sign, refreshTakes]);

  // Attach the camera AFTER the recording view mounts. Doing it inside start() set
  // srcObject on a ref that did not exist yet — the setup screen has no <video> — so the
  // preview stayed black while the reference played happily beside it.
  useEffect(() => {
    if (!started || !camRef.current || !streamRef.current) return;
    camRef.current.srcObject = streamRef.current;
    camRef.current.play().catch((e) => setError(`Camera preview: ${e.message}`));
  }, [started]);

  // stop every timer and the camera when leaving
  useEffect(() => () => {
    timersRef.current.forEach(clearTimeout);
    streamRef.current?.getTracks().forEach((t) => t.stop());
  }, []);

  const bump = (gloss: string, by: number) =>
    setSigns((prev) => prev.map((s) => (s.gloss === gloss ? { ...s, count: s.count + by } : s)));

  const advance = useCallback(() => {
    setSigns((prev) => {
      const from = (current + 1) % prev.length;
      // the least-recorded sign, so coverage stays even without anyone thinking about it
      let best = from, bestCount = Infinity;
      for (let i = 0; i < prev.length; i++) {
        const j = (from + i) % prev.length;
        if (prev[j].count < bestCount) { best = j; bestCount = prev[j].count; }
      }
      setCurrent(best);
      return prev;
    });
  }, [current]);

  const record = useCallback(() => {
    if (!streamRef.current || !sign || phase !== "idle") return;
    setError("");
    setPhase("counting");
    setCountdown(Math.ceil(COUNTDOWN_MS / 1000));

    const tick = window.setInterval(() => setCountdown((c) => Math.max(0, c - 1)), 1000);
    timersRef.current.push(tick as unknown as number);

    const begin = window.setTimeout(() => {
      window.clearInterval(tick);
      const rec = new MediaRecorder(streamRef.current!, { mimeType: pickMime() });
      chunksRef.current = [];
      rec.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
      rec.onstop = async () => {
        setPhase("saving");
        const blob = new Blob(chunksRef.current, { type: rec.mimeType });
        try {
          await store!.save(signer.trim().toLowerCase(), sign.gloss, blob);
          bump(sign.gloss, 1);
          setPhase("idle");
          await refreshTakes(sign.gloss);
          if (!auto) advance();
          if (auto) timersRef.current.push(window.setTimeout(() => record(), 900) as unknown as number);
        } catch (e) {
          setError(`Not saved: ${e instanceof Error ? e.message : String(e)}`);
          setPhase("idle");
        }
      };
      rec.start();
      setPhase("recording");
      timersRef.current.push(window.setTimeout(() => rec.stop(), CLIP_MS) as unknown as number);
    }, COUNTDOWN_MS);
    timersRef.current.push(begin as unknown as number);
  }, [sign, phase, signer, auto, advance, store, refreshTakes]);

  const undo = useCallback(async () => {
    if (!sign || !takes.length) return;
    await discard(takes[takes.length - 1].name);
  }, [sign, takes, discard]);

  // keyboard: space records, u undoes, n skips
  useEffect(() => {
    if (!started) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.code === "Space") { e.preventDefault(); record(); }
      else if (e.key === "u") undo();
      else if (e.key === "n") advance();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [started, record, undo, advance]);

  if (!started) {
    return (
      <main style={S.page}>
        <h1 style={S.h1}>SignSight — record</h1>
        <p style={S.lede}>
          Your camera on one side, the sign to copy on the other. Roughly 45 minutes for
          all 24 signs.
        </p>
        <div style={S.tabs}>
          {TEAM.map((name) => (
            <button
              key={name}
              onClick={() => setSigner(name)}
              style={{ ...S.tab, ...(signer === name ? S.tabOn : {}) }}
            >
              {name}
            </button>
          ))}
        </div>
        <p style={S.hint}>
          Pick your own tab. This is your signer id, and the model is evaluated by holding
          one person out, so it has to be the same every session — that is why it is a
          list and not a text box.
        </p>

        <label style={S.label}>
          Clips per sign
          <input style={S.input} type="number" min={1} max={50} value={target}
                 onChange={(e) => setTarget(Math.max(1, Number(e.target.value) || 10))} />
        </label>

        {supportsLocalFolder() ? (
          <>
            <button style={S.primary} onClick={startLocal} disabled={!signer}>
              {signer ? "Choose a folder and start" : "Pick your tab first"}
            </button>
            <p style={S.hint}>
              Clips are written straight into a folder you choose on this laptop — nothing
              is uploaded anywhere. When you are done, zip that folder and send it over.
              Pick the same folder next time and it carries on from where you stopped.
            </p>
            <details style={S.details}>
              <summary style={S.summary}>Send to the project backend instead</summary>
              <p style={{ ...S.hint, marginTop: 8 }}>
                Only useful on the machine running this project — it writes into the
                repository's own clips directory.
              </p>
              <button style={{ ...S.ghost, marginTop: 8 }} onClick={startBackend} disabled={!signer}>
                Start, saving to the backend
              </button>
            </details>
          </>
        ) : (
          <>
            <button style={S.primary} onClick={startBackend} disabled={!signer}>
              {signer ? "Allow camera and start" : "Pick your tab first"}
            </button>
            <p style={S.hint}>
              This browser cannot write to a folder directly, so clips go to the project
              backend, which must be running. Chrome can save locally instead.
            </p>
          </>
        )}
        {error && <p style={S.error}>{error}</p>}
      </main>
    );
  }

  return (
    <main style={S.page}>
      <header style={S.bar}>
        <b style={S.who}>{signer}</b>
        <span style={S.muted}>{done} / {goal} clips</span>
        <span style={S.muted}>saving to {store?.label}</span>
        <div style={S.progress}><div style={{ ...S.progressFill, width: `${goal ? (done / goal) * 100 : 0}%` }} /></div>
        <label style={S.check}>
          <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />
          keep going automatically
        </label>
      </header>

      <div style={S.stage}>
        <figure style={S.panel}>
          <figcaption style={S.capLabel}>Copy this</figcaption>
          {sign?.reference ? (
            <video ref={refRef} src={`${API}${sign.reference}`} autoPlay loop muted playsInline style={S.video} />
          ) : (
            <div style={{ ...S.video, ...S.noRef }}>no reference clip for this sign</div>
          )}
        </figure>

        <figure style={S.panel}>
          <figcaption style={S.capLabel}>You</figcaption>
          <div style={{ position: "relative" }}>
            <video ref={camRef} muted playsInline style={{ ...S.video, transform: "scaleX(-1)" }} />
            {phase === "counting" && <div style={S.overlay}>{countdown || "go"}</div>}
            {phase === "recording" && <div style={{ ...S.overlay, color: "#f87171" }}>● REC</div>}
            {phase === "saving" && <div style={S.overlay}>saving…</div>}
          </div>
        </figure>
      </div>

      <div style={S.signRow}>
        <span style={S.gloss}>{sign?.gloss ?? "—"}</span>
        <span style={S.muted}>{sign?.count ?? 0} of {target} recorded</span>
      </div>

      <div style={S.row}>
        <button style={S.primary} onClick={record} disabled={phase !== "idle"}>
          {phase === "idle" ? "Record (space)" : phase === "counting" ? "Get ready…" : phase === "recording" ? "Recording…" : "Saving…"}
        </button>
        <button style={S.ghost} onClick={undo} disabled={!sign?.count}>Undo (u)</button>
        <button style={S.ghost} onClick={advance}>Skip (n)</button>
      </div>

      {error && <p style={S.error}>{error}</p>}

      <section style={S.takes}>
        <div style={S.takesHead}>
          <span style={S.capLabel}>Your takes of {sign?.gloss}</span>
          <span style={S.muted}>
            {takes.length ? "watch them, bin the bad ones, record again" : "none yet"}
          </span>
        </div>
        {takes.length > 0 && (
          <div style={S.takeRow}>
            {takes.map((clip, i) => (
              <figure key={clip.name} style={S.take}>
                <video src={clip.url} style={S.takeVideo} controls muted playsInline
                       preload="metadata" />
                <figcaption style={S.takeCap}>
                  <span>take {i + 1}</span>
                  <button style={S.binBtn} onClick={() => discard(clip.name)}
                          title={`delete ${clip.name}`}>delete</button>
                </figcaption>
              </figure>
            ))}
          </div>
        )}
      </section>

      <details style={S.details}>
        <summary style={S.summary}>Progress by sign</summary>
        <div style={S.chips}>
          {signs.map((s, i) => (
            <button key={s.gloss} onClick={() => setCurrent(i)}
                    style={{ ...S.chip, ...(s.count >= target ? S.chipDone : {}), ...(i === current ? S.chipNow : {}) }}>
              {s.gloss} {s.count}
            </button>
          ))}
        </div>
      </details>

      <p style={S.note}>
        Mirror what the reference does — your view is flipped, so if they use their right
        hand, use yours. Change room or lighting halfway through: variety is the point.
      </p>
    </main>
  );
}

function pickMime(): string {
  for (const m of ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"]) {
    if (MediaRecorder.isTypeSupported(m)) return m;
  }
  return "";
}

const S: Record<string, React.CSSProperties> = {
  page: { fontFamily: "system-ui, sans-serif", maxWidth: 1100, margin: "0 auto", padding: 24,
          color: "#e2e8f0", background: "#0f172a", minHeight: "100vh",
          display: "flex", flexDirection: "column", gap: 16 },
  h1: { fontSize: 24, margin: 0 },
  lede: { color: "#94a3b8", margin: 0, maxWidth: "60ch" },
  label: { display: "flex", flexDirection: "column", gap: 6, fontSize: 13, color: "#94a3b8", maxWidth: 420 },
  input: { padding: "9px 11px", borderRadius: 8, border: "1px solid #334155",
           background: "#1e293b", color: "#e2e8f0", fontSize: 15 },
  bar: { display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap", fontSize: 14 },
  who: { background: "#2563eb", color: "#fff", padding: "3px 12px", borderRadius: 999,
         fontSize: 13, letterSpacing: 0.4 },
  tabs: { display: "flex", gap: 8, flexWrap: "wrap" },
  tab: { display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 2,
         padding: "10px 18px", borderRadius: 10, border: "1px solid #334155",
         background: "transparent", color: "#cbd5e1", fontSize: 15, cursor: "pointer",
         textTransform: "capitalize", minWidth: 120 },
  tabOn: { background: "#2563eb", borderColor: "#2563eb", color: "#fff" },
  tabCount: { fontSize: 11, opacity: 0.75, textTransform: "none" },
  hint: { fontSize: 12, color: "#64748b", maxWidth: "62ch", margin: 0 },
  muted: { color: "#94a3b8", fontSize: 13, fontVariantNumeric: "tabular-nums" },
  progress: { flex: 1, minWidth: 120, height: 6, background: "#1e293b", borderRadius: 999, overflow: "hidden" },
  progressFill: { height: "100%", background: "#16a34a", transition: "width .3s" },
  check: { display: "flex", alignItems: "center", gap: 6, fontSize: 13, color: "#94a3b8" },
  stage: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 14 },
  panel: { margin: 0, display: "flex", flexDirection: "column", gap: 6 },
  capLabel: { fontSize: 12, textTransform: "uppercase", letterSpacing: 1, color: "#64748b" },
  video: { width: "100%", aspectRatio: "4/3", objectFit: "cover", background: "#020617", borderRadius: 12 },
  noRef: { display: "grid", placeItems: "center", color: "#64748b", fontSize: 13 },
  overlay: { position: "absolute", inset: 0, display: "grid", placeItems: "center",
             fontSize: 64, fontWeight: 700, color: "#e2e8f0", textShadow: "0 2px 12px #000" },
  signRow: { display: "flex", alignItems: "baseline", gap: 14 },
  gloss: { fontSize: 30, fontWeight: 600, letterSpacing: 0.5 },
  row: { display: "flex", gap: 10, flexWrap: "wrap" },
  primary: { padding: "11px 20px", borderRadius: 8, border: 0, background: "#2563eb",
             color: "#fff", fontSize: 15, cursor: "pointer" },
  ghost: { padding: "11px 16px", borderRadius: 8, border: "1px solid #334155",
           background: "transparent", color: "#cbd5e1", fontSize: 14, cursor: "pointer" },
  error: { color: "#f87171", fontSize: 14, margin: 0 },
  takes: { display: "flex", flexDirection: "column", gap: 8 },
  takesHead: { display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" },
  takeRow: { display: "flex", gap: 10, overflowX: "auto", paddingBottom: 6 },
  take: { margin: 0, flex: "0 0 auto", width: 168, display: "flex",
          flexDirection: "column", gap: 4 },
  takeVideo: { width: 168, aspectRatio: "4/3", objectFit: "cover",
               background: "#020617", borderRadius: 8, transform: "scaleX(-1)" },
  takeCap: { display: "flex", justifyContent: "space-between", alignItems: "center",
             fontSize: 11, color: "#94a3b8" },
  binBtn: { fontSize: 11, padding: "2px 8px", borderRadius: 6, cursor: "pointer",
            border: "1px solid #7f1d1d", background: "transparent", color: "#f87171" },
  details: { border: "1px solid #1e293b", borderRadius: 10, padding: "10px 12px" },
  summary: { cursor: "pointer", fontSize: 13, color: "#94a3b8" },
  chips: { display: "flex", flexWrap: "wrap", gap: 6, marginTop: 10 },
  chip: { fontSize: 12, padding: "4px 9px", borderRadius: 999, border: "1px solid #334155",
          background: "transparent", color: "#94a3b8", cursor: "pointer" },
  chipDone: { borderColor: "#16a34a", color: "#4ade80" },
  chipNow: { background: "#2563eb", borderColor: "#2563eb", color: "#fff" },
  note: { fontSize: 12, color: "#64748b", maxWidth: "70ch", margin: 0 },
};
