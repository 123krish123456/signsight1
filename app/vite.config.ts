import { readFileSync, existsSync } from "node:fs";
import { extname, resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const REFERENCE_DIR = resolve(__dirname, "../assets/reference");

/** Serve the reference sign clips straight from assets/ during development.
 *  They live outside app/, and copying them into app/public would duplicate ~1 MB of
 *  video in git. Serving them here means the recorder works with only `npm run dev` —
 *  no Python backend needed just to record. */
function referenceClips() {
  return {
    name: "signsight-reference-clips",
    configureServer(server: { middlewares: { use: (fn: unknown) => void } }) {
      server.middlewares.use((req: any, res: any, next: () => void) => {
        if (!req.url?.startsWith("/reference/")) return next();
        const file = resolve(REFERENCE_DIR, decodeURIComponent(req.url.slice("/reference/".length)));
        // never serve outside the reference directory
        if (!file.startsWith(REFERENCE_DIR) || !existsSync(file)) {
          res.statusCode = 404;
          return res.end("no such reference clip");
        }
        res.setHeader("Content-Type", extname(file) === ".mp4" ? "video/mp4" : "application/octet-stream");
        res.end(readFileSync(file));
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), referenceClips()],
  // /record has no file of its own; serve index.html for it (the app picks the page).
  appType: "spa",
  server: {
    port: 5173,
    // MediaPipe WASM wants a real origin; localhost is treated as secure so
    // getUserMedia works without HTTPS.
    host: "127.0.0.1",
  },
});
