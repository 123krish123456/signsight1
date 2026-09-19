import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // /record has no file of its own; serve index.html for it (the app picks the page).
  appType: "spa",
  server: {
    port: 5173,
    // MediaPipe WASM wants a real origin; localhost is treated as secure so
    // getUserMedia works without HTTPS.
    host: "127.0.0.1",
  },
});
