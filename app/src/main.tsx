import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import Recorder from "./Recorder";

document.body.style.margin = "0";
// Two pages, no router dependency: /record is the capture tool, / is the speaker app.
const onRecordPage =
  location.pathname.replace(/\/$/, "").endsWith("/record") || location.hash === "#record";

createRoot(document.getElementById("root")!).render(
  <StrictMode>{onRecordPage ? <Recorder /> : <App />}</StrictMode>,
);
