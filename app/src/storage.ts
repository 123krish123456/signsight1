/** Where recorded clips go.
 *
 *  Two backends, chosen at runtime:
 *
 *  - A folder on your own disk, via the File System Access API. You pick it once and
 *    clips are written straight there, so nothing needs the Python backend running and
 *    you keep the files to hand over however you like.
 *  - The project backend, for browsers without that API (Firefox and Safari have no
 *    `showDirectoryPicker`), which writes into the repository's own clips directory.
 *
 *  Either way the layout is the same — `<gloss>/<signer>_<gloss>_NNN.webm` — because the
 *  manifest is rebuilt from that structure, so a handed-over folder needs no merging.
 */

export interface ClipStore {
  readonly label: string;
  /** How many clips of this sign already exist. */
  count(gloss: string): Promise<number>;
  save(signer: string, gloss: string, blob: Blob): Promise<void>;
  /** Remove the most recent clip of a sign. Returns false if there was nothing to remove. */
  undo(signer: string, gloss: string): Promise<boolean>;
}

// ---------------------------------------------------------------- local folder

interface DirHandle {
  name: string;
  getDirectoryHandle(name: string, o?: { create?: boolean }): Promise<DirHandle>;
  getFileHandle(name: string, o?: { create?: boolean }): Promise<FileHandle>;
  removeEntry(name: string): Promise<void>;
  values(): AsyncIterable<{ kind: string; name: string }>;
}
interface FileHandle {
  createWritable(): Promise<{ write(d: Blob): Promise<void>; close(): Promise<void> }>;
}

export function supportsLocalFolder(): boolean {
  return typeof (window as unknown as { showDirectoryPicker?: unknown }).showDirectoryPicker === "function";
}

export async function pickLocalFolder(): Promise<ClipStore | null> {
  const picker = (window as unknown as {
    showDirectoryPicker(o?: { mode?: string }): Promise<DirHandle>;
  }).showDirectoryPicker;
  let root: DirHandle;
  try {
    root = await picker({ mode: "readwrite" });
  } catch {
    return null; // the user closed the dialog
  }

  const folderFor = (gloss: string) => root.getDirectoryHandle(gloss, { create: true });

  const namesIn = async (gloss: string): Promise<string[]> => {
    const names: string[] = [];
    try {
      const dir = await folderFor(gloss);
      for await (const entry of dir.values()) {
        if (entry.kind === "file") names.push(entry.name);
      }
    } catch { /* folder not created yet */ }
    return names.sort();
  };

  return {
    label: root.name,
    async count(gloss) {
      return (await namesIn(gloss)).length;
    },
    async save(signer, gloss, blob) {
      const dir = await folderFor(gloss);
      // Number from what is already on disk, so stopping and resuming never overwrites.
      const n = (await namesIn(gloss)).length;
      const name = `${signer}_${gloss}_${String(n).padStart(3, "0")}.webm`;
      const writable = await (await dir.getFileHandle(name, { create: true })).createWritable();
      await writable.write(blob);
      await writable.close();
    },
    async undo(_signer, gloss) {
      const names = await namesIn(gloss);
      if (!names.length) return false;
      const dir = await folderFor(gloss);
      await dir.removeEntry(names[names.length - 1]);
      return true;
    },
  };
}

// ---------------------------------------------------------------- project backend

export function backendStore(api: string, counts: Map<string, number>): ClipStore {
  return {
    label: "the project backend",
    async count(gloss) {
      return counts.get(gloss) ?? 0;
    },
    async save(signer, gloss, blob) {
      const body = new FormData();
      body.append("signer", signer);
      body.append("gloss", gloss);
      body.append("video", blob, "clip");
      const r = await fetch(`${api}/capture/clip`, { method: "POST", body });
      if (!r.ok) {
        const detail = await r.json().catch(() => ({}));
        throw new Error(detail.detail ?? `HTTP ${r.status}`);
      }
      counts.set(gloss, (counts.get(gloss) ?? 0) + 1);
    },
    async undo(signer, gloss) {
      const r = await fetch(
        `${api}/capture/clip?signer=${encodeURIComponent(signer)}&gloss=${encodeURIComponent(gloss)}`,
        { method: "DELETE" },
      );
      if (!r.ok) return false;
      counts.set(gloss, Math.max(0, (counts.get(gloss) ?? 1) - 1));
      return true;
    },
  };
}
