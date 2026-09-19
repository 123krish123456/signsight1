/** Where recorded clips go, and how to review them afterwards.
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

export interface StoredClip {
  name: string;
  /** Playable URL. For local folders this is an object URL: call release() when done. */
  url: string;
}

export interface ClipStore {
  readonly label: string;
  count(gloss: string): Promise<number>;
  /** Every clip of a sign, oldest first, so they can be watched and thrown away. */
  list(gloss: string): Promise<StoredClip[]>;
  save(signer: string, gloss: string, blob: Blob): Promise<void>;
  remove(gloss: string, name: string): Promise<boolean>;
  /** Free any object URLs handed out by list(). */
  release(clips: StoredClip[]): void;
}

/** Next filename for a sign.
 *
 *  Numbered from the highest index already present, NOT from how many files there are.
 *  Counting breaks as soon as anything is deleted: with 000-009 on disk, removing 005
 *  leaves nine files, and the next clip would be written as 009 — on top of an existing
 *  one. Silent data loss, and exactly what the review panel invites you to do.
 */
export function nextName(signer: string, gloss: string, existing: string[]): string {
  let highest = -1;
  for (const name of existing) {
    const m = name.match(/_(\d+)\.[a-z0-9]+$/i);
    if (m) highest = Math.max(highest, Number(m[1]));
  }
  return `${signer}_${gloss}_${String(highest + 1).padStart(3, "0")}.webm`;
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
  getFile(): Promise<File>;
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
    async list(gloss) {
      const dir = await folderFor(gloss);
      const out: StoredClip[] = [];
      for (const name of await namesIn(gloss)) {
        const file = await (await dir.getFileHandle(name)).getFile();
        out.push({ name, url: URL.createObjectURL(file) });
      }
      return out;
    },
    async save(signer, gloss, blob) {
      const dir = await folderFor(gloss);
      const name = nextName(signer, gloss, await namesIn(gloss));
      const writable = await (await dir.getFileHandle(name, { create: true })).createWritable();
      await writable.write(blob);
      await writable.close();
    },
    async remove(gloss, name) {
      try {
        await (await folderFor(gloss)).removeEntry(name);
        return true;
      } catch {
        return false;
      }
    },
    release(clips) {
      clips.forEach((c) => URL.revokeObjectURL(c.url));
    },
  };
}

// ---------------------------------------------------------------- project backend

export function backendStore(api: string, signer: string): ClipStore {
  const names = async (gloss: string): Promise<string[]> => {
    const r = await fetch(`${api}/capture/clips?signer=${encodeURIComponent(signer)}&gloss=${encodeURIComponent(gloss)}`);
    return r.ok ? (await r.json()).clips : [];
  };

  return {
    label: "the project backend",
    async count(gloss) {
      return (await names(gloss)).length;
    },
    async list(gloss) {
      return (await names(gloss)).map((name) => ({
        name,
        url: `${api}/capture/file?signer=${encodeURIComponent(signer)}&gloss=${encodeURIComponent(gloss)}&name=${encodeURIComponent(name)}`,
      }));
    },
    async save(_signer, gloss, blob) {
      const body = new FormData();
      body.append("signer", signer);
      body.append("gloss", gloss);
      body.append("video", blob, "clip");
      const r = await fetch(`${api}/capture/clip`, { method: "POST", body });
      if (!r.ok) {
        const detail = await r.json().catch(() => ({}));
        throw new Error(detail.detail ?? `HTTP ${r.status}`);
      }
    },
    async remove(gloss, name) {
      const r = await fetch(
        `${api}/capture/clip?signer=${encodeURIComponent(signer)}&gloss=${encodeURIComponent(gloss)}&name=${encodeURIComponent(name)}`,
        { method: "DELETE" },
      );
      return r.ok;
    },
    release() { /* plain URLs, nothing to revoke */ },
  };
}
