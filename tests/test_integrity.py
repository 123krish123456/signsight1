"""Cross-cutting invariants: the things that are true of no single file.

Every other test checks one component. These check that components still agree with each
other, which is the failure mode this project keeps producing — the model exported for
one vocabulary and the server loading another, the mock naming signs that no longer
exist, a documented environment variable that stopped being a setting.

Each of these has already happened here at least once. None was caught by a unit test,
because each component was individually correct.
"""

import json
import re
from pathlib import Path

import pytest

from backend.config import ROOT, Settings, settings
from backend.vocab.schema import load_pack

PACK = load_pack(settings.vocab_pack)


def test_the_shipped_model_matches_the_shipped_vocabulary():
    """A model trained on a different pack loads fine and is wrong about everything.

    The server refuses this at startup; this catches it at commit time instead.
    """
    if not settings.model_path.exists():
        pytest.skip(f"no model at {settings.model_path}")
    import onnxruntime as ort

    session = ort.InferenceSession(str(settings.model_path),
                                   providers=["CPUExecutionProvider"])
    out = session.get_outputs()[0].shape
    assert out[-1] == len(PACK.labels), (
        f"{settings.model_path.name} predicts {out[-1]} classes, "
        f"{PACK.name} has {len(PACK.labels)}"
    )

    shape = session.get_inputs()[0].shape
    assert shape[-1] == settings.expected_dim, f"input is {shape[-1]}-d, config expects {settings.expected_dim}"
    assert shape[-2] == settings.segment_resample_frames, (
        f"model wants {shape[-2]} frames, the segmenter emits {settings.segment_resample_frames}"
    )


def test_the_model_metadata_lists_the_same_classes_as_the_pack():
    meta = settings.model_path.with_suffix(".json")
    if not meta.exists():
        pytest.skip("no sidecar metadata")
    assert json.loads(meta.read_text(encoding="utf-8"))["classes"] == PACK.labels


def test_the_mock_only_produces_signs_the_system_knows():
    """The mock existed for months naming signs from a pack we had stopped shipping."""
    from backend.mock import SCRIPTS

    scripted = {g for script in SCRIPTS for g in script}
    assert scripted <= set(PACK.labels), sorted(scripted - set(PACK.labels))


def test_every_mock_script_assembles_into_english():
    """Otherwise the mock demonstrates glosses where a real session shows a sentence."""
    from backend.mock import SCRIPTS
    from backend.pipeline.assembler import Assembler

    for script in SCRIPTS:
        assembler = Assembler(pack=PACK)
        assert [assembler.push(g) for g in script][-1], f"{script} matches no template"


def test_every_documented_env_var_is_a_real_setting():
    """A documented variable that silently does nothing is worse than none at all."""
    known = set(Settings.model_fields)
    documented = re.findall(r"^SIGNSIGHT_([A-Z_]+)=", (ROOT / ".env.example").read_text(
        encoding="utf-8"), re.M)
    unknown = [v for v in documented if v.lower() not in known]
    assert not unknown, f".env.example documents non-settings: {unknown}"


def test_every_reference_clip_named_by_the_pack_exists():
    """The recorder shows these to people copying signs they do not know."""
    missing = [e.gloss for e in PACK.entries
               if e.reference_video and not (ROOT / e.reference_video).exists()]
    assert not missing, f"pack names reference clips that are not here: {missing}"


def test_the_feature_spec_agrees_across_python_and_typescript():
    """`tests/test_parity.py` proves the maths matches. This proves the constants do,
    which is the cheap half and fails first when someone edits one side."""
    from ml.features.extract import FACE_N, FEATURE_DIM, HAND_N, POSE_N

    ts = (ROOT / "app" / "src" / "normalise.ts").read_text(encoding="utf-8")
    for name, value in (("POSE_N", POSE_N), ("HAND_N", HAND_N), ("FACE_N", FACE_N)):
        found = re.search(rf"export const {name} = (\d+)", ts)
        assert found, f"{name} missing from normalise.ts"
        assert int(found.group(1)) == value, f"{name}: python {value}, ts {found.group(1)}"

    assert FEATURE_DIM == settings.feature_dim, "extract.py and config.py disagree"


def test_the_extension_manifest_points_at_files_that_exist():
    ext = ROOT / "extension"
    manifest = json.loads((ext / "manifest.json").read_text(encoding="utf-8"))
    referenced = [manifest["background"]["service_worker"], manifest["action"]["default_popup"]]
    referenced += [f for cs in manifest.get("content_scripts", []) for f in cs["js"]]
    missing = [r for r in referenced if not (ext / r).exists()]
    assert not missing, f"manifest.json references missing files: {missing}"


def test_documentation_links_resolve():
    """Broken links in a handover document waste someone else's afternoon."""
    broken = []
    for doc in [*(ROOT / "docs").glob("*.md"), ROOT / "README.md"]:
        for link in re.findall(r"\]\((?!https?:|#)([^)#]+)\)", doc.read_text(encoding="utf-8")):
            if not (doc.parent / link).exists():
                broken.append(f"{doc.name} -> {link}")
    assert not broken, broken


def test_manifest_clip_paths_are_sane():
    """Rows become filesystem reads, and the file is merged between machines by hand.

    Absolute paths are expected and fine: the INCLUDE corpus is 57 GB and lives outside
    the repository. What must not appear is a relative path that climbs out of it — our
    own recordings are all under ml/data/clips, and anything else is a mistake.
    """
    rows = (ROOT / "ml" / "data" / "manifest.csv").read_text(encoding="utf-8").splitlines()[1:]
    assert rows, "manifest is empty"
    for row in rows:
        clip = Path(row.split(",")[0])
        assert ".." not in clip.parts, f"path climbs out of the project: {clip}"
        if not clip.is_absolute():
            assert (ROOT / clip).resolve().is_relative_to(ROOT), clip
