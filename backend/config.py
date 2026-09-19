"""All tunables. PRD §8: no magic numbers in pipeline code, every value env-overridable."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SIGNSIGHT_", env_file=".env", extra="ignore"
    )

    # --- server ---
    # Bind to 0.0.0.0 (SIGNSIGHT_HOST=0.0.0.0) to let teammates on the same wifi record
    # into one machine's clips directory from their own laptops — their own cameras and
    # rooms, which is variation we want, with no file shuffling afterwards.
    host: str = "127.0.0.1"
    port: int = 8000
    # A private-network origin is allowed so that setup works without editing config.
    # Everything here writes to local disk and has no authentication, so do not expose
    # this host to the internet.
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    cors_allow_lan: bool = True

    # --- features (PRD §4.2) ---
    feature_dim: int = 261
    use_velocity_features: bool = False  # true → clients send 522
    target_fps: int = 15

    # --- segmentation (PRD §4.3) ---
    enter_thresh: float = 0.08  # normalised units/frame
    exit_thresh: float = 0.04  # lower than enter: hysteresis kills boundary flicker
    enter_frames: int = 3
    exit_frames: int = 4
    min_segment_frames: int = 8
    max_segment_frames: int = 45  # 3 s at 15 FPS
    energy_smoothing_frames: int = 5
    segment_resample_frames: int = 45

    # --- smoothing / gating (PRD §4.5) ---
    confidence_thresh: float = 0.75
    smoothing_window: int = 3
    repeat_cooldown_ms: int = 1200

    # --- assembly (PRD §4.6, §4.7) ---
    assembly_timeout_ms: int = 2500
    fingerspell_timeout_ms: int = 1500
    fingerspell_min_letters: int = 3

    # --- backpressure (PRD §5.3) ---
    max_queue_frames: int = 90  # 6 s at 15 FPS; oldest dropped past this

    # --- development ---
    # Fabricates glosses so the front ends can be built before a model exists.
    # Never enable for a real demo: it does not look at the landmarks.
    mock_recognition: bool = False

    # --- paths ---
    # The shipped vocabulary. isl_v1 is the specification's original 50, which the
    # available corpus covered 3 of; isl_v2_words is the 24 the data actually supports
    # and what the model, the recorder and the docs all use. Leaving the default on v1
    # meant the recorder offered 24 signs the backend would reject 20 of.
    vocab_pack: Path = ROOT / "backend" / "vocab" / "isl_v2_words.json"
    model_path: Path = ROOT / "ml" / "models" / "signsight_v1.onnx"
    db_path: Path = ROOT / "signsight.db"

    @property
    def cors_regex(self) -> str | None:
        """Any host on a private network, on the dev port. RFC 1918 ranges only."""
        if not self.cors_allow_lan:
            return None
        return (r"http://(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|"
                r"192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?")

    @property
    def expected_dim(self) -> int:
        return self.feature_dim * 2 if self.use_velocity_features else self.feature_dim


settings = Settings()
