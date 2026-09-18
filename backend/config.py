"""All tunables. PRD §8: no magic numbers in pipeline code, every value env-overridable."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SIGNSIGHT_", env_file=".env", extra="ignore"
    )

    # --- server ---
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

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

    # --- paths ---
    vocab_pack: Path = ROOT / "backend" / "vocab" / "isl_v1.json"
    model_path: Path = ROOT / "ml" / "models" / "signsight_v1.onnx"
    db_path: Path = ROOT / "signsight.db"

    @property
    def expected_dim(self) -> int:
        return self.feature_dim * 2 if self.use_velocity_features else self.feature_dim


settings = Settings()
