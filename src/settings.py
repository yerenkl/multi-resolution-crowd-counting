from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file_encoding="utf-8", case_sensitive=True)

    RANDOM_SEED: int = 42

    DATA_DIR: Path = Path("/dtu/blackhole/02/137570/MultiRes")
    RESULTS_DIR: Path = Path("results")

    DANN_CHECKPOINTS_DIR: Path = Path("/work3/s225224/multi-resolution-crowd-counting/checkpoints/dann")

    CLIP_EBC_DIR: Path = Path("/dtu/blackhole/0a/224426/CLIP-EBC-main")
    CLIP_EBC_WEIGHTS: Path = Path("/dtu/blackhole/0a/224426/best_mae.pth")
    NWPU_DOWNSCALED_DIR: Path = Path("/dtu/blackhole/0a/224426/NWPU_downscaled")

    @property
    def nwpu_dir(self) -> Path:
        return self.DATA_DIR / "NWPU_crowd"

    @property
    def zoom_pairs_dir(self) -> Path:
        return self.DATA_DIR / "test"


settings = Settings()
