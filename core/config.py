from __future__ import annotations
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
import os
BASE_DIR = Path(__file__).resolve().parent.parent  # .../app
DEFAULT_TPL_DIR = BASE_DIR / "templates"
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,

    )

    app_secret: str = "change-me"
    llm_model: str = "gpt-4o-mini"
    prompt_version: str = "v1.0"
    openai_api_key: str

    template_path: str = str(DEFAULT_TPL_DIR)

    @property
    def template_dir(self) -> str:

        p = Path(self.template_path)
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        return str(p)


settings = Settings()