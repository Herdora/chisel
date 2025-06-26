import os
from pathlib import Path
import toml

CONFIG_FILE = Path.home() / ".config" / "chisel" / "config.toml"
ENV_TOKEN_VAR = "CHISEL_DO_TOKEN"


def get_token() -> str | None:
    return os.getenv(ENV_TOKEN_VAR) or _load_token_from_file()


def _load_token_from_file() -> str | None:
    if CONFIG_FILE.exists():
        try:
            data = toml.load(CONFIG_FILE)
            return data.get("digitalocean", {}).get("token")
        except Exception:
            pass
    return None


def save_token(token: str) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"digitalocean": {"token": token}}
    with CONFIG_FILE.open("w") as f:
        toml.dump(data, f)
