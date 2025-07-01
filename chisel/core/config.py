import os
from pathlib import Path

import toml

CONFIG_FILE = Path.home() / ".config" / "chisel" / "config.toml"
ENV_TOKEN_VAR = "CHISEL_DO_TOKEN"
ENV_CHISEL_TOKEN_VAR = "CHISEL_TOKEN"


def get_token() -> str | None:
    """Get DigitalOcean token (legacy method)"""
    return os.getenv(ENV_TOKEN_VAR) or _load_token_from_file()

def get_chisel_token() -> str | None:
    """Get Chisel token from env or config file"""
    return os.getenv(ENV_CHISEL_TOKEN_VAR) or _load_chisel_token_from_file()

def get_auth_mode() -> str:
    """Determine if using chisel tokens or direct DO tokens"""
    if get_chisel_token():
        return "managed"
    elif get_token():
        return "direct"
    else:
        return "none"


def _load_token_from_file() -> str | None:
    if CONFIG_FILE.exists():
        try:
            data = toml.load(CONFIG_FILE)
            return data.get("digitalocean", {}).get("token")
        except Exception:
            pass
    return None

def _load_chisel_token_from_file() -> str | None:
    if CONFIG_FILE.exists():
        try:
            data = toml.load(CONFIG_FILE)
            return data.get("chisel", {}).get("token")
        except Exception:
            pass
    return None


def save_token(token: str) -> None:
    """Save DigitalOcean token (legacy method)"""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"digitalocean": {"token": token}}
    with CONFIG_FILE.open("w") as f:
        toml.dump(data, f)

def save_chisel_token(token: str) -> None:
    """Save Chisel token to config file"""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing config
    data = {}
    if CONFIG_FILE.exists():
        try:
            data = toml.load(CONFIG_FILE)
        except Exception:
            pass
    
    # Update chisel token
    data["chisel"] = {"token": token}
    
    with CONFIG_FILE.open("w") as f:
        toml.dump(data, f)
