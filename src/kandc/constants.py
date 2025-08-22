import os

DEFAULT_TRACE_ACTIVITIES = ["CPU", "CUDA"]
TRACE_DIR = "traces"
ARTIFACTS_DIR = "artifacts"


KANDC_BACKEND_RUN_ENV_KEY = "KANDC_BACKEND_RUN"
KANDC_JOB_ID_ENV_KEY = "KANDC_JOB_ID"
KANDC_BACKEND_APP_NAME_ENV_KEY = "KANDC_BACKEND_APP_NAME"
KANDC_TRACE_BASE_DIR_ENV_KEY = "KANDC_TRACE_BASE_DIR"

KANDC_DISABLED_ENV_KEY = "KANDC_DISABLED"  # if set, kandc will not run
KANDC_PROFILER_DISABLED_ENV_KEY = "KANDC_PROFILER_DISABLED"  # if set, profiler will not run

DEV_MODE = os.getenv("DEV_MODE", "False").lower() == "true"  # Default to production mode
if DEV_MODE:
    KANDC_BACKEND_URL = "http://localhost:8000"
    KANDC_FRONTEND_URL = "http://localhost:3000"
else:
    KANDC_BACKEND_URL = "https://api.keysandcaches.com"
    KANDC_FRONTEND_URL = "https://keysandcaches.com"

KANDC_API_BASE_URL = f"{KANDC_BACKEND_URL}/api/v1"
KANDC_WEBSOCKET_URL = KANDC_BACKEND_URL.replace("http://", "ws://").replace("https://", "wss://")

KANDC_DASHBOARD_URL = f"{KANDC_FRONTEND_URL}/dashboard"
KANDC_AUTH_URL = f"{KANDC_FRONTEND_URL}/auth"
KANDC_DOCS_URL = f"{KANDC_FRONTEND_URL}/docs"

KANDC_CONTACT_EMAIL = "hello@keysandcaches.com"
KANDC_SUPPORT_EMAIL = "contact@herdora.com"


def print_config():
    """Print current configuration for debugging."""
    print("🔧 Keys & Caches Configuration:")
    print(f"   Dev Mode: {DEV_MODE}")
    print(f"   Backend URL: {KANDC_BACKEND_URL}")
    print(f"   Frontend URL: {KANDC_FRONTEND_URL}")
    print(f"   API Base URL: {KANDC_API_BASE_URL}")
    print(f"   Dashboard URL: {KANDC_DASHBOARD_URL}")
    print(f"   WebSocket URL: {KANDC_WEBSOCKET_URL}")
    print(f"   Auth URL: {KANDC_AUTH_URL}")
    print(f"   Docs URL: {KANDC_DOCS_URL}")
    print()
    print("💡 To switch modes, edit DEV_MODE in kandc/src/kandc/constants.py")


"""Constants for kandc package (no GPU enum exported)."""
