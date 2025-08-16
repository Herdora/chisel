import os
from enum import Enum

# Tracing constants
DEFAULT_TRACE_ACTIVITIES = ["CPU", "CUDA"]
TRACE_DIR = "traces"
ARTIFACTS_DIR = "artifacts"

# Environment variable keys
KANDC_BACKEND_RUN_ENV_KEY = "KANDC_BACKEND_RUN"
KANDC_JOB_ID_ENV_KEY = "KANDC_JOB_ID"
KANDC_BACKEND_APP_NAME_ENV_KEY = "KANDC_BACKEND_APP_NAME"
KANDC_TRACE_BASE_DIR_ENV_KEY = "KANDC_TRACE_BASE_DIR"

# NEW
KANDC_DISABLED_ENV_KEY = "KANDC_DISABLED"  # if set, kandc will not run

# Backend configuration
KANDC_BACKEND_URL = os.environ.get("KANDC_BACKEND_URL", "http://localhost:8000")

# Frontend configuration
KANDC_FRONTEND_URL = os.environ.get("KANDC_FRONTEND_URL", "http://localhost:3000")

"""Constants for kandc package (no GPU enum exported)."""
