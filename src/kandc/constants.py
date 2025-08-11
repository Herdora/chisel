import os
from enum import Enum

DEFAULT_TRACE_ACTIVITIES = ["CPU", "CUDA"]
MINIMUM_PACKAGES = ["torch", "requests", "kandc"]
# MINIMUM_PACKAGES = ["torch", "requests", "git+https://github.com/Herdora/kandc.git@dev"]
ENV_CACHY_ENABLED = "CACHY_ENABLED"

TRACE_DIR = "traces"
REPO_NAME = "kandc"

KANDC_BACKEND_RUN_ENV_KEY = "KANDC_BACKEND_RUN"
KANDC_JOB_ID_ENV_KEY = "KANDC_JOB_ID"
KANDC_API_KEY_ENV_KEY = "KANDC_API_KEY"
KANDC_BACKEND_URL_ENV_KEY = "KANDC_BACKEND_URL"
KANDC_BACKEND_APP_NAME_ENV_KEY = "KANDC_BACKEND_APP_NAME"
KANDC_TRACE_BASE_DIR_ENV_KEY = "KANDC_TRACE_BASE_DIR"

# Default to production API, use localhost only in development
if os.environ.get("KANDC_DEV"):
    KANDC_BACKEND_URL = "http://localhost:8000"
else:
    KANDC_BACKEND_URL = "https://api.keysandcaches.com"


class GPUType(Enum):
    # A100 GPUs (40GB and 80GB variants)
    A100_40GB_1 = "A100:1"
    A100_40GB_2 = "A100:2"
    A100_40GB_4 = "A100:4"
    A100_40GB_8 = "A100:8"
    A100_80GB_1 = "A100-80GB:1"
    A100_80GB_2 = "A100-80GB:2"
    A100_80GB_4 = "A100-80GB:4"
    A100_80GB_8 = "A100-80GB:8"

    # H100 GPUs (up to 8)
    H100_1 = "H100:1"
    H100_2 = "H100:2"
    H100_4 = "H100:4"
    H100_8 = "H100:8"

    # L4 GPUs (up to 8)
    L4_1 = "L4:1"
    L4_2 = "L4:2"
    L4_4 = "L4:4"
    L4_8 = "L4:8"
