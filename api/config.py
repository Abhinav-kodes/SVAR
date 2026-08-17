import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SAMPLE_CALLS_DIR = os.path.join(REPO_ROOT, "data", "sample_calls")
DASHBOARD_DIR = os.path.join(REPO_ROOT, "dashboard")

PORT = int(os.getenv("SVAR_PORT", "8050"))
REDIS_URL = os.getenv("SVAR_REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL = os.getenv("SVAR_DATABASE_URL", "postgresql://svar:svar@localhost:5432/svar")
