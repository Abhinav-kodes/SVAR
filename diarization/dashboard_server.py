"""Backward-compatible entry point — redirects to dashboard.dashboard_server."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dashboard.dashboard_server import main

if __name__ == "__main__":
    main()
