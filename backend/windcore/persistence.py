import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import inspect
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo
import gridstatus

# -----------------------------
# Persistence
# -----------------------------
def load_projects_from_disk() -> dict:
    if PROJECTS_FILE.exists():
        try:
            return json.loads(PROJECTS_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_projects_to_disk(projects: dict) -> None:
    PROJECTS_FILE.write_text(json.dumps(projects, indent=2, sort_keys=True))