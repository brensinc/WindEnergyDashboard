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
# Timezone helpers
# -----------------------------
def get_tz_from_latlon(lat: float, lon: float) -> ZoneInfo:
    tf = TimezoneFinder()
    tzname = tf.timezone_at(lat=lat, lng=lon)
    return ZoneInfo(tzname) if tzname else ZoneInfo("UTC")

def _ensure_tzaware(dt_like, tz: ZoneInfo):
    """
    Ensure datetime-like is tz-aware in tz.
    Works for: Series, DatetimeIndex, list/array of datetime-like.
    """
    x = pd.to_datetime(dt_like)

    if isinstance(x, pd.DatetimeIndex):
        if x.tz is None:
            return x.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        return x.tz_convert(tz)

    if isinstance(x, pd.Series):
        if x.dt.tz is None:
            return x.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        return x.dt.tz_convert(tz)

    s = pd.Series(x)
    if s.dt.tz is None:
        return s.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    return s.dt.tz_convert(tz)

def _localize_safely(naive_dt, tz: ZoneInfo):
    """
    Localize datetimes to tz while handling DST transitions.
    """
    s = pd.to_datetime(naive_dt)

    def _localize_idx(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if idx.tz is not None:
            return idx.tz_convert(tz)
        try:
            return idx.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            return idx.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")

    if isinstance(s, pd.DatetimeIndex):
        return _localize_idx(s)

    # Series path
    if getattr(s.dt, "tz", None) is not None:
        return s.dt.tz_convert(tz)

    try:
        return s.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        return s.dt.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
