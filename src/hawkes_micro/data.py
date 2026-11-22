"""Point process event ingestion and validation."""
import pandas as pd
import numpy as np
from .core import validate_events


def read_events(path):
    """CSV columns: time (nonnegative relative seconds), side ('buy' or 'sell')."""
    frame=pd.read_csv(path)
    if not {'time','side'}.issubset(frame):raise ValueError('Requires time and side columns')
    side=frame.side.astype(str).str.lower()
    if not side.isin(['buy','sell']).all():raise ValueError('side must be buy or sell')
    times=pd.to_numeric(frame.time,errors='raise').to_numpy(dtype=float)
    marks=side.map({'buy':0,'sell':1}).to_numpy(dtype=int)
    validate_events(times,marks,n_types=2)
    return times,marks
