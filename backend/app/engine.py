"""Feature encoding and Keras inference for Trip Decider."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'destinations.csv'
MODEL = ROOT / 'models' / 'trip_decider_model.keras'
SCHEMA = ROOT / 'models' / 'feature_schema.json'
TYPES = ['mountains','beaches','nature','historical','adventure','spiritual','city','relaxation','wildlife','culture','food']
WEATHERS = ['cool','pleasant','warm','tropical']
STYLES = ['budget','balanced','comfort','luxury']
COMPANIONS = ['solo','couple','friends','family','group']
ACTIVITIES = ['trekking','sightseeing','photography','food','water activities','camping','wildlife','shopping','historical','relaxation']


def destinations():
    if not DATA.exists(): raise RuntimeError('Destination dataset is missing')
    df = pd.read_csv(DATA)
    if len(df) < 1 or df.name.isna().any() or df.name.duplicated().any(): raise RuntimeError('Destination dataset is empty or invalid')
    return df


def encode(user, dest):
    """Fixed schema keeps training and inference transformations identical."""
    budget = int(user.get('budget', 10000)); days = int(user.get('days', 4)); month = int(user.get('month', 6))
    types = set(user.get('types') or []); activities = set(user.get('activities') or [])
    dtypes = set(str(dest['types']).split('|')); dacts = set(str(dest['activities']).split('|'))
    features = [budget/60000, days/14, month/12, float(dest.min_cost)/60000, float(dest.max_cost)/60000,
                float(dest.min_days)/14, float(dest.max_days)/14, float(dest.crowd)/5]
    features += [float(t in types and t in dtypes) for t in TYPES]
    features += [float(t in dtypes) for t in TYPES]
    features += [float(a in activities and a in dacts) for a in ACTIVITIES]
    features += [float(a in dacts) for a in ACTIVITIES]
    features += [float(user.get('weather') == w and dest.climate == w) for w in WEATHERS]
    features += [float(user.get('style') == s) for s in STYLES]
    features += [float(user.get('companions') == c) for c in COMPANIONS]
    features += [float(month in [int(m) for m in str(dest.best_months).split('|')])]
    features += [float(user.get('avoid_crowds', False)), float(user.get('accessibility', False)),
                 float(user.get('hill_station', False)), float(user.get('road_trip', False))]
    return np.asarray(features, dtype=np.float32)


def score_label(user, dest):
    """Transparent synthetic relevance target used for educational training data."""
    x = encode(user, dest)
    # Similarity features are binary: overlap between requested and destination attributes.
    n = len(TYPES)
    type_score = max(0, x[8:8+n].sum()) / max(1, len(user.get('types') or []))
    activity_start = 8 + 2*n
    activity_score = x[activity_start:activity_start+len(ACTIVITIES)].sum()/max(1, len(user.get('activities') or []))
    budget = user.get('budget',10000); days=user.get('days',4)
    costfit = 1 if dest.min_cost <= budget else max(0, 1-(dest.min_cost-budget)/max(budget,1))
    duration = 1 if dest.min_days <= days <= dest.max_days else .55
    weather = 1 if user.get('weather') in ('', 'any', None) or dest.climate == user.get('weather') else .55
    month = x[-5]
    return float(np.clip(.32*type_score + .25*activity_score + .18*costfit + .12*duration + .08*weather + .05*month, .02, .98))


def load_model():
    if not MODEL.exists(): raise FileNotFoundError(f'Model artifact missing: {MODEL}. Run python scripts/train_model.py')
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL, compile=False)


def explain(user, d):
    lines=[]
    match = [t for t in (user.get('types') or []) if t in str(d.types).split('|')]
    acts = [a for a in (user.get('activities') or []) if a in str(d.activities).split('|')]
    if match: lines.append('Matches your interest in ' + ', '.join(match))
    if acts: lines.append('Offers ' + ', '.join(acts))
    budget=user.get('budget',10000)
    lines.append('Estimated cost is within your budget' if d.min_cost <= budget else f'Estimated minimum cost ₹{int(d.min_cost):,} may exceed your budget')
    lines.append('Climate matches your preference' if user.get('weather') in (None,'any','',d.climate) else f'Climate is {d.climate}, rather than your preferred {user["weather"]}')
    days=user.get('days',4)
    lines.append('Suggested duration fits your trip' if d.min_days <= days <= d.max_days else f'Suggested stay is {int(d.min_days)}–{int(d.max_days)} days')
    return lines
