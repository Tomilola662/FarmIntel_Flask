import os
import pickle
import numpy as np
import requests
from datetime import datetime, timedelta
from flask import Flask, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db

app = Flask(__name__)
CORS(app)

# 1. Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://soil-quality-data-946da-default-rtdb.firebaseio.com'
})

# 2. Load ML Model & Label Encoder
with open('crop_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

# 3. Rainfall via Open-Meteo (no API key needed)
AKURE_LAT = 7.2571
AKURE_LON = 5.2058
RAINFALL_WINDOW_DAYS = 90       # matches the ~seasonal scale of the training data
RAINFALL_CACHE_TTL = timedelta(hours=12)

# Bounds pulled directly from Crop_recommendation.csv (rainfall column).
RAINFALL_TRAIN_MIN = 20.21
RAINFALL_TRAIN_MAX = 298.56

# How many of the most recent Firebase readings to average over for the
# model input. NPK sensors are noisy and change slowly day-to-day, so a
# single instantaneous reading is not representative — this smooths it out.
RECOMMENDATION_WINDOW_SIZE = 20

# Humidity sensor isn't wired up yet — safe baseline percentage used until it is.
AVERAGE_HUMIDITY_PLACEHOLDER = 60.0

# Format the device writes into Firebase, e.g. "2026-07-16 19:17:36"
DEVICE_TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

_rainfall_cache = {'value': None, 'fetched_at': None}


def get_recent_rainfall_mm():
    """Sum of actual precipitation (mm) over the last RAINFALL_WINDOW_DAYS for Akure.
    Cached for RAINFALL_CACHE_TTL since this doesn't need to be refetched every request."""
    now = datetime.utcnow()
    cache_age = now - _rainfall_cache['fetched_at'] if _rainfall_cache['fetched_at'] else None
    if _rainfall_cache['value'] is not None and cache_age is not None and cache_age < RAINFALL_CACHE_TTL:
        return _rainfall_cache['value']

    end_date = (now - timedelta(days=1)).date()   # archive data usually lags ~1 day
    start_date = end_date - timedelta(days=RAINFALL_WINDOW_DAYS)

    try:
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": AKURE_LAT,
                "longitude": AKURE_LON,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "daily": "precipitation_sum",
                "timezone": "auto",
            },
            timeout=10,
        )
        resp.raise_for_status()
        daily_values = resp.json()['daily']['precipitation_sum']
        total_mm = sum(v for v in daily_values if v is not None)

        _rainfall_cache['value'] = total_mm
        _rainfall_cache['fetched_at'] = now
        return total_mm

    except Exception as e:
        print(f"[Weather] Failed to fetch rainfall from Open-Meteo: {e}")
        return _rainfall_cache['value']


def clip_rainfall_for_model(rainfall_raw):
    """Bounds the live rainfall reading to the range the model was trained on."""
    clipped = max(RAINFALL_TRAIN_MIN, min(rainfall_raw, RAINFALL_TRAIN_MAX))
    if clipped != rainfall_raw:
        print(f"[Weather] Clipped rainfall {rainfall_raw:.2f}mm -> {clipped:.2f}mm (outside training range "
              f"[{RAINFALL_TRAIN_MIN}, {RAINFALL_TRAIN_MAX}])")
    return clipped


def parse_device_timestamp(raw_value):
    """Parses the device's "YYYY-MM-DD HH:MM:SS" string into a datetime.
    Returns None if missing or malformed, so callers can fall back safely."""
    if not raw_value:
        return None
    try:
        return datetime.strptime(raw_value, DEVICE_TIMESTAMP_FORMAT)
    except (ValueError, TypeError):
        return None


def sorted_log_keys_by_timestamp(all_logs):
    """Sorts Firebase entries chronologically using the real device timestamp
    field, falling back to push-key lexical order for any entry missing or
    malformed timestamps (push keys are only approximately time-ordered)."""
    def sort_key(key):
        parsed = parse_device_timestamp(all_logs[key].get('timestamp'))
        # Entries with a real timestamp sort by that; anything unparsable
        # falls back to the push key itself so it doesn't crash the sort.
        return (0, parsed) if parsed else (1, key)

    return sorted(all_logs.keys(), key=sort_key)


def compute_windowed_average(all_logs, log_keys, window_size):
    """Averages nitrogen, phosphorus, potassium and temperature over the
    most recent `window_size` Firebase entries, instead of trusting a single
    (possibly noisy) latest reading."""
    recent_keys = log_keys[-window_size:]
    fields = ['nitrogen', 'phosphorus', 'potassium', 'temperature']
    sums = {f: 0.0 for f in fields}
    counts = {f: 0 for f in fields}

    for key in recent_keys:
        entry = all_logs[key]
        for f in fields:
            if f in entry and entry[f] is not None:
                try:
                    sums[f] += float(entry[f])
                    counts[f] += 1
                except (TypeError, ValueError):
                    continue

    averaged = {f: (sums[f] / counts[f]) if counts[f] > 0 else None for f in fields}
    return averaged, len(recent_keys)


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'message': 'Crop Recommendation API is running successfully!'
    })


@app.route('/api/dashboard-data', methods=['GET'])
def get_dashboard_data():
    try:
        ref = db.reference('sensor_readings')
        all_logs = ref.get()

        if not all_logs:
            return jsonify({'status': 'error', 'message': 'No data found in Firebase'}), 404

        # Sort by the real device timestamp now that it's available, instead
        # of relying on Firebase push-key ordering (which is only approximate).
        log_keys = sorted_log_keys_by_timestamp(all_logs)
        latest_key = log_keys[-1]
        latest_reading = dict(all_logs[latest_key])

        # Rainfall from Open-Meteo (real data, not hardcoded)
        rainfall_raw = get_recent_rainfall_mm()
        if rainfall_raw is None:
            return jsonify({'status': 'error', 'message': 'Unable to fetch rainfall data and no cached value available yet'}), 503

        rainfall_for_model = clip_rainfall_for_model(rainfall_raw)

        # Average the recent window instead of trusting one noisy reading.
        averaged, sample_count = compute_windowed_average(all_logs, log_keys, RECOMMENDATION_WINDOW_SIZE)

        # Fall back to the single latest reading for any field that couldn't
        # be averaged (e.g. not enough history yet). Not enforced strictly
        # for now — revisit once there's a reliable backlog of readings.
        for f in ['nitrogen', 'phosphorus', 'potassium', 'temperature']:
            if averaged[f] is None:
                try:
                    averaged[f] = float(latest_reading.get(f))
                except (TypeError, ValueError):
                    averaged[f] = 0.0

        # Merge raw rainfall + placeholder humidity into the latest reading for dashboard display.
        latest_reading['rainfall'] = rainfall_raw
        # Re-emit the device's raw timestamp as proper ISO 8601, since
        # "YYYY-MM-DD HH:MM:SS" (no 'T', no timezone) is not reliably
        # parsed by JavaScript's Date() across all browsers.
        parsed_latest_ts = parse_device_timestamp(latest_reading.get('timestamp'))
        if parsed_latest_ts:
            latest_reading['timestamp'] = parsed_latest_ts.isoformat()

        # Format data for the Random Forest model (N, P, K, temperature, humidity, rainfall — no pH)
        features = [
            averaged['nitrogen'],
            averaged['phosphorus'],
            averaged['potassium'],
            averaged['temperature'],
            averaged['humidity'],
            rainfall_for_model,
        ]

        prediction_numeric = model.predict(np.array([features]))
        recommended_crop = le.inverse_transform(prediction_numeric)[0]

        # Firebase RTDB returns a dict keyed by push-id, not a list — convert
        # it to a chronological array (by real timestamp) and stamp
        # rainfall/humidity onto every entry, since those two aren't logged
        # per-sensor-reading. Timestamps are normalized to ISO 8601 for the
        # frontend charts.
        history = []
        for key in log_keys:
            entry = dict(all_logs[key])
            entry.setdefault('rainfall', rainfall_raw)

            parsed_ts = parse_device_timestamp(entry.get('timestamp'))
            entry['timestamp'] = parsed_ts.isoformat() if parsed_ts else key

            history.append(entry)

        return jsonify({
            'status': 'success',
            'latest_reading': latest_reading,
            'averaged_reading_used_for_prediction': {
                'nitrogen': averaged['nitrogen'],
                'phosphorus': averaged['phosphorus'],
                'potassium': averaged['potassium'],
                'temperature': averaged['temperature'],
                'humidity':   averaged['humidity'],
                'sample_size': sample_count,
            },
            'rainfall_mm_last_90_days': rainfall_raw,
            'rainfall_used_for_prediction': rainfall_for_model,
            'recommended_crop': recommended_crop,
            'history': history
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)