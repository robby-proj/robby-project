import os, json, time, math, sqlite3
from datetime import datetime, timezone
import paho.mqtt.client as mqtt

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import IsolationForest


# -----------------------------
# Environment
# -----------------------------
MQTT_HOST = os.getenv("MQTT_HOST", "mqtt")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_SUB  = os.getenv("MQTT_SUB", "opta/+/+/+/telemetry")

MQTT_RECO_TOPIC = os.getenv("MQTT_PUB", "coach/recipe/recommendations")
SUMMARY_TOPIC   = os.getenv("SUMMARY_TOPIC", "coach/summary/top_consumers")

STATE_DB = os.getenv("STATE_DB", "/app/state/state.sqlite")
CONFIG_PATH = os.getenv("CONFIG_PATH", "/app/config.json")
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")

os.makedirs(os.path.dirname(STATE_DB), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------
# Config defaults
# -----------------------------
DEFAULT_CFG = {
    # rules
    "quiet_hours": [0, 1, 2, 3, 4],
    "min_power_w_quiet": 30.0,
    "min_delta_kwh_trigger": 0.001,
    "cooldown_minutes": 60,

    # baseline z rule (kept)
    "z_threshold": 3.0,
    "baseline_min_samples": 10,

    # cost projection
    "waste_hours_per_day": 2.0,
    "fix_effectiveness": 0.9,

    # summary
    "active_power_w": 30.0,
    "summary_every_sec": 300,
    "top_n": 5,
    "verbose": True,

    # ML
    "ml_enabled": True,
    "ml_contamination": 0.02,           # expected outliers (2%)
    "ml_min_samples": 200,              # per circuit before training
    "ml_train_every_sec": 1800,         # retrain every 30 min
    "ml_train_window_hours": 48,        # train on last 48 hours
    "ml_score_threshold": 0.75,         # 0..1 (higher = stricter)
    "ml_feature_roll_minutes": 15,      # rolling window for mean/std
}

def load_cfg():
    cfg = dict(DEFAULT_CFG)
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                cfg.update(json.load(f))
        else:
            print(f"[WARN] config not found at {CONFIG_PATH}. Using defaults.", flush=True)
    except Exception as e:
        print(f"[WARN] config load failed: {e}. Using defaults.", flush=True)
    return cfg

CFG = load_cfg()

QUIET_HOURS = set(CFG["quiet_hours"])
MIN_POWER_W_QUIET = float(CFG["min_power_w_quiet"])
MIN_DELTA_KWH_TRIGGER = float(CFG["min_delta_kwh_trigger"])
COOLDOWN_MIN = int(CFG["cooldown_minutes"])

Z_THRESHOLD = float(CFG["z_threshold"])
BASELINE_MIN_SAMPLES = int(CFG["baseline_min_samples"])

WASTE_HOURS_PER_DAY = float(CFG["waste_hours_per_day"])
FIX_EFFECTIVENESS = float(CFG["fix_effectiveness"])

ACTIVE_POWER_W = float(CFG["active_power_w"])
SUMMARY_EVERY_SEC = int(CFG["summary_every_sec"])
TOP_N = int(CFG["top_n"])
VERBOSE = bool(CFG["verbose"])

ML_ENABLED = bool(CFG["ml_enabled"])
ML_CONTAMINATION = float(CFG["ml_contamination"])
ML_MIN_SAMPLES = int(CFG["ml_min_samples"])
ML_TRAIN_EVERY_SEC = int(CFG["ml_train_every_sec"])
ML_TRAIN_WINDOW_HOURS = int(CFG["ml_train_window_hours"])
ML_SCORE_THRESHOLD = float(CFG["ml_score_threshold"])
ML_FEATURE_ROLL_MIN = int(CFG["ml_feature_roll_minutes"])


def log(*args):
    if VERBOSE:
        print(*args, flush=True)


# -----------------------------
# Time helpers
# -----------------------------
MIN_VALID_EPOCH = 1700000000  # ~2023-11

def normalize_ts(ts_raw, fallback_now=None) -> int:
    if fallback_now is None:
        fallback_now = int(time.time())
    try:
        ts = int(ts_raw)
    except Exception:
        return fallback_now
    if ts <= 0:
        return fallback_now
    if ts > 10_000_000_000:  # probably ms
        ts //= 1000
    if ts < MIN_VALID_EPOCH:
        return fallback_now
    return ts

def utc_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def hour_of_day(ts: int) -> int:
    return datetime.fromtimestamp(ts, tz=timezone.utc).hour

def day_utc(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

def dow_utc(ts: int) -> int:
    return int(datetime.fromtimestamp(ts, tz=timezone.utc).weekday())


# -----------------------------
# MQTT topic parsing
# -----------------------------
def parse_topic(topic: str):
    parts = topic.split("/")
    # opta/<site>/<panel>/<circuit>/telemetry
    if len(parts) < 5 or parts[0] != "opta":
        raise ValueError(f"Bad topic format: {topic}")
    return parts[1], parts[2], parts[3]


# -----------------------------
# SQLite schema
# -----------------------------
def db():
    conn = sqlite3.connect(STATE_DB)
    conn.execute("""
      CREATE TABLE IF NOT EXISTS last_reading(
        site TEXT, panel TEXT, circuit TEXT,
        last_total_kwh REAL,
        last_ts INTEGER,
        last_recipe_ts INTEGER,
        PRIMARY KEY(site,panel,circuit)
      )
    """)
    conn.execute("""
      CREATE TABLE IF NOT EXISTS baseline_hour(
        site TEXT, panel TEXT, circuit TEXT,
        hour INTEGER,
        n INTEGER,
        mean_kw REAL,
        m2 REAL,
        PRIMARY KEY(site,panel,circuit,hour)
      )
    """)
    conn.execute("""
      CREATE TABLE IF NOT EXISTS daily_energy(
        day_utc TEXT,
        site TEXT, panel TEXT, circuit TEXT,
        kwh REAL DEFAULT 0,
        cost_usd REAL DEFAULT 0,
        active_seconds INTEGER DEFAULT 0,
        updated_ts INTEGER,
        PRIMARY KEY(day_utc, site, panel, circuit)
      )
    """)
    # NEW: raw feature store for ML
    conn.execute("""
      CREATE TABLE IF NOT EXISTS ml_samples(
        ts INTEGER,
        site TEXT, panel TEXT, circuit TEXT,
        kw REAL,
        power_w REAL,
        delta_kwh REAL,
        dollars_per_kwh REAL,
        hour INTEGER,
        dow INTEGER,
        PRIMARY KEY(ts, site, panel, circuit)
      )
    """)
    # NEW: last train timestamp per circuit
    conn.execute("""
      CREATE TABLE IF NOT EXISTS ml_train_state(
        site TEXT, panel TEXT, circuit TEXT,
        last_train_ts INTEGER,
        PRIMARY KEY(site,panel,circuit)
      )
    """)
    conn.commit()
    return conn


# -----------------------------
# Baseline (Welford) kept for rule #2
# -----------------------------
def welford_update(n, mean, m2, x):
    n2 = n + 1
    delta = x - mean
    mean2 = mean + delta / n2
    delta2 = x - mean2
    m2_2 = m2 + delta * delta2
    return n2, mean2, m2_2

def std_from(n, m2):
    if n < 2:
        return 0.0
    return math.sqrt(m2 / (n - 1))

def baseline_get(conn, key, hr):
    site, panel, circuit = key
    row = conn.execute(
        "SELECT n, mean_kw, m2 FROM baseline_hour WHERE site=? AND panel=? AND circuit=? AND hour=?",
        (site, panel, circuit, hr)
    ).fetchone()
    if not row:
        return 0, 0.0, 0.0
    return int(row[0]), float(row[1]), float(row[2])

def baseline_set(conn, key, hr, n, mean, m2):
    site, panel, circuit = key
    conn.execute("""
      INSERT INTO baseline_hour(site,panel,circuit,hour,n,mean_kw,m2)
      VALUES(?,?,?,?,?,?,?)
      ON CONFLICT(site,panel,circuit,hour) DO UPDATE SET
        n=excluded.n, mean_kw=excluded.mean_kw, m2=excluded.m2
    """, (site, panel, circuit, hr, n, mean, m2))
    conn.commit()


# -----------------------------
# Last readings
# -----------------------------
def last_get(conn, key):
    site, panel, circuit = key
    row = conn.execute(
        "SELECT last_total_kwh, last_ts, last_recipe_ts FROM last_reading WHERE site=? AND panel=? AND circuit=?",
        (site, panel, circuit)
    ).fetchone()
    return None if not row else (float(row[0]), int(row[1]), int(row[2]) if row[2] else 0)

def last_set(conn, key, total_kwh, ts, recipe_ts=None):
    site, panel, circuit = key
    if recipe_ts is None:
        conn.execute("""
          INSERT INTO last_reading(site,panel,circuit,last_total_kwh,last_ts,last_recipe_ts)
          VALUES(?,?,?,?,?,COALESCE((SELECT last_recipe_ts FROM last_reading WHERE site=? AND panel=? AND circuit=?),0))
          ON CONFLICT(site,panel,circuit) DO UPDATE SET
            last_total_kwh=excluded.last_total_kwh,
            last_ts=excluded.last_ts
        """, (site, panel, circuit, total_kwh, ts, site, panel, circuit))
    else:
        conn.execute("""
          INSERT INTO last_reading(site,panel,circuit,last_total_kwh,last_ts,last_recipe_ts)
          VALUES(?,?,?,?,?,?)
          ON CONFLICT(site,panel,circuit) DO UPDATE SET
            last_total_kwh=excluded.last_total_kwh,
            last_ts=excluded.last_ts,
            last_recipe_ts=excluded.last_recipe_ts
        """, (site, panel, circuit, total_kwh, ts, recipe_ts))
    conn.commit()

def cooldown(last_recipe_ts, now_ts):
    return (now_ts - last_recipe_ts) < (COOLDOWN_MIN * 60)


# -----------------------------
# Daily totals accumulation
# -----------------------------
def daily_add(conn, key, ts, delta_kwh, dollars_per_kwh, dt_sec, power_w):
    site, panel, circuit = key
    d = day_utc(ts)

    delta_kwh = max(0.0, float(delta_kwh))
    dollars_per_kwh = max(0.0, float(dollars_per_kwh))
    delta_cost = delta_kwh * dollars_per_kwh

    active_sec = int(dt_sec) if float(power_w) >= ACTIVE_POWER_W else 0

    conn.execute("""
      INSERT INTO daily_energy(day_utc, site, panel, circuit, kwh, cost_usd, active_seconds, updated_ts)
      VALUES(?,?,?,?,?,?,?,?)
      ON CONFLICT(day_utc, site, panel, circuit) DO UPDATE SET
        kwh = daily_energy.kwh + excluded.kwh,
        cost_usd = daily_energy.cost_usd + excluded.cost_usd,
        active_seconds = daily_energy.active_seconds + excluded.active_seconds,
        updated_ts = excluded.updated_ts
    """, (d, site, panel, circuit, delta_kwh, delta_cost, active_sec, int(ts)))
    conn.commit()

def _rows_to_list(rows):
    out = []
    for site, panel, circuit, kwh, cost_usd, active_seconds in rows:
        out.append({
            "site": site,
            "panel": panel,
            "circuit": circuit,
            "kwh": round(float(kwh), 3),
            "cost_usd": round(float(cost_usd), 2),
            "active_hours": round(float(active_seconds) / 3600.0, 2),
        })
    return out

def publish_summary(client, conn, ts):
    d = day_utc(ts)

    top_by_kwh = conn.execute("""
      SELECT site, panel, circuit, kwh, cost_usd, active_seconds
      FROM daily_energy WHERE day_utc=? ORDER BY kwh DESC LIMIT ?
    """, (d, TOP_N)).fetchall()

    top_by_cost = conn.execute("""
      SELECT site, panel, circuit, kwh, cost_usd, active_seconds
      FROM daily_energy WHERE day_utc=? ORDER BY cost_usd DESC LIMIT ?
    """, (d, TOP_N)).fetchall()

    top_by_active = conn.execute("""
      SELECT site, panel, circuit, kwh, cost_usd, active_seconds
      FROM daily_energy WHERE day_utc=? ORDER BY active_seconds DESC LIMIT ?
    """, (d, TOP_N)).fetchall()

    payload = {
        "ts": utc_iso(ts),
        "window": "today_utc",
        "top_by_kwh": _rows_to_list(top_by_kwh),
        "top_by_cost": _rows_to_list(top_by_cost),
        "top_by_active_hours": _rows_to_list(top_by_active),
    }

    client.publish(SUMMARY_TOPIC, json.dumps(payload), qos=0, retain=True)
    log("PUBLISHED SUMMARY ->", SUMMARY_TOPIC, "day", d)


# -----------------------------
# ML store + training + scoring
# -----------------------------
FEATURE_COLS = ["kw", "power_w", "delta_kwh", "dollars_per_kwh", "hour", "dow", "roll_mean_kw", "roll_std_kw"]

def model_path(site, panel, circuit) -> str:
    safe = f"{site}__{panel}__{circuit}".replace("/", "_")
    return os.path.join(MODELS_DIR, f"{safe}.joblib")

def ml_insert_sample(conn, key, ts, kw, power_w, delta_kwh, dollars_per_kwh, hr, dow):
    site, panel, circuit = key
    conn.execute("""
      INSERT OR IGNORE INTO ml_samples(ts,site,panel,circuit,kw,power_w,delta_kwh,dollars_per_kwh,hour,dow)
      VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (int(ts), site, panel, circuit, float(kw), float(power_w), float(delta_kwh), float(dollars_per_kwh), int(hr), int(dow)))
    conn.commit()

def ml_get_sample_count(conn, key, since_ts=None):
    site, panel, circuit = key
    if since_ts is None:
        row = conn.execute("""
          SELECT COUNT(*) FROM ml_samples WHERE site=? AND panel=? AND circuit=?
        """, (site, panel, circuit)).fetchone()
    else:
        row = conn.execute("""
          SELECT COUNT(*) FROM ml_samples WHERE site=? AND panel=? AND circuit=? AND ts>=?
        """, (site, panel, circuit, int(since_ts))).fetchone()
    return int(row[0]) if row else 0

def ml_get_last_train_ts(conn, key):
    site, panel, circuit = key
    row = conn.execute("""
      SELECT last_train_ts FROM ml_train_state WHERE site=? AND panel=? AND circuit=?
    """, (site, panel, circuit)).fetchone()
    return int(row[0]) if row and row[0] else 0

def ml_set_last_train_ts(conn, key, ts):
    site, panel, circuit = key
    conn.execute("""
      INSERT INTO ml_train_state(site,panel,circuit,last_train_ts)
      VALUES(?,?,?,?)
      ON CONFLICT(site,panel,circuit) DO UPDATE SET last_train_ts=excluded.last_train_ts
    """, (site, panel, circuit, int(ts)))
    conn.commit()

def ml_load_training_df(conn, key, now_ts):
    """
    Load last N hours, compute rolling mean/std on kw.
    """
    site, panel, circuit = key
    since_ts = int(now_ts - ML_TRAIN_WINDOW_HOURS * 3600)
    rows = conn.execute("""
      SELECT ts, kw, power_w, delta_kwh, dollars_per_kwh, hour, dow
      FROM ml_samples
      WHERE site=? AND panel=? AND circuit=? AND ts>=?
      ORDER BY ts ASC
    """, (site, panel, circuit, since_ts)).fetchall()

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["ts","kw","power_w","delta_kwh","dollars_per_kwh","hour","dow"])
    # rolling features (time-based using samples count approximation)
    # approximate "N minutes" by assuming ~1 sample/sec? No. Use 15 min window in seconds.
    # We'll use a fixed number of samples based on typical publish rate.
    # If you publish every 5s, 15min ~= 180 samples.
    # If you publish every 1s, 15min ~= 900 samples.
    # We'll estimate from median delta ts.
    if len(df) >= 5:
        dt = df["ts"].diff().median()
        dt = max(1.0, float(dt))
        win = int((ML_FEATURE_ROLL_MIN * 60) / dt)
        win = max(5, min(win, 2000))
    else:
        win = 60

    df["roll_mean_kw"] = df["kw"].rolling(window=win, min_periods=max(5, win//5)).mean()
    df["roll_std_kw"] = df["kw"].rolling(window=win, min_periods=max(5, win//5)).std().fillna(0.0)

    df = df.dropna(subset=["roll_mean_kw"])  # remove early rows without rolling mean
    if df.empty:
        return None
    return df

def ml_train_if_due(conn, key, now_ts):
    if not ML_ENABLED:
        return

    last_train = ml_get_last_train_ts(conn, key)
    if (now_ts - last_train) < ML_TRAIN_EVERY_SEC:
        return

    # require minimum samples in window
    since_ts = int(now_ts - ML_TRAIN_WINDOW_HOURS * 3600)
    n = ml_get_sample_count(conn, key, since_ts=since_ts)
    if n < ML_MIN_SAMPLES:
        return

    df = ml_load_training_df(conn, key, now_ts)
    if df is None or len(df) < ML_MIN_SAMPLES:
        return

    X = df[FEATURE_COLS].astype(float).to_numpy()

    model = IsolationForest(
        n_estimators=200,
        contamination=ML_CONTAMINATION,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X)

    dump(model, model_path(*key))
    ml_set_last_train_ts(conn, key, now_ts)
    log("ML TRAINED:", key, "samples=", len(df), "saved=", model_path(*key))

def ml_score(conn, key, now_ts, feature_row: dict):
    """
    Returns anomaly_score in 0..1 (higher = more anomalous), or None if no model yet.
    """
    path = model_path(*key)
    if not os.path.exists(path):
        return None

    try:
        model = load(path)
    except Exception as e:
        log("ML load failed:", key, e)
        return None

    x = np.array([[float(feature_row[c]) for c in FEATURE_COLS]], dtype=float)

    # IsolationForest decision_function: higher = more normal
    # We'll convert to anomaly score 0..1
    try:
        normality = float(model.decision_function(x)[0])  # ~ [-0.5..0.5]
    except Exception:
        return None

    # Map roughly: normality high => anomaly low
    # Clamp to reasonable scale
    # Convert with a logistic-like squashing
    score = 1.0 / (1.0 + math.exp(6.0 * normality))  # tune factor=6
    return float(score)


# -----------------------------
# Cost projection
# -----------------------------
def estimate_weekly_cost(excess_kw: float, dollars_per_kwh: float):
    excess_kw = max(0.0, float(excess_kw))
    dpk = max(0.0, float(dollars_per_kwh))

    weekly_extra_kwh = excess_kw * WASTE_HOURS_PER_DAY * 7.0
    weekly_waste_cost = weekly_extra_kwh * dpk
    weekly_savings = weekly_waste_cost * max(0.0, min(1.0, FIX_EFFECTIVENESS))

    return weekly_extra_kwh, weekly_waste_cost, weekly_savings


def publish_recommendation(client, payload: dict):
    client.publish(MQTT_RECO_TOPIC, json.dumps(payload), qos=0, retain=False)
    log("PUBLISHED RECO ->", MQTT_RECO_TOPIC, payload.get("rule_id"), payload.get("summary"))


# -----------------------------
# Main MQTT callback
# -----------------------------
def on_message(client, userdata, msg):
    conn = userdata["conn"]
    try:
        site, panel, circuit = parse_topic(msg.topic)
        key = (site, panel, circuit)

        data = json.loads(msg.payload.decode("utf-8"))

        now_ts = int(time.time())
        ts = normalize_ts(data.get("ts", now_ts), fallback_now=now_ts)

        power_w = float(data.get("power_W", 0.0))
        total_kwh = float(data.get("totalEnergy_kWh", 0.0))
        dollars_per_kwh = float(data.get("dollars_per_kWh", data.get("dollars_per_kwh", 0.0)))

        kw = float(data.get("kW", power_w / 1000.0))
        hr = hour_of_day(ts)
        dow = dow_utc(ts)

        prev = last_get(conn, key)
        if prev is None:
            last_set(conn, key, total_kwh, ts)
            ml_insert_sample(conn, key, ts, kw, power_w, 0.0, dollars_per_kwh, hr, dow)
            log("INIT", msg.topic, "kWh=", total_kwh)
            return

        last_total_kwh, last_ts, last_recipe_ts = prev
        delta_kwh = total_kwh - last_total_kwh

        if delta_kwh < 0:
            last_set(conn, key, total_kwh, ts)
            ml_insert_sample(conn, key, ts, kw, power_w, 0.0, dollars_per_kwh, hr, dow)
            log("RESET/ROLLOVER", msg.topic, "kWh=", total_kwh)
            return

        dt_sec = max(1, ts - last_ts)

        # Daily totals
        daily_add(conn, key, ts, delta_kwh, dollars_per_kwh, dt_sec, power_w)

        # Baseline stats (rule #2 fallback)
        n, mean_kw, m2 = baseline_get(conn, key, hr)
        n2, mean2, m2_2 = welford_update(n, mean_kw, m2, kw)
        baseline_set(conn, key, hr, n2, mean2, m2_2)

        # periodic summary
        last_sum = userdata.get("last_summary_ts", 0)
        if (ts - last_sum) >= SUMMARY_EVERY_SEC:
            publish_summary(client, conn, ts)
            userdata["last_summary_ts"] = ts

        # store ML sample
        ml_insert_sample(conn, key, ts, kw, power_w, delta_kwh, dollars_per_kwh, hr, dow)

        # Train ML when due
        ml_train_if_due(conn, key, ts)

        # Cooldown for recos
        if cooldown(last_recipe_ts, ts):
            last_set(conn, key, total_kwh, ts)
            return

        # Trigger only when energy changed enough
        if delta_kwh < MIN_DELTA_KWH_TRIGGER:
            last_set(conn, key, total_kwh, ts)
            return

        # Build feature row for scoring
        df_tail = ml_load_training_df(conn, key, ts)
        # If df_tail exists, take last rolling features; else approximate with kw only
        if df_tail is not None and not df_tail.empty:
            last_row = df_tail.iloc[-1]
            feat = {
                "kw": float(kw),
                "power_w": float(power_w),
                "delta_kwh": float(delta_kwh),
                "dollars_per_kwh": float(dollars_per_kwh),
                "hour": int(hr),
                "dow": int(dow),
                "roll_mean_kw": float(last_row["roll_mean_kw"]),
                "roll_std_kw": float(last_row["roll_std_kw"]),
            }
        else:
            feat = {
                "kw": float(kw),
                "power_w": float(power_w),
                "delta_kwh": float(delta_kwh),
                "dollars_per_kwh": float(dollars_per_kwh),
                "hour": int(hr),
                "dow": int(dow),
                "roll_mean_kw": float(kw),
                "roll_std_kw": 0.0,
            }

        # -----------------------------
        # Rule A: ML anomaly
        # -----------------------------
        if ML_ENABLED:
            a = ml_score(conn, key, ts, feat)
            if a is not None and a >= ML_SCORE_THRESHOLD:
                # Estimate excess vs rolling mean
                excess_kw = max(0.0, feat["kw"] - feat["roll_mean_kw"])
                weekly_kwh, weekly_cost, weekly_savings = estimate_weekly_cost(excess_kw, dollars_per_kwh)

                payload = {
                    "ts": utc_iso(ts),
                    "severity": "high" if a >= min(0.95, ML_SCORE_THRESHOLD + 0.15) else "medium",
                    "rule_id": "ML_ANOMALY",
                    "site": site, "panel": panel, "circuit": circuit,
                    "recommendation": (
                        f"Anomalous consumption detected on '{circuit}'. "
                        f"Current load is unusual for this time."
                    ),
                    "evidence": {
                        "anomaly_score": round(a, 3),
                        "kw_now": round(kw, 3),
                        "kw_roll_mean": round(feat["roll_mean_kw"], 3),
                        "kw_roll_std": round(feat["roll_std_kw"], 3),
                        "delta_kWh": round(delta_kwh, 6),
                        "dollars_per_kWh_now": round(dollars_per_kwh, 5),
                    },
                    "cost_impact": {
                        "excess_kw_est": round(excess_kw, 3),
                        "assumed_waste_hours_per_day": WASTE_HOURS_PER_DAY,
                        "weekly_extra_kwh_est": round(weekly_kwh, 3),
                        "weekly_cost_if_no_action_usd": round(weekly_cost, 2),
                        "weekly_savings_if_fixed_usd": round(weekly_savings, 2),
                        "assumed_fix_effectiveness": FIX_EFFECTIVENESS,
                    },
                    "actions": [
                        "Check what turned ON recently (schedule/automation).",
                        "Look for stuck relays/contactors or timers.",
                        "If this is off-hours, add an interlock or cutoff schedule."
                    ],
                    "topics": {"telemetry": msg.topic, "recommendations": MQTT_RECO_TOPIC}
                }
                publish_recommendation(client, payload)
                last_set(conn, key, total_kwh, ts, recipe_ts=ts)
                return

        # -----------------------------
        # Rule B: Quiet hours misbehavior (existing)
        # -----------------------------
        if hr in QUIET_HOURS and power_w >= MIN_POWER_W_QUIET:
            excess_kw = kw
            weekly_kwh, weekly_cost, weekly_savings = estimate_weekly_cost(excess_kw, dollars_per_kwh)

            payload = {
                "ts": utc_iso(ts),
                "severity": "medium",
                "rule_id": "QUIET_HOURS_LOAD",
                "site": site, "panel": panel, "circuit": circuit,
                "recommendation": f"{circuit} consuming power during quiet hours (hour={hr}).",
                "evidence": {
                    "power_W": round(power_w, 2),
                    "kw_now": round(kw, 3),
                    "delta_kWh": round(delta_kwh, 6),
                    "dollars_per_kWh_now": round(dollars_per_kwh, 5),
                },
                "cost_impact": {
                    "excess_kw_est": round(excess_kw, 3),
                    "assumed_waste_hours_per_day": WASTE_HOURS_PER_DAY,
                    "weekly_extra_kwh_est": round(weekly_kwh, 3),
                    "weekly_cost_if_no_action_usd": round(weekly_cost, 2),
                    "weekly_savings_if_fixed_usd": round(weekly_savings, 2),
                    "assumed_fix_effectiveness": FIX_EFFECTIVENESS
                },
                "actions": [
                    "Verify schedule / automation is correct.",
                    "Check timers/relays for false overnight triggers.",
                    "Add an off-hours cutoff schedule if appropriate."
                ],
                "topics": {"telemetry": msg.topic, "recommendations": MQTT_RECO_TOPIC}
            }
            publish_recommendation(client, payload)
            last_set(conn, key, total_kwh, ts, recipe_ts=ts)
            return

        # -----------------------------
        # Rule C: Spike vs baseline (existing fallback)
        # -----------------------------
        st = std_from(n2, m2_2)
        if st > 0 and n2 >= BASELINE_MIN_SAMPLES:
            z = (kw - mean2) / st
            if z >= Z_THRESHOLD:
                excess_kw = max(0.0, kw - mean2)
                weekly_kwh, weekly_cost, weekly_savings = estimate_weekly_cost(excess_kw, dollars_per_kwh)

                payload = {
                    "ts": utc_iso(ts),
                    "severity": "high" if z > (Z_THRESHOLD + 1.5) else "medium",
                    "rule_id": "HOURLY_SPIKE",
                    "site": site, "panel": panel, "circuit": circuit,
                    "recommendation": f"{circuit} higher than normal for this hour (z={z:.2f}).",
                    "evidence": {
                        "kw_now": round(kw, 3),
                        "kw_hourly_mean": round(mean2, 3),
                        "z_score": round(z, 2),
                        "delta_kWh": round(delta_kwh, 6),
                        "dollars_per_kWh_now": round(dollars_per_kwh, 5),
                    },
                    "cost_impact": {
                        "excess_kw_est": round(excess_kw, 3),
                        "assumed_waste_hours_per_day": WASTE_HOURS_PER_DAY,
                        "weekly_extra_kwh_est": round(weekly_kwh, 3),
                        "weekly_cost_if_no_action_usd": round(weekly_cost, 2),
                        "weekly_savings_if_fixed_usd": round(weekly_savings, 2),
                        "assumed_fix_effectiveness": FIX_EFFECTIVENESS
                    },
                    "actions": [
                        "Check what equipment is running now vs normal schedule.",
                        "Look for stuck relays/contactors or timer misconfiguration.",
                        "If repeatable, add schedule/interlock to cap runtime."
                    ],
                    "topics": {"telemetry": msg.topic, "recommendations": MQTT_RECO_TOPIC}
                }
                publish_recommendation(client, payload)
                last_set(conn, key, total_kwh, ts, recipe_ts=ts)
                return

        # update last state
        last_set(conn, key, total_kwh, ts)
        log("RX", msg.topic, "kW=", round(kw, 3), "ΔkWh=", round(delta_kwh, 6))

    except Exception as e:
        print("ERR:", e, flush=True)


def main():
    conn = db()
    client = mqtt.Client(userdata={"conn": conn, "last_summary_ts": 0})
    client.on_message = on_message

    def on_connect(c, userdata, flags, rc):
        print("MQTT connected rc=", rc, "subscribing:", MQTT_SUB, flush=True)
        c.subscribe(MQTT_SUB)

    client.on_connect = on_connect

    while True:
        try:
            print(f"Connecting to MQTT broker {MQTT_HOST}:{MQTT_PORT} ...", flush=True)
            client.connect(MQTT_HOST, MQTT_PORT, 60)
            break
        except Exception as e:
            print("MQTT connect failed, retrying in 3s:", e, flush=True)
            time.sleep(3)

    print("Energy coach subscribed to:", MQTT_SUB, flush=True)
    print("Publishing recommendations to:", MQTT_RECO_TOPIC, flush=True)
    print("Publishing summaries to:", SUMMARY_TOPIC, f"every {SUMMARY_EVERY_SEC}s", flush=True)
    print("ML enabled:", ML_ENABLED, "score_threshold:", ML_SCORE_THRESHOLD, "train_every_sec:", ML_TRAIN_EVERY_SEC, flush=True)

    client.loop_forever()


if __name__ == "__main__":
    main()
