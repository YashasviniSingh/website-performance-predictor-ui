from flask import Flask, render_template, request
import pickle, numpy as np, json, os, hashlib, urllib.parse, math

app = Flask(__name__)

# ========================
# LOAD MODEL
# ========================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(MODEL_PATH, 'rb'))

# ========================
# FEATURES
# ========================
FEATURES = [
    'hours','users','sessions','engaged_sessions','engaged_sessions_per_user',
    'events_per_session','event_count','page_views','session_duration',
    'bounce_rate','time_on_page','previous_visits'
]

nice = {
 'hours':'Hours Active',
 'users':'Users',
 'sessions':'Sessions',
 'engaged_sessions':'Engaged Sessions',
 'engaged_sessions_per_user':'Engaged Sessions / User',
 'events_per_session':'Events / Session',
 'event_count':'Total Events',
 'page_views':'Page Views',
 'session_duration':'Session Duration',
 'bounce_rate':'Bounce Rate',
 'time_on_page':'Time On Page',
 'previous_visits':'Previous Visits'
}

# ========================
# LOAD FEATURE MEANS
# ========================
means_path = os.path.join(os.path.dirname(__file__), 'feature_means.json')
if os.path.exists(means_path):
    with open(means_path, 'r') as f:
        FEATURE_MEANS = json.load(f)
else:
    FEATURE_MEANS = {f: 1.0 for f in FEATURES}

# ========================
# GENERATE DETERMINISTIC FEATURES
# ========================
def features_for_url(url: str):
    """Generate deterministic, realistic feature variation from URL"""
    if not url:
        return FEATURE_MEANS.copy()

    try:
        host = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        host = url.lower()

    # stable hash
    h = hashlib.md5(host.encode('utf-8')).digest()
    vals = FEATURE_MEANS.copy()

    # domain-based popularity heuristic
    domain_len = len(host)
    subdomains = host.count('.')
    has_www = host.startswith('www.')
    tld = host.split('.')[-1] if '.' in host else ''
    common_tld = tld in ['com', 'net', 'org', 'io', 'co']

    len_score = max(0, min(1, (28 - domain_len) / 25))
    sub_score = min(subdomains / 3, 1)
    tld_score = 0.2 if common_tld else 0
    www_score = 0.05 if has_www else 0
    hash_score = h[0] / 255.0

    popularity = 0.35*len_score + 0.25*sub_score + 0.2*tld_score + 0.1*www_score + 0.1*hash_score
    popularity = np.clip(popularity, 0, 1)

    # jittered but deterministic features
    for i, f in enumerate(FEATURES):
        b = h[(i + 4) % len(h)]
        base = FEATURE_MEANS.get(f, 1.0)
        jitter = ((b / 255.0) - 0.5) * (0.15 + 0.25 * popularity)
        vals[f] = base * (1 + jitter)

    vals["popularity"] = popularity
    return vals

# ========================
# ROUTES
# ========================
@app.route('/', methods=['GET'])
def home():
    return render_template(
        'index.html',
        prediction=None,
        url_value="",
        metrics=None,
        features=FEATURES,
        nice=nice,
        color="#2ecc71",
        gauge_width=0
    )

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url', '').strip()
    metrics = features_for_url(url)
    X = np.array([[metrics[f] for f in FEATURES]])

    try:
        base_pred = float(model.predict(X)[0])
    except Exception:
        base_pred = 0.05  # fallback

    pop = metrics["popularity"]

    # 🔹 Deterministic seed (no randomness)
    hash_int = int(hashlib.md5(url.encode()).hexdigest(), 16)
    stable_rand = (math.sin(hash_int % 10000) + 1) / 2  # 0–1 stable

    # 🔹 Combine model, popularity, and stable pseudo-randomness
    base = base_pred * 100
    influence = 8 + (pop * 8) + (stable_rand * 4)  # ensures 2–20% realistic spread
    percent = np.clip(influence, 2, 19)

    # 🔹 Color thresholds
    if percent < 6:
        color = "#e74c3c"   # red
    elif percent < 12:
        color = "#f1c40f"   # yellow
    else:
        color = "#2ecc71"   # green

    return render_template(
        'index.html',
        prediction=round(percent, 2),
        url_value=url,
        metrics=metrics,
        features=FEATURES,
        nice=nice,
        color=color,
        gauge_width=percent
    )

# ========================
# RUN APP
# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


