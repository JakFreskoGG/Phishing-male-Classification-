from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import re

app = Flask(__name__)

config = {
    "app": {"host": "0.0.0.0", "port": 8000, "debug": True}
}

models = {}
vectorizers = {}

FREE_EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "mail.ru", "yandex.ru"]
URGENCY_WORDS = ["urgent", "immediately", "now", "act now", "limited time", "expire", "deadline", "today", "hurry", "instant", "asap"]
THREAT_WORDS = ["suspended", "blocked", "account", "unauthorized", "security", "alert", "warning", "fraud", "illegal"]
REWARD_WORDS = ["won", "prize", "gift", "free", "bonus", "congratulations", "winner", "reward", "claim", "receive"]


def extract_urls(text: str) -> list:
    return re.findall(r"http\S+|www\.\S+", text)


def extract_features(sender: str, subject: str, body: str) -> dict:
    text = f"{subject} {body}"
    
    domain = re.search(r"@([\w.-]+)", sender or "")
    domain = domain.group(1) if domain else ""
    
    urls = extract_urls(text)
    
    features = {
        "is_free_email": 1 if domain.lower() in FREE_EMAIL_DOMAINS else 0,
        "domain_length": len(domain),
        "has_numbers_in_domain": int(bool(re.search(r"\d", domain))),
        "url_count": len(urls),
        "avg_url_length": sum(len(u) for u in urls) / len(urls) if urls else 0,
        "max_url_length": max(len(u) for u in urls) if urls else 0,
        "min_url_length": min(len(u) for u in urls) if urls else 0,
        "has_suspicious_tld": int(any(u.endswith(tld) for u in urls for tld in [".xyz", ".top", ".click", ".work"])),
        "has_ip_url": int(bool(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", text))),
        "avg_special_chars": sum(sum(1 for c in u if c in "@#$%^&*()_+-=[]{}|;':\",./<>?") for u in urls) / len(urls) if urls else 0,
        "urgency_word_count": sum(1 for w in URGENCY_WORDS if w in text.lower()),
        "threat_word_count": sum(1 for w in THREAT_WORDS if w in text.lower()),
        "reward_word_count": sum(1 for w in REWARD_WORDS if w in text.lower()),
        "has_generic_greeting": int(any(p in text.lower() for p in ["dear user", "dear customer", "dear member", "valued customer"])),
        "capital_count": sum(1 for c in text if c.isupper()),
        "exclamation_count": text.count("!"),
        "question_mark_count": text.count("?"),
        "text_length": len(text),
        "word_count": len(text.split()),
        "avg_word_length": sum(len(w) for w in text.split()) / len(text.split()) if text.split() else 0,
        "has_attachment": int(bool(re.search(r"\.(exe|scr|bat|cmd|com|pif|msi|jar|js|vbs|zip|rar|7z|tar|gz)", text.lower()))),
        "has_suspicious_attachment": int(bool(re.search(r"\.(exe|scr|bat|cmd|com|jar|js|vbs)", text.lower()))),
        "attachment_count": len(re.findall(r"\.(exe|scr|bat|cmd|com|pif|msi|jar|js|vbs|zip|rar|7z|tar|gz)", text.lower())),
        "has_macros_keywords": int(any(k in text.lower() for k in ["macro", "enable content", "enable macros", "vba", "script"])),
    }
    return features


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    sender = data.get("sender", "")
    subject = data.get("subject", "")
    body = data.get("body", "")
    model_name = data.get("model", "xgboost")
    
    model = models.get(model_name)
    vectorizer = vectorizers.get(model_name)
    
    if model is None or vectorizer is None:
        return jsonify({"error": f"Model {model_name} not found"}), 404
    
    features = extract_features(sender, subject, body)
    
    text = subject + " " + body
    tfidf_matrix = vectorizer.transform([text])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
    
    X = pd.concat([pd.DataFrame([features]), tfidf_df], axis=1)
    
    if model_name == "xgboost":
        feature_names = model.get_booster().feature_names
        for col in X.columns:
            if col not in feature_names:
                X[col] = 0
        X = X[feature_names]
    
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    if model_name == "xgboost" and prediction == 0:
        probability = 1 - probability
    
    if model_name == "logistic":
        if prediction == 0:
            probability = 1 - probability
            label = "Phishing"
        else:
            label = "Legitimate"
    else:
        label = "Phishing" if prediction == 0 else "Legitimate"
    
    print(f"Debug - Model: {model_name}, Prediction: {prediction}, Prob: {probability}")
    
    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability),
        "label": label,
        "model": model_name
    })


def load_models():
    global models, vectorizers
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models")
    
    models["xgboost"] = joblib.load(os.path.join(model_path, "xgboost_model.pkl"))
    vectorizers["xgboost"] = joblib.load(os.path.join(model_path, "vectorizer.pkl"))
    
    models["logistic"] = joblib.load(os.path.join(model_path, "logistic_model.pkl"))
    vectorizers["logistic"] = joblib.load(os.path.join(model_path, "logistic_vectorizer.pkl"))


if __name__ == "__main__":
    try:
        load_models()
        print("Models loaded successfully: xgboost, logistic")
    except Exception as e:
        print(f"Warning: Models not found. Error: {e}")
    
    app_config = config.get("app", {})
    app.run(
        host=app_config.get("host", "0.0.0.0"),
        port=app_config.get("port", 8000),
        debug=app_config.get("debug", True)
    )
