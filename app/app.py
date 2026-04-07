from flask import Flask, request, jsonify, render_template
import joblib
import os
import pandas as pd
import sys
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.features.feature_builder import build_features

app = Flask(__name__)

with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

models = {}
vectorizers = {}


def make_model_compatible(model):
    if model.__class__.__name__ == "LogisticRegression" and not hasattr(model, "multi_class"):
        model.multi_class = "deprecated"
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    for estimator in getattr(model, "estimators_", []):
        if hasattr(estimator, "n_jobs"):
            estimator.n_jobs = 1
        for _, step in getattr(estimator, "steps", []):
            if hasattr(step, "n_jobs"):
                step.n_jobs = 1
    return model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    
    sender = data.get("sender", "")
    subject = data.get("subject", "")
    body = data.get("body", "")
    model_name = data.get("model", "xgboost")
    
    model = models.get(model_name)
    vectorizer = vectorizers.get(model_name)
    
    if model is None or vectorizer is None:
        return jsonify({"error": f"Model {model_name} not found"}), 404
    
    features = build_features(sender, subject, body)
    
    text = subject + " " + body
    tfidf_matrix = vectorizer.transform([text])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
    
    X = pd.concat([pd.DataFrame([features]), tfidf_df], axis=1)

    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    elif model_name == "xgboost":
        feature_names = model.get_booster().feature_names
    else:
        feature_names = None

    if feature_names:
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[feature_names]
    
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    if model_name == "xgboost":
        if prediction == 0:
            probability = 1 - probability
            label = "Legitimate"
        else:
            label = "Phishing"
    else:
        if prediction == 0:
            probability = 1 - probability
            label = "Legitimate"
        else:
            label = "Phishing"
    
    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability),
        "label": label,
        "model": model_name
    })


def load_models():
    global models, vectorizers
    model_path = os.path.join(BASE_DIR, config["paths"]["models"])
    
    models["xgboost"] = make_model_compatible(joblib.load(os.path.join(model_path, "xgboost_model.pkl")))
    vectorizers["xgboost"] = joblib.load(os.path.join(model_path, "vectorizer.pkl"))
    
    models["logistic"] = make_model_compatible(joblib.load(os.path.join(model_path, "logistic_model.pkl")))
    vectorizers["logistic"] = joblib.load(os.path.join(model_path, "logistic_vectorizer.pkl"))

    ensemble_model_path = os.path.join(model_path, "ensemble_model.pkl")
    ensemble_vectorizer_path = os.path.join(model_path, "ensemble_vectorizer.pkl")
    if os.path.exists(ensemble_model_path) and os.path.exists(ensemble_vectorizer_path):
        models["ensemble"] = make_model_compatible(joblib.load(ensemble_model_path))
        vectorizers["ensemble"] = joblib.load(ensemble_vectorizer_path)


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
