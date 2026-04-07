import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.feature_builder import build_features_frame
from src.utils.evaluation import evaluate_model, print_metrics


def load_config():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["project_root"] = project_root
    config["paths"]["data"]["raw"] = os.path.join(project_root, config["paths"]["data"]["raw"])
    config["paths"]["models"] = os.path.join(project_root, config["paths"]["models"])
    return config


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["body", "label"])
    df["label"] = df["label"].astype(int)
    return df


def build_matrix(df_train, df_test, text_train, text_test, tfidf_config):
    features_train = build_features_frame(df_train).reset_index(drop=True)
    features_test = build_features_frame(df_test).reset_index(drop=True)
    print(f"Extracted {len(features_train.columns)} manual features")

    vectorizer = TfidfVectorizer(
        max_features=tfidf_config["max_features"],
        ngram_range=tuple(tfidf_config["ngram_range"]),
        min_df=tfidf_config["min_df"],
        max_df=tfidf_config["max_df"],
    )
    tfidf_train = vectorizer.fit_transform(text_train)
    tfidf_test = vectorizer.transform(text_test)

    tfidf_train_df = pd.DataFrame(tfidf_train.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_train.shape[1])])
    tfidf_test_df = pd.DataFrame(tfidf_test.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_test.shape[1])])

    X_train = pd.concat([features_train, tfidf_train_df], axis=1)
    X_test = pd.concat([features_test, tfidf_test_df], axis=1)
    return X_train, X_test, vectorizer


def main():
    config = load_config()
    data_path = config["paths"]["data"]["raw"]
    filename = config["data"]["filename"]
    filepath = os.path.join(data_path, filename)

    df = load_data(filepath)
    print(f"Loaded {len(df)} emails")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    text_data = df["subject"].fillna("") + " " + df["body"].fillna("")
    y = df["label"].values

    df_train, df_test, text_train, text_test, y_train, y_test = train_test_split(
        df,
        text_data,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    X_train, X_test, vectorizer = build_matrix(
        df_train,
        df_test,
        text_train,
        text_test,
        config["features"]["text"],
    )

    ensemble_config = config["models"].get("ensemble", {})
    rf_config = ensemble_config.get("random_forest", {})
    gb_config = ensemble_config.get("gradient_boosting", {})
    lr_config = ensemble_config.get("logistic_regression", {})

    model = VotingClassifier(
        estimators=[
            (
                "logistic",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=lr_config.get("max_iter", 3000),
                        C=lr_config.get("C", 1.0),
                        solver=lr_config.get("solver", "lbfgs"),
                        random_state=config["data"]["random_state"],
                        n_jobs=lr_config.get("n_jobs", -1),
                    ),
                ),
            ),
            (
                "gradient_boosting",
                GradientBoostingClassifier(
                    n_estimators=gb_config.get("n_estimators", 100),
                    learning_rate=gb_config.get("learning_rate", 0.1),
                    max_depth=gb_config.get("max_depth", 3),
                    random_state=config["data"]["random_state"],
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=rf_config.get("n_estimators", 200),
                    max_depth=rf_config.get("max_depth"),
                    min_samples_split=rf_config.get("min_samples_split", 2),
                    random_state=config["data"]["random_state"],
                    n_jobs=rf_config.get("n_jobs", -1),
                ),
            ),
        ],
        voting=ensemble_config.get("voting", "soft"),
        weights=ensemble_config.get("weights"),
        n_jobs=ensemble_config.get("n_jobs"),
    )

    print("\nTraining Ensemble: Logistic Regression + Gradient Boosting + Random Forest...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_proba)
    print_metrics(metrics)

    model_path = config["paths"]["models"]
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(model, os.path.join(model_path, "ensemble_model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_path, "ensemble_vectorizer.pkl"))
    print(f"\nModel saved to {os.path.join(model_path, 'ensemble_model.pkl')}")


if __name__ == "__main__":
    main()
