import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
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


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_features_frame(df)


def main():
    config = load_config()
    
    data_path = config["paths"]["data"]["raw"]
    filename = config["data"]["filename"]
    filepath = f"{data_path}/{filename}"
    
    df = load_data(filepath)
    print(f"Loaded {len(df)} emails")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    text_data = df["subject"].fillna("") + " " + df["body"].fillna("")
    y = df["label"].values

    df_train, df_test, text_train, text_test, y_train, y_test = train_test_split(
        df, text_data, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y
    )

    features_train = extract_features(df_train).reset_index(drop=True)
    features_test = extract_features(df_test).reset_index(drop=True)
    print(f"Extracted {len(features_train.columns)} features")
    
    tfidf_config = config["features"]["text"]
    vectorizer = TfidfVectorizer(
        max_features=tfidf_config["max_features"],
        ngram_range=tuple(tfidf_config["ngram_range"]),
        min_df=tfidf_config["min_df"],
        max_df=tfidf_config["max_df"]
    )
    tfidf_train = vectorizer.fit_transform(text_train)
    tfidf_test = vectorizer.transform(text_test)

    tfidf_train_df = pd.DataFrame(tfidf_train.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_train.shape[1])])
    tfidf_test_df = pd.DataFrame(tfidf_test.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_test.shape[1])])

    X_train = pd.concat([features_train, tfidf_train_df], axis=1)
    X_test = pd.concat([features_test, tfidf_test_df], axis=1)
    
    xgb_config = config["models"]["xgboost"]
    model = XGBClassifier(
        n_estimators=xgb_config["n_estimators"],
        max_depth=xgb_config["max_depth"],
        learning_rate=xgb_config["learning_rate"],
        subsample=xgb_config["subsample"],
        colsample_bytree=xgb_config["colsample_bytree"],
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=config["data"]["random_state"]
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_proba)
    print_metrics(metrics)
    
    model_path = config["paths"]["models"]
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(model, f"{model_path}/xgboost_model.pkl")
    joblib.dump(vectorizer, f"{model_path}/vectorizer.pkl")
    print(f"\nModel saved to {model_path}/xgboost_model.pkl")


if __name__ == "__main__":
    main()
