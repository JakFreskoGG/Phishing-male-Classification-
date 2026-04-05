import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from src.utils.evaluation import evaluate_model, print_metrics


def load_config():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return {
        "paths": {"data": {"raw": os.path.join(project_root, "raw_data")}, "models": os.path.join(project_root, "models")},
        "data": {"filename": "CEAS_08.csv", "test_size": 0.2, "random_state": 42},
        "features": {"text": {"max_features": 5000, "ngram_range": [1, 2], "min_df": 2, "max_df": 0.95}},
        "models": {"xgboost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}}
    }


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["body", "label"])
    df["label"] = df["label"].astype(int)
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    from src.features.sender_features import sender_features
    from src.features.url_features import url_features
    from src.features.text_features import text_features
    from src.features.attachment_features import attachment_features

    feature_list = []
    for idx, row in df.iterrows():
        sender = row.get("sender", "")
        body = row.get("body", "")
        subject = row.get("subject", "")
        
        text = f"{subject} {body}"
        
        feats = {}
        feats.update(sender_features(sender))
        feats.update(url_features(text))
        feats.update(text_features(text))
        feats.update(attachment_features(text))
        feature_list.append(feats)
    
    return pd.DataFrame(feature_list)


def main():
    config = load_config()
    
    data_path = config["paths"]["data"]["raw"]
    filename = config["data"]["filename"]
    filepath = f"{data_path}/{filename}"
    
    df = load_data(filepath)
    print(f"Loaded {len(df)} emails")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    features_df = extract_features(df)
    features_df = features_df.drop(columns=["sender_domain"], errors="ignore")
    features_df = features_df.astype(float)
    print(f"Extracted {len(features_df.columns)} features")
    
    text_data = df["subject"].fillna("") + " " + df["body"].fillna("")
    
    tfidf_config = config["features"]["text"]
    vectorizer = TfidfVectorizer(
        max_features=tfidf_config["max_features"],
        ngram_range=tuple(tfidf_config["ngram_range"]),
        min_df=tfidf_config["min_df"],
        max_df=tfidf_config["max_df"]
    )
    tfidf_matrix = vectorizer.fit_transform(text_data)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
    
    X = pd.concat([features_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    y = df["label"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"]
    )
    
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
    joblib.dump(model, f"{model_path}/xgboost_model.pkl")
    joblib.dump(vectorizer, f"{model_path}/vectorizer.pkl")
    print(f"\nModel saved to {model_path}/xgboost_model.pkl")


if __name__ == "__main__":
    main()
