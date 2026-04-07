import pandas as pd

from src.features.attachment_features import attachment_features
from src.features.sender_features import sender_features
from src.features.text_features import text_features
from src.features.url_features import url_features


def build_features(sender: str = "", subject: str = "", body: str = "") -> dict:
    text = f"{subject or ''} {body or ''}"

    features = {}
    features.update(sender_features(sender or ""))
    features.update(url_features(text))
    features.update(text_features(text))
    features.update(attachment_features(text))
    features.pop("sender_domain", None)
    return features


def build_features_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        build_features(
            sender=row.get("sender", ""),
            subject=row.get("subject", ""),
            body=row.get("body", ""),
        )
        for _, row in df.iterrows()
    ]
    return pd.DataFrame(rows).astype(float)
