import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str, stopwords: list) -> str:
    words = text.split()
    return " ".join(w for w in words if w not in stopwords)


def lemmatize_text(text: str) -> str:
    return text
