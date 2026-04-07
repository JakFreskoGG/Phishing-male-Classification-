import re
from urllib.parse import urlparse
from src.features.sender_features import SUSPICIOUS_TLDS


def extract_urls(text: str) -> list:
    url_pattern = re.compile(r"(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    return url_pattern.findall(text)


def count_urls(text: str) -> int:
    return len(extract_urls(text))


def url_length(url: str) -> int:
    return len(url)


def has_suspicious_tld(url: str) -> int:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        for tld in SUSPICIOUS_TLDS:
            if domain.endswith(tld):
                return 1
    except:
        pass
    return 0


def has_ip_in_url(url: str) -> int:
    ip_pattern = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
    return int(bool(ip_pattern.search(url)))


def count_special_chars(url: str) -> int:
    return sum(1 for c in url if c in "@#$%^&*()_+-=[]{}|;':\",./<>?")


def url_features(text: str) -> dict:
    urls = extract_urls(text)
    features = {
        "url_count": len(urls),
        "avg_url_length": sum(len(u) for u in urls) / len(urls) if urls else 0,
        "max_url_length": max(len(u) for u in urls) if urls else 0,
        "min_url_length": min(len(u) for u in urls) if urls else 0,
        "has_suspicious_tld": max(has_suspicious_tld(u) for u in urls) if urls else 0,
        "has_ip_url": max(has_ip_in_url(u) for u in urls) if urls else 0,
        "avg_special_chars": sum(count_special_chars(u) for u in urls) / len(urls) if urls else 0,
    }
    return features
