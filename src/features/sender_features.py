import re
from urllib.parse import urlparse


FREE_EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "mail.ru", "yandex.ru", "rambler.ru", "icloud.com"
]

SUSPICIOUS_TLDS = [
    ".xyz", ".top", ".club", ".win", ".click", ".loan",
    ".work", ".tk", ".ml", ".ga", ".cf", ".gq"
]


def extract_sender_domain(sender: str) -> str:
    if not sender:
        return ""
    match = re.search(r"@([\w.-]+)", sender)
    return match.group(1) if match else ""


def is_free_email(sender: str) -> int:
    domain = extract_sender_domain(sender).lower()
    return 1 if domain in FREE_EMAIL_DOMAINS else 0


def sender_features(sender: str) -> dict:
    domain = extract_sender_domain(sender)
    return {
        "sender_domain": domain,
        "is_free_email": is_free_email(sender),
        "domain_length": len(domain),
        "has_numbers_in_domain": int(bool(re.search(r"\d", domain))),
    }
