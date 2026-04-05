import re
from pathlib import Path


SUSPICIOUS_EXTENSIONS = [
    ".exe", ".scr", ".bat", ".cmd", ".com", ".pif", ".msi",
    ".jar", ".js", ".vbs", ".wsf", ".sh", ".bash", ".zip",
    ".rar", ".7z", ".tar", ".gz"
]

DANGEROUS_EXTENSIONS = [".exe", ".scr", ".bat", ".cmd", ".com", ".jar", ".js", ".vbs"]


def extract_attachments(text: str) -> list:
    file_pattern = re.compile(r"[\w.-]+\.(?:{})".format("|".join(ext.replace(".", "") for ext in SUSPICIOUS_EXTENSIONS)))
    return file_pattern.findall(text.lower())


def has_attachment(text: str) -> int:
    return 1 if extract_attachments(text) else 0


def has_suspicious_attachment(text: str) -> int:
    attachments = extract_attachments(text)
    return 1 if any(Path(a).suffix.lower() in DANGEROUS_EXTENSIONS for a in attachments) else 0


def attachment_count(text: str) -> int:
    return len(extract_attachments(text))


def has_macros_keywords(text: str) -> int:
    keywords = ["macro", "enable content", "enable macros", "vba", "script"]
    text_lower = text.lower()
    return int(any(kw in text_lower for kw in keywords))


def attachment_features(text: str) -> dict:
    features = {
        "has_attachment": has_attachment(text),
        "has_suspicious_attachment": has_suspicious_attachment(text),
        "attachment_count": attachment_count(text),
        "has_macros_keywords": has_macros_keywords(text),
    }
    return features
