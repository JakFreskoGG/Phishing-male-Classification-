import re
from collections import Counter


URGENCY_WORDS = [
    "urgent", "immediately", "now", "act now", "limited time",
    "expire", "deadline", "today", "hurry", "instant", "asap"
]

THREAT_WORDS = [
    "suspended", "blocked", "unauthorized", "compromised",
    "security", "alert", "warning", "fraud", "illegal", "law enforcement"
]

REWARD_WORDS = [
    "won", "prize", "gift", "free", "bonus", "congratulations",
    "winner", "reward", "claim", "receive", "get now"
]

GREETING_PATTERNS = [
    r"dear\s+user", r"dear\s+customer", r"dear\s+member",
    r"dear\s+account", r"valued\s+customer", r"dear\s+client"
]


def count_keyword_matches(text: str, keywords: list) -> int:
    text = text.lower()
    count = 0
    for keyword in sorted(keywords, key=len, reverse=True):
        pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
        matches = re.findall(pattern, text)
        count += len(matches)
        text = re.sub(pattern, " ", text)
    return count


def count_urgency_words(text: str) -> int:
    return count_keyword_matches(text, URGENCY_WORDS)


def count_threat_words(text: str) -> int:
    return count_keyword_matches(text, THREAT_WORDS)


def count_reward_words(text: str) -> int:
    return count_keyword_matches(text, REWARD_WORDS)


def has_generic_greeting(text: str) -> int:
    text = text.lower()
    return int(any(re.search(pattern, text) for pattern in GREETING_PATTERNS))


def count_capitals(text: str) -> int:
    return sum(1 for c in text if c.isupper())


def count_exclamation(text: str) -> int:
    return text.count("!")


def count_question_marks(text: str) -> int:
    return text.count("?")


def text_length(text: str) -> int:
    return len(text)


def word_count(text: str) -> int:
    return len(text.split())


def avg_word_length(text: str) -> float:
    words = text.split()
    if not words:
        return 0
    return sum(len(w) for w in words) / len(words)


def text_features(text: str) -> dict:
    text_lower = text.lower()
    features = {
        "urgency_word_count": count_urgency_words(text),
        "threat_word_count": count_threat_words(text),
        "reward_word_count": count_reward_words(text),
        "has_generic_greeting": has_generic_greeting(text),
        "capital_count": count_capitals(text),
        "exclamation_count": count_exclamation(text),
        "question_mark_count": count_question_marks(text),
        "text_length": text_length(text),
        "word_count": word_count(text),
        "avg_word_length": avg_word_length(text),
    }
    return features
