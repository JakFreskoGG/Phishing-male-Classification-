import pytest
import sys
sys.path.insert(0, ".")

from src.features.sender_features import (
    extract_sender_domain, is_free_email, sender_features
)
from src.features.url_features import (
    extract_urls, count_urls, url_length, has_suspicious_tld,
    has_ip_in_url, url_features
)
from src.features.text_features import (
    count_urgency_words, count_threat_words, count_reward_words,
    has_generic_greeting, text_features
)
from src.features.attachment_features import (
    extract_attachments, has_attachment, has_suspicious_attachment,
    attachment_features
)


class TestSenderFeatures:
    def test_extract_sender_domain(self):
        assert extract_sender_domain("user@gmail.com") == "gmail.com"
        assert extract_sender_domain("test@company.co.uk") == "company.co.uk"
        assert extract_sender_domain("invalid") == ""

    def test_is_free_email(self):
        assert is_free_email("user@gmail.com") == 1
        assert is_free_email("user@yahoo.com") == 1
        assert is_free_email("user@company.com") == 0

    def test_sender_features(self):
        feats = sender_features("test@gmail.com")
        assert "sender_domain" in feats
        assert "is_free_email" in feats
        assert feats["is_free_email"] == 1


class TestUrlFeatures:
    def test_extract_urls(self):
        text = "Visit http://example.com and www.test.org"
        urls = extract_urls(text)
        assert len(urls) == 2
        assert "http://example.com" in urls

    def test_count_urls(self):
        assert count_urls("Check http://a.com and http://b.com") == 2
        assert count_urls("No URLs here") == 0

    def test_url_length(self):
        assert url_length("http://example.com") == 18

    def test_has_suspicious_tld(self):
        assert has_suspicious_tld("http://site.xyz") == 1
        assert has_suspicious_tld("http://site.com") == 0

    def test_has_ip_in_url(self):
        assert has_ip_in_url("http://192.168.1.1") == 1
        assert has_ip_in_url("http://example.com") == 0

    def test_url_features(self):
        text = "Visit http://site.xyz for more info"
        feats = url_features(text)
        assert "url_count" in feats
        assert feats["url_count"] >= 1


class TestTextFeatures:
    def test_count_urgency_words(self):
        assert count_urgency_words("Urgent! Act now!") == 2
        assert count_urgency_words("Hello world") == 0

    def test_count_threat_words(self):
        assert count_threat_words("Account suspended!") == 1
        assert count_threat_words("Hello friend") == 0

    def test_count_reward_words(self):
        assert count_reward_words("You won a prize!") == 2
        assert count_reward_words("Regular email") == 0

    def test_has_generic_greeting(self):
        assert has_generic_greeting("Dear user, please...") == 1
        assert has_generic_greeting("Hello John") == 0

    def test_text_features(self):
        text = "URGENT: Your account will be suspended!"
        feats = text_features(text)
        assert "urgency_word_count" in feats
        assert "text_length" in feats


class TestAttachmentFeatures:
    def test_extract_attachments(self):
        text = "Please download file.exe and document.pdf"
        attachments = extract_attachments(text)
        assert "file.exe" in attachments

    def test_has_attachment(self):
        assert has_attachment("See attachment.exe") == 1
        assert has_attachment("No attachments") == 0

    def test_has_suspicious_attachment(self):
        assert has_suspicious_attachment("Download file.exe") == 1
        assert has_suspicious_attachment("Download document.pdf") == 0

    def test_attachment_features(self):
        text = "Attached file.exe"
        feats = attachment_features(text)
        assert "has_attachment" in feats
        assert feats["has_attachment"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
