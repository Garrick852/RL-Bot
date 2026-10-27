"""Contracts for the self-contained planning-only dashboard."""

import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DashboardParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.elements = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "dashboard.html").read_text(encoding="utf-8")
        self.page = DashboardParser()
        self.page.feed(self.source)

    def test_page_has_no_runtime_or_external_assets(self):
        forbidden = {"script", "iframe", "object", "embed", "form", "link", "img", "video", "audio"}
        for tag, attrs in self.page.elements:
            self.assertNotIn(tag, forbidden)
            self.assertFalse(any(name.lower().startswith("on") for name in attrs))
            self.assertNotIn("src", attrs)
        for value in ("@import", "url(", "@keyframes", "animation:", "transition:"):
            self.assertNotIn(value, self.source)
        self.assertTrue(any(
            tag == "meta"
            and attrs.get("http-equiv") == "Content-Security-Policy"
            and "default-src 'none'" in attrs.get("content", "")
            for tag, attrs in self.page.elements
        ))

    def test_planning_states_are_explicit(self):
        text = " ".join(self.page.text)
        for label in (
            "DRAFT_INCOMPLETE",
            "EVIDENCE_UNCHECKED",
            "尚未執行驗證",
            "尚未選定",
            "未啟動",
            "不適用",
            "離線審計報告",
            "No active runtime records.",
            "未確認提案",
        ):
            self.assertIn(label, text)
        self.assertEqual(text.count("規劃中／尚未執行"), 5)
        self.assertNotIn("US500.cash", text)
        self.assertNotIn("Stage 0", text)

    def test_navigation_and_accessibility_structure(self):
        ids = [
            attrs["id"] for _, attrs in self.page.elements if "id" in attrs
        ]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(tag == "main" for tag, _ in self.page.elements), 1)
        self.assertEqual(sum(tag == "h1" for tag, _ in self.page.elements), 1)
        self.assertTrue(any(tag == "html" and attrs.get("lang") == "zh-Hant"
                            for tag, attrs in self.page.elements))
        for _, attrs in self.page.elements:
            for reference in attrs.get("aria-labelledby", "").split():
                self.assertIn(reference, ids)
            if "href" in attrs:
                href = attrs["href"]
                if href.startswith("#"):
                    self.assertIn(href[1:], ids)
                else:
                    self.assertNotIn(":", href)
                    self.assertTrue((ROOT / href).is_file(), href)


if __name__ == "__main__":
    unittest.main()
