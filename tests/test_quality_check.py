import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'starter_code'))
from quality_check import check_record, check_batch, run_semantic_checks

# A fully valid record used as base for most tests
VALID = {
    "document_id": "doc-001",
    "source_type": "PDF",
    "author": "Dr. Smith",
    "category": "Machine Learning",
    "content": (
        "This is a sufficiently long content string that passes all quality checks. "
        "It discusses data engineering and vector databases in detail."
    ),
    "timestamp": "2024-01-15T09:00:00Z",
}


def _with(**overrides):
    return {**VALID, **overrides}


def _codes(items):
    return [item["code"] for item in items]


class TestCheckRecord(unittest.TestCase):

    # --- Happy path ---

    def test_valid_record_passes(self):
        r = check_record(VALID)
        self.assertTrue(r["passed"])
        self.assertEqual(r["errors"], [])
        self.assertEqual(r["summary"]["record_id"], "doc-001")
        self.assertEqual(r["summary"]["source_type"], "PDF")

    def test_total_checks_field_present(self):
        r = check_record(VALID)
        self.assertIn("total_checks", r)
        self.assertGreater(r["total_checks"], 0)

    # --- Group A: Completeness ---

    def test_missing_field_raises_error(self):
        for field in ["document_id", "source_type", "author", "category", "content", "timestamp"]:
            rec = {k: v for k, v in VALID.items() if k != field}
            r = check_record(rec)
            self.assertFalse(r["passed"], msg=f"Should fail when '{field}' is missing")
            self.assertIn("MISSING_FIELD", _codes(r["errors"]))

    def test_empty_string_field_raises_error(self):
        r = check_record(_with(author=""))
        self.assertFalse(r["passed"])
        self.assertIn("EMPTY_FIELD", _codes(r["errors"]))

    def test_whitespace_only_field_raises_error(self):
        r = check_record(_with(category="   "))
        self.assertFalse(r["passed"])
        self.assertIn("EMPTY_FIELD", _codes(r["errors"]))

    def test_none_field_raises_error(self):
        r = check_record(_with(author=None))
        self.assertFalse(r["passed"])
        self.assertIn("EMPTY_FIELD", _codes(r["errors"]))

    # --- Group B: Type & Format ---

    def test_invalid_source_type_raises_error(self):
        r = check_record(_with(source_type="audio"))
        self.assertFalse(r["passed"])
        self.assertIn("INVALID_SOURCE_TYPE", _codes(r["errors"]))

    def test_valid_source_type_lowercase(self):
        r = check_record(_with(source_type="pdf"))
        self.assertTrue(r["passed"])

    def test_valid_source_type_video(self):
        r = check_record(_with(source_type="Video"))
        self.assertTrue(r["passed"])

    def test_invalid_timestamp_raises_error(self):
        r = check_record(_with(timestamp="not-a-date"))
        self.assertFalse(r["passed"])
        self.assertIn("INVALID_TIMESTAMP", _codes(r["errors"]))

    def test_valid_timestamp_formats(self):
        for ts in ["2024-01-15T09:00:00Z", "2024-01-15T09:00:00",
                   "2024-01-15 09:00:00", "2024-01-15", "2024"]:
            r = check_record(_with(timestamp=ts))
            self.assertNotIn("INVALID_TIMESTAMP", _codes(r["errors"]),
                             msg=f"Timestamp '{ts}' should be valid")

    def test_future_timestamp_is_warning_not_error(self):
        r = check_record(_with(timestamp="2099-01-01T00:00:00"))
        self.assertTrue(r["passed"])  # warning only — should not fail
        self.assertIn("FUTURE_TIMESTAMP", _codes(r["warnings"]))

    def test_invalid_document_id_raises_error(self):
        r = check_record(_with(document_id="doc@#$001"))
        self.assertFalse(r["passed"])
        self.assertIn("INVALID_DOCUMENT_ID", _codes(r["errors"]))

    def test_valid_document_id_with_special_chars(self):
        for doc_id in ["doc-001", "vid_993", "report.2024", "path/to/doc"]:
            r = check_record(_with(document_id=doc_id))
            self.assertNotIn("INVALID_DOCUMENT_ID", _codes(r["errors"]),
                             msg=f"document_id '{doc_id}' should be valid")

    # --- Group C: Content Quality ---

    def test_content_critical_short_raises_error(self):
        r = check_record(_with(content="short"))  # 5 chars < 10
        self.assertFalse(r["passed"])
        self.assertIn("CONTENT_CRITICAL_SHORT", _codes(r["errors"]))

    def test_content_exactly_10_chars_passes(self):
        r = check_record(_with(content="0123456789"))  # exactly 10
        self.assertNotIn("CONTENT_CRITICAL_SHORT", _codes(r["errors"]))

    def test_content_between_10_and_50_raises_warning(self):
        r = check_record(_with(content="Just twenty chars here."))  # 23 chars
        self.assertNotIn("CONTENT_CRITICAL_SHORT", _codes(r["errors"]))
        self.assertIn("CONTENT_TOO_SHORT", _codes(r["warnings"]))

    def test_toxic_keyword_null_pointer_raises_error(self):
        r = check_record(_with(content="Critical Error: Null pointer exception found"))
        self.assertFalse(r["passed"])
        self.assertIn("TOXIC_KEYWORD", _codes(r["errors"]))

    def test_toxic_keyword_ocr_error_raises_error(self):
        r = check_record(_with(content="OCR Error detected during scanning process and more text here."))
        self.assertFalse(r["passed"])
        self.assertIn("TOXIC_KEYWORD", _codes(r["errors"]))

    def test_toxic_keyword_traceback_raises_error(self):
        r = check_record(_with(content="Traceback (most recent call last): some long error message."))
        self.assertFalse(r["passed"])
        self.assertIn("TOXIC_KEYWORD", _codes(r["errors"]))

    def test_garbled_encoding_raises_warning(self):
        garbled = "Valid start text " + "�" * 30 + " end of content here."
        r = check_record(_with(content=garbled))
        self.assertIn("GARBLED_ENCODING", _codes(r["warnings"]))

    def test_excessive_whitespace_raises_warning(self):
        spacy = "a" + " " * 200 + "b"  # >60% whitespace
        r = check_record(_with(content=spacy))
        self.assertIn("EXCESSIVE_WHITESPACE", _codes(r["warnings"]))

    # --- Group E: Source-specific ---

    def test_pdf_short_content_raises_warning(self):
        r = check_record(_with(source_type="PDF", content="Short PDF content text here."))
        self.assertIn("SUSPICIOUS_CONTENT_LENGTH", _codes(r["warnings"]))

    def test_video_very_long_content_raises_warning(self):
        long_transcript = "word " * 12000  # >50 000 chars
        r = check_record(_with(source_type="Video", content=long_transcript))
        self.assertIn("SUSPICIOUS_CONTENT_LENGTH", _codes(r["warnings"]))

    # --- TypeError on bad input ---

    def test_raises_type_error_on_non_dict(self):
        with self.assertRaises(TypeError):
            check_record("not a dict")

    def test_raises_type_error_on_list_input(self):
        with self.assertRaises(TypeError):
            check_record([VALID])


class TestCheckBatch(unittest.TestCase):

    def test_all_valid_records_pass(self):
        records = [VALID, _with(document_id="doc-002")]
        r = check_batch(records)
        self.assertTrue(r["passed"])
        self.assertEqual(r["duplicates"], [])
        self.assertEqual(r["total_records"], 2)

    def test_duplicate_ids_detected(self):
        records = [VALID, _with(document_id="doc-001")]  # same ID as VALID
        r = check_batch(records)
        self.assertFalse(r["passed"])
        self.assertIn("doc-001", r["duplicates"])

    def test_per_record_length_matches_input(self):
        records = [VALID, _with(document_id="doc-002"), _with(document_id="doc-003")]
        r = check_batch(records)
        self.assertEqual(len(r["per_record"]), 3)

    def test_total_errors_aggregated(self):
        bad = _with(source_type="audio", document_id="doc-002")
        r = check_batch([VALID, bad])
        self.assertGreater(r["total_errors"], 0)

    def test_raises_type_error_on_non_list(self):
        with self.assertRaises(TypeError):
            check_batch({"not": "a list"})

    def test_empty_list_passes(self):
        r = check_batch([])
        self.assertTrue(r["passed"])
        self.assertEqual(r["total_records"], 0)


class TestRunSemanticChecks(unittest.TestCase):
    """Autograding contract: these exact assertions must hold."""

    def test_valid_full_record_passes(self):
        self.assertTrue(run_semantic_checks(VALID))

    def test_toxic_keyword_fails(self):
        self.assertFalse(run_semantic_checks({"content": "Critical Error: Null pointer exception found"}))

    def test_short_content_fails(self):
        self.assertFalse(run_semantic_checks({"content": "short"}))

    def test_empty_content_fails(self):
        self.assertFalse(run_semantic_checks({"content": ""}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
