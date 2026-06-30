"""Tests for content chunking utilities."""

import json

from cograph_client.resolver.chunker import (
    chunk_text,
    chunk_json_array,
    split_json_array_chunk,
    json_array_len,
)


class TestChunkText:
    def test_small_text_single_chunk(self):
        text = "Hello world. This is short."
        chunks = chunk_text(text, max_chars=3000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_large_text_splits(self):
        sentences = [f"Sentence number {i} is here." for i in range(100)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_chars=200)
        assert len(chunks) > 1
        # All content should be represented
        combined = " ".join(chunks)
        for s in sentences:
            assert s in combined

    def test_splits_into_multiple_chunks(self):
        sentences = [f"This is sentence {i}." for i in range(50)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_chars=100, overlap=0)
        assert len(chunks) > 1

    def test_empty_text(self):
        chunks = chunk_text("")
        assert chunks == [""]

    def test_overlap_provides_context(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, max_chars=40, overlap=20)
        if len(chunks) > 1:
            # Second chunk should contain overlap from first
            assert len(chunks[1]) > 0


class TestChunkJsonArray:
    def test_small_array_single_chunk(self):
        data = [{"id": i} for i in range(10)]
        content = json.dumps(data)
        chunks = chunk_json_array(content, batch_size=50)
        assert len(chunks) == 1

    def test_large_array_splits(self):
        data = [{"id": i, "name": f"item_{i}"} for i in range(120)]
        content = json.dumps(data)
        chunks = chunk_json_array(content, batch_size=50)
        assert len(chunks) == 3  # 50 + 50 + 20

        # Verify all items present
        all_items = []
        for chunk in chunks:
            all_items.extend(json.loads(chunk))
        assert len(all_items) == 120

    def test_non_array_single_chunk(self):
        content = json.dumps({"key": "value"})
        chunks = chunk_json_array(content)
        assert len(chunks) == 1
        assert chunks[0] == content

    def test_invalid_json(self):
        chunks = chunk_json_array("not json at all")
        assert len(chunks) == 1
        assert chunks[0] == "not json at all"

    def test_empty_array(self):
        chunks = chunk_json_array("[]")
        assert len(chunks) == 1

    def test_default_batch_size_is_25(self):
        """FIX 1: the default batch was lowered 50 → 25 so a denser reified
        chunk's JSON output stays under the LLM token cap."""
        data = [{"id": i} for i in range(60)]
        content = json.dumps(data)
        chunks = chunk_json_array(content)  # default batch_size
        assert len(chunks) == 3  # 25 + 25 + 10
        all_items = []
        for c in chunks:
            all_items.extend(json.loads(c))
        assert len(all_items) == 60


class TestSplitJsonArrayChunk:
    """FIX 1: the recovery helper that halves a chunk whose extraction failed."""

    def test_splits_in_half_conserving_records(self):
        data = [{"id": i} for i in range(10)]
        halves = split_json_array_chunk(json.dumps(data))
        assert len(halves) == 2
        left, right = json.loads(halves[0]), json.loads(halves[1])
        assert len(left) == 5 and len(right) == 5
        # No record lost, order preserved.
        assert left + right == data

    def test_odd_length_splits_lower_then_upper(self):
        data = [{"id": i} for i in range(7)]
        halves = split_json_array_chunk(json.dumps(data))
        left, right = json.loads(halves[0]), json.loads(halves[1])
        assert len(left) == 3 and len(right) == 4
        assert left + right == data

    def test_single_record_cannot_split(self):
        assert split_json_array_chunk(json.dumps([{"id": 1}])) == []

    def test_empty_array_cannot_split(self):
        assert split_json_array_chunk("[]") == []

    def test_non_array_cannot_split(self):
        assert split_json_array_chunk(json.dumps({"k": "v"})) == []

    def test_invalid_json_cannot_split(self):
        assert split_json_array_chunk("not json") == []


class TestJsonArrayLen:
    def test_counts_array_records(self):
        assert json_array_len(json.dumps([{"a": 1}, {"b": 2}, {"c": 3}])) == 3

    def test_empty_array_is_zero(self):
        assert json_array_len("[]") == 0

    def test_non_array_is_zero(self):
        assert json_array_len(json.dumps({"k": "v"})) == 0

    def test_invalid_json_is_zero(self):
        assert json_array_len("garbage") == 0
