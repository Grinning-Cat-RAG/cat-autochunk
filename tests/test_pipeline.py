from types import SimpleNamespace

try:
    from .. import pipeline
except ImportError:  # pytest (see conftest.py)
    from autochunk_plugin import pipeline


def _point(metadata):
    return SimpleNamespace(id="x", payload={"page_content": "...", "metadata": metadata})


def test_custom_metadata_are_the_common_ones():
    points = [
        _point({"source": "a.pdf", "when": 1, "hash": "h", "page": 1, "author": "Alice", "tags": ["x", "y"]}),
        _point({"source": "a.pdf", "when": 2, "hash": "h", "page": 2, "author": "Alice", "tags": ["x", "y"]}),
        # an image point (multimodal ingestion) does not count
        _point({"source": "a.pdf", "image_file": "a_img_1.png", "author": "Alice"}),
    ]
    assert pipeline.extract_custom_metadata(points, generated_keys={"page", "source"}) == {
        "author": "Alice", "tags": ["x", "y"],
    }


def test_generated_keys_are_excluded_for_single_chunk_files():
    points = [_point({"source": "a.txt", "when": 1, "page": 1, "author": "Bob", "_is_merged": True})]
    assert pipeline.extract_custom_metadata(points, generated_keys={"page"}) == {"author": "Bob"}


def test_different_values_are_not_custom_metadata():
    points = [_point({"source": "a.txt", "author": "Bob"}), _point({"source": "a.txt", "author": "Carl"})]
    assert pipeline.extract_custom_metadata(points, generated_keys=set()) == {}


def test_no_points():
    assert pipeline.extract_custom_metadata([], generated_keys=set()) == {}
