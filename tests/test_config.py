from config import VECTORS_DIR, vector_cache_path


def test_vector_cache_path_is_under_vectors_dir():
    path = vector_cache_path("BAAI/bge-small-en-v1.5")
    assert path.parent == VECTORS_DIR


def test_vector_cache_filename_embeds_model_name_and_extension():
    path = vector_cache_path("BAAI/bge-small-en-v1.5")
    assert "bge-small-en-v1.5" in path.name
    assert path.name.endswith(".pt")


def test_vector_cache_path_sanitises_slashes():
    path = vector_cache_path("BAAI/bge-small-en-v1.5")
    # No nested subdirectory should be created from the "/" in the model name.
    assert path.parent == VECTORS_DIR
    assert "/" not in path.name
