import pytest

TINY_CLIP = "hf-internal-testing/tiny-random-CLIPModel"


@pytest.fixture(scope="session")
def tiny_clip():
    """A 5-block CLIP with random weights, a few MB from the Hugging Face Hub.

    Tests that need it are skipped, not failed, when the Hub is unreachable.
    """
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained(TINY_CLIP)
    except OSError as error:
        pytest.skip(f"cannot reach the Hugging Face Hub: {error}")
    return TINY_CLIP
