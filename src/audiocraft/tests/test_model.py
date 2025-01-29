from unittest.mock import MagicMock, patch

import pytest
import torch
from musicgen.model import TransformerTextualInversion


@pytest.fixture
def mock_model():
    mock_model = MagicMock()
    mock_model.text_conditioner.autocast = MagicMock()
    mock_model.model.autocast = MagicMock()
    mock_model.tokenizer = MagicMock()
    mock_tokenized = {"attention_mask": torch.tensor([[1]])}  # Added this line
    mock_model.tokenizer.return_value = mock_tokenized
    mock_model.text_model = MagicMock()
    mock_model.text_conditioner.output_proj = MagicMock()
    mock_model.model.lm = MagicMock()
    return mock_model


@pytest.fixture
def example_encoded_music():
    return torch.randn(1, 256, 512)  # Example tensor for encoded music


@pytest.fixture
def example_prompts():
    return ["This is a test prompt"]


def test_forward_calls_tokenizer(mock_model, example_encoded_music, example_prompts):
    """
    Test that the forward method correctly calls the tokenizer with the prompts.
    """
    ti_instance = TransformerTextualInversion(model=mock_model, cfg=MagicMock())
    ti_instance.forward(example_encoded_music, example_prompts)

    mock_model.tokenizer.assert_called_once_with(
        example_prompts, return_tensors="pt", padding=True, add_special_tokens=False
    )


def test_forward_mask_creation(mock_model, example_encoded_music, example_prompts):
    """
    Test that the mask is correctly created from the tokenized prompts.
    """
    mock_tokenized = {"attention_mask": torch.tensor([[1, 1, 0, 0]])}
    mock_model.tokenizer.return_value = mock_tokenized
    ti_instance = TransformerTextualInversion(model=mock_model, cfg=MagicMock())

    ti_instance.forward(example_encoded_music, example_prompts)

    assert mock_model.tokenizer.call_count == 1
    assert "attention_mask" in mock_tokenized


def test_forward_text_model_call(mock_model, example_encoded_music, example_prompts):
    """
    Test that the forward method calls the text model with the correct input.
    """
    mock_tokenized = {"attention_mask": torch.tensor([[1, 1, 0, 0]])}
    mock_model.tokenizer.return_value = mock_tokenized
    ti_instance = TransformerTextualInversion(model=mock_model, cfg=MagicMock())

    ti_instance.forward(example_encoded_music, example_prompts)

    mock_model.text_model.assert_called_once_with(**mock_tokenized)
