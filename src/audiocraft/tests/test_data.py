from unittest.mock import MagicMock, patch

import pytest
from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import T5Conditioner
from musicgen.data import TextConcepts, Concept


def test_text_concepts_initialization():
    """
    Test initializing a TextConcepts instance with a list of Concept objects.
    """
    concept1 = Concept(name="concept1", token_ids=[1, 2], tokens=["a", "b"])
    concept2 = Concept(name="concept2", token_ids=[3, 4], tokens=["c", "d"])

    text_concepts = TextConcepts([concept1, concept2])

    assert len(text_concepts.concepts) == 2
    assert "concept1" in text_concepts.concepts
    assert "concept2" in text_concepts.concepts


def test_text_concepts_concepts_names():
    """
    Test the concepts_names property returns the correct names.
    """
    concept1 = Concept(name="concept1", token_ids=[1, 2], tokens=["a", "b"])
    concept2 = Concept(name="concept2", token_ids=[3, 4], tokens=["c", "d"])

    text_concepts = TextConcepts([concept1, concept2])

    assert text_concepts.concepts_names == ["concept1", "concept2"]


def test_text_concepts_all_token_ids():
    """
    Test the cached_property all_token_ids returns collected token IDs.
    """
    concept1 = Concept(name="concept1", token_ids=[1, 2], tokens=["a", "b"])
    concept2 = Concept(name="concept2", token_ids=[3, 4], tokens=["c", "d"])

    text_concepts = TextConcepts([concept1, concept2])

    assert text_concepts.all_token_ids == [1, 2, 3, 4]


def test_text_concepts_execute():
    """
    Test the `execute` method applies a function to all concepts.
    """
    concept1 = Concept(name="concept1", token_ids=[1, 2], tokens=["a", "b"])
    concept2 = Concept(name="concept2", token_ids=[3, 4], tokens=["c", "d"])

    text_concepts = TextConcepts([concept1, concept2])

    mock_func = MagicMock()
    text_concepts.execute(mock_func)

    assert mock_func.call_count == 2
    mock_func.assert_any_call(concept1)
    mock_func.assert_any_call(concept2)


def test_text_concepts_from_init():
    """
    Test the from_init method correctly constructs a TextConcepts object.
    """
    T5TokenizerMock = MagicMock()
    T5Mock = MagicMock()
    TextConditionerMock = MagicMock(
        t5_tokenizer=T5TokenizerMock,
        t5=T5Mock,
    )
    TokensProviderMock = MagicMock()

    tokens_provider = TokensProviderMock()
    tokens_provider.get.side_effect = [["a", "b"], ["c", "d"]]
    T5TokenizerMock.add_tokens.side_effect = None
    T5TokenizerMock.convert_tokens_to_ids.side_effect = [[1, 2], [3, 4]]

    concepts = ["concept1", "concept2"]

    text_concepts = TextConcepts.from_init(
        TextConditionerMock,
        tokens_provider,
        concepts,
    )

    assert len(text_concepts.concepts) == 2
    assert "concept1" in text_concepts.concepts
    assert "concept2" in text_concepts.concepts


def test_text_concepts_from_musicgen():
    """
    Test the from_musicgen method correctly constructs a TextConcepts object.
    """
    MusicGenMock = MagicMock()
    ConditionerMock = MagicMock()
    TokensProviderMock = MagicMock()

    music_model = MusicGenMock()
    music_model.lm.condition_provider.conditioners = {"key": ConditionerMock}
    tokens_provider = TokensProviderMock()
    tokens_provider.get.side_effect = [["a", "b"], ["c", "d"]]

    concepts = ["concept1", "concept2"]
    ConditionerMock.t5_tokenizer.add_tokens.side_effect = None
    ConditionerMock.t5_tokenizer.convert_tokens_to_ids.side_effect = [[1, 2], [3, 4]]

    text_concepts = TextConcepts.from_musicgen(
        music_model,
        tokens_provider,
        concepts,
    )

    assert len(text_concepts.concepts) == 2
    assert "concept1" in text_concepts.concepts
    assert "concept2" in text_concepts.concepts


def test_text_concepts_getitem():
    """
    Test the __getitem__ method returns the correct Concept.
    """
    concept1 = Concept(name="concept1", token_ids=[1, 2], tokens=["a", "b"])
    concept2 = Concept(name="concept2", token_ids=[3, 4], tokens=["c", "d"])

    text_concepts = TextConcepts([concept1, concept2])

    assert text_concepts["concept1"] == concept1
    assert text_concepts["concept2"] == concept2
