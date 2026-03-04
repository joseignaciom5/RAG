"""
Tests para el módulo de configuración.
"""
import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import Settings, get_settings
from src.exceptions import ConfigurationError


class TestSettings:
    """Tests para la clase Settings."""

    def test_settings_default_values(self):
        """Test que los valores por defecto son correctos."""
        settings = Settings(_env_file=None)
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.llm_model_name == "qwen2.5:7b"
        assert settings.llm_temperature == 0.1
        assert settings.embedding_model_name == "intfloat/multilingual-e5-small"
        assert settings.vectorstore_path == "./vectorstore"
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.retrieval_k == 4

    def test_settings_custom_values(self):
        """Test que se pueden configurar valores personalizados."""
        env_vars = {
            "OLLAMA_BASE_URL": "http://localhost:9999",
            "LLM_MODEL_NAME": "llama3:8b",
            "LLM_TEMPERATURE": "0.5",
            "CHUNK_SIZE": "500",
            "RETRIEVAL_K": "6",
        }
        with patch.dict(os.environ, env_vars):
            settings = Settings(_env_file=None)
            assert settings.ollama_base_url == "http://localhost:9999"
            assert settings.llm_model_name == "llama3:8b"
            assert settings.llm_temperature == 0.5
            assert settings.chunk_size == 500
            assert settings.retrieval_k == 6

    def test_temperature_validation_min(self):
        """Test que temperature debe ser >= 0."""
        with patch.dict(os.environ, {"LLM_TEMPERATURE": "-0.5"}):
            with pytest.raises(ValidationError):
                Settings(_env_file=None)

    def test_temperature_validation_max(self):
        """Test que temperature debe ser <= 2."""
        with patch.dict(os.environ, {"LLM_TEMPERATURE": "2.5"}):
            with pytest.raises(ValidationError):
                Settings(_env_file=None)

    def test_chunk_size_must_be_positive(self):
        """Test que chunk_size debe ser positivo."""
        with patch.dict(os.environ, {"CHUNK_SIZE": "0"}):
            with pytest.raises(ValidationError):
                Settings(_env_file=None)

    def test_chunk_overlap_cannot_be_negative(self):
        """Test que chunk_overlap no puede ser negativo."""
        with patch.dict(os.environ, {"CHUNK_OVERLAP": "-1"}):
            with pytest.raises(ValidationError):
                Settings(_env_file=None)

    def test_retrieval_k_must_be_positive(self):
        """Test que retrieval_k debe ser positivo."""
        with patch.dict(os.environ, {"RETRIEVAL_K": "0"}):
            with pytest.raises(ValidationError):
                Settings(_env_file=None)


class TestGetSettings:
    """Tests para la función get_settings."""

    def test_get_settings_returns_settings(self):
        """Test que get_settings retorna un Settings válido."""
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_raises_configuration_error(self):
        """Test que get_settings lanza ConfigurationError cuando falla."""
        get_settings.cache_clear()
        with patch("src.config.Settings", side_effect=Exception("config error")):
            with pytest.raises(ConfigurationError):
                get_settings()

    def test_get_settings_is_cached(self):
        """Test que get_settings retorna la misma instancia (cache)."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
