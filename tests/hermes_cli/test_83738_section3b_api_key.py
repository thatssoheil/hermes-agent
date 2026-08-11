"""Regression tests for bare-custom picker probing with the config api_key (#83837).

Section 3b of ``list_authenticated_providers`` probes the active bare custom
endpoint (``model.provider: custom`` + ``model.base_url``) for its live
model catalog. It used to hardcode an empty api_key, so authenticated
endpoints rejected the probe with 401 and the picker silently collapsed to
``[current_model]``. The fix resolves the key from config exactly as the
runtime does (the ``("api_key", "api")`` loop over model config) and
threads it into the probe.
"""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers


def _bare_custom_rows(monkeypatch, captured):
    """Run list_authenticated_providers for a bare custom endpoint and
    capture the api_key passed to cached_fetch_api_models."""
    import hermes_cli.providers as providers_mod

    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})

    def _fake_fetch(api_key, base_url, **kwargs):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        return ["model-a", "model-b"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", _fake_fetch)
    # Section 3b only runs when the current provider is bare 'custom' with a
    # base_url and no named custom_providers row matches that URL. Probe
    # live so cached_fetch_api_models actually calls fetch_api_models.
    return list_authenticated_providers(
        current_provider="custom",
        current_base_url="https://custom.example.com/v1",
        current_model="model-a",
        custom_providers=[],
        probe_custom_providers=False,
        probe_current_custom_provider=True,
        max_models=50,
    )


def test_section3b_threads_config_api_key_into_probe(monkeypatch):
    captured = {}
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={
            "model": {
                "provider": "custom",
                "base_url": "https://custom.example.com/v1",
                "api_key": "sk-test-123",
            }
        },
    ):
        rows = _bare_custom_rows(monkeypatch, captured)

    assert captured["api_key"] == "sk-test-123"
    assert captured["base_url"] == "https://custom.example.com/v1"
    # The probe succeeded, so the live catalog replaces [current_model].
    custom = [r for r in rows if r["slug"] == "custom"]
    assert custom and custom[0]["models"] == ["model-a", "model-b"]


def test_section3b_honors_api_alias_like_runtime(monkeypatch):
    """The runtime resolves bare-custom keys via the (\"api_key\", \"api\")
    loop (runtime_provider.py); the probe must accept model.api too."""
    captured = {}
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={
            "model": {
                "provider": "custom",
                "base_url": "https://custom.example.com/v1",
                "api": "sk-api-alias",
            }
        },
    ):
        rows = _bare_custom_rows(monkeypatch, captured)

    assert captured["api_key"] == "sk-api-alias"
    custom = [r for r in rows if r["slug"] == "custom"]
    assert custom and custom[0]["models"] == ["model-a", "model-b"]


def test_section3b_api_key_precedence_over_api(monkeypatch):
    """api_key wins over the api alias, matching the runtime loop order."""
    captured = {}
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={
            "model": {
                "provider": "custom",
                "base_url": "https://custom.example.com/v1",
                "api_key": "sk-primary",
                "api": "sk-alias",
            }
        },
    ):
        rows = _bare_custom_rows(monkeypatch, captured)

    assert captured["api_key"] == "sk-primary"
    custom = [r for r in rows if r["slug"] == "custom"]
    assert custom and custom[0]["models"] == ["model-a", "model-b"]


def test_section3b_without_key_still_probes_locally(monkeypatch):
    """No api_key configured: the probe still runs (bare local endpoints
    expose their catalog without auth) but passes the empty key."""
    captured = {}
    with patch(
        "hermes_cli.config.load_config_readonly",
        return_value={
            "model": {
                "provider": "custom",
                "base_url": "https://custom.example.com/v1",
            }
        },
    ):
        rows = _bare_custom_rows(monkeypatch, captured)

    assert captured["api_key"] == ""
    custom = [r for r in rows if r["slug"] == "custom"]
    assert custom and custom[0]["models"] == ["model-a", "model-b"]
