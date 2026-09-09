from __future__ import annotations

import pytest

from profiling import ProfileConfig


def test_profile_config_disabled_by_default():
    config = ProfileConfig()
    assert not config.enabled
    assert config.total_steps == 3


def test_profile_config_enabled():
    config = ProfileConfig(steps=5, wait=1, warmup=2)
    assert config.enabled
    assert config.total_steps == 8


def test_profile_config_rejects_negative_steps():
    with pytest.raises(ValueError, match="profile steps"):
        ProfileConfig(steps=-1)
