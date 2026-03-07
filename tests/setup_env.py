"""
Tests for setup_env.py — Jetson detection and pip_install argument assembly.

These tests are fully offline: subprocess.run and Path.exists are mocked
so no actual pip invocations or filesystem reads occur.
"""

import os
from unittest.mock import MagicMock, patch

from setup_env import (
    JETSON_INDEX,
    JETSON_TORCH_ARCH,
    is_jetson,
    pip_install,
)

# ───────────────────────────────────────────────────────────────────────────────
# is_jetson detection
# ───────────────────────────────────────────────────────────────────────────────


class TestIsJetson:
    def test_detects_via_nv_tegra_release(self):
        with patch("setup_env.Path") as MockPath:
            instance = MagicMock()
            instance.exists.return_value = True
            MockPath.return_value = instance
            assert is_jetson() is True

    def test_detects_via_device_tree_model(self, tmp_path):
        model_file = tmp_path / "model"
        model_file.write_text("NVIDIA Jetson Orin NX")

        def fake_path(p):
            m = MagicMock()
            if "nv_tegra" in str(p):
                m.exists.return_value = False
            elif "device-tree" in str(p):
                m.exists.return_value = True
                m.read_text.return_value = "NVIDIA Jetson Orin NX"
            return m

        with patch("setup_env.Path", side_effect=fake_path):
            assert is_jetson() is True

    def test_detects_via_env_var(self):
        with patch("setup_env.Path") as MockPath:
            MockPath.return_value.exists.return_value = False
            with patch.dict(os.environ, {"JETSON_MODEL_NAME": "Jetson Orin"}):
                assert is_jetson() is True

    def test_env_var_case_insensitive(self):
        with patch("setup_env.Path") as MockPath:
            MockPath.return_value.exists.return_value = False
            with patch.dict(os.environ, {"JETSON_MODEL_NAME": "JETSON NANO"}):
                assert is_jetson() is True

    def test_returns_false_on_non_jetson(self):
        with patch("setup_env.Path") as MockPath:
            MockPath.return_value.exists.return_value = False
            with patch.dict(os.environ, {}, clear=True):
                assert is_jetson() is False

    def test_device_tree_non_jetson_content(self):
        def fake_path(p):
            m = MagicMock()
            if "nv_tegra" in str(p):
                m.exists.return_value = False
            elif "device-tree" in str(p):
                m.exists.return_value = True
                m.read_text.return_value = "Raspberry Pi 4 Model B"
            return m

        with patch("setup_env.Path", side_effect=fake_path):
            with patch.dict(os.environ, {}, clear=True):
                assert is_jetson() is False


# ───────────────────────────────────────────────────────────────────────────────
# pip_install command assembly
# ───────────────────────────────────────────────────────────────────────────────


class TestPipInstall:
    def _run(self, extra_index, dev, torch_arch=None):
        with patch("setup_env.subprocess.run") as mock_run:
            pip_install(extra_index=extra_index, dev=dev, torch_arch=torch_arch)
            call = mock_run.call_args
            return call[0][0], call[1]  # (cmd list, kwargs)

    def test_standard_install_command(self):
        cmd, _ = self._run(extra_index=None, dev=False)
        assert cmd[-1] == "."
        assert "--extra-index-url" not in cmd

    def test_jetson_index_is_added(self):
        cmd, _ = self._run(extra_index=JETSON_INDEX, dev=False)
        assert "--extra-index-url" in cmd
        assert JETSON_INDEX in cmd

    def test_dev_extras_included(self):
        cmd, _ = self._run(extra_index=None, dev=True)
        assert ".[dev]" in cmd

    def test_dev_and_jetson_together(self):
        cmd, _ = self._run(extra_index=JETSON_INDEX, dev=True)
        assert ".[dev]" in cmd
        assert "--extra-index-url" in cmd
        assert JETSON_INDEX in cmd

    def test_subprocess_called_with_check_true(self):
        _, kwargs = self._run(extra_index=None, dev=False)
        assert kwargs.get("check") is True

    def test_torch_arch_set_in_env_for_jetson(self):
        _, kwargs = self._run(extra_index=JETSON_INDEX, dev=False, torch_arch=JETSON_TORCH_ARCH)
        assert kwargs["env"]["TORCH_CUDA_ARCH_LIST"] == JETSON_TORCH_ARCH

    def test_torch_arch_is_ampere_sm87(self):
        # Orin Nano must always target sm_87 specifically.
        assert JETSON_TORCH_ARCH == "8.7"

    def test_torch_arch_not_set_for_standard_install(self):
        _, kwargs = self._run(extra_index=None, dev=False, torch_arch=None)
        # TORCH_CUDA_ARCH_LIST should not be injected for non-Jetson builds.
        assert "TORCH_CUDA_ARCH_LIST" not in kwargs.get("env", {})

    def test_torch_arch_absent_leaves_parent_env_intact(self):
        with patch.dict(os.environ, {"SOME_EXISTING_VAR": "1"}):
            _, kwargs = self._run(extra_index=None, dev=False, torch_arch=None)
            # env is not passed to subprocess when torch_arch is None,
            # so the child inherits the parent env untouched.
            assert kwargs.get("env") is None or "TORCH_CUDA_ARCH_LIST" not in kwargs.get("env", {})
