import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import cli_progress


class _FakeStream:
    def __init__(self, is_tty: bool = True) -> None:
        self._is_tty = is_tty
        self.messages: list[str] = []

    def isatty(self) -> bool:
        return self._is_tty

    def write(self, text: str) -> None:
        self.messages.append(text)

    def flush(self) -> None:
        return None


class _FailingFirstWriteStream(_FakeStream):
    def __init__(self) -> None:
        super().__init__(is_tty=True)
        self._writes = 0

    def write(self, text: str) -> None:
        self._writes += 1
        if self._writes == 1:
            raise OSError("console write failed")
        self.messages.append(text)


class CliProgressTests(unittest.TestCase):
    def test_non_tty_stream_disables_ui(self) -> None:
        stream = _FakeStream(is_tty=False)

        enabled = cli_progress.should_enable_cli_ui(force=False, disable=False, stream=stream)

        self.assertFalse(enabled)

    def test_windows_vt_success_enables_ui(self) -> None:
        stream = _FakeStream(is_tty=True)

        with mock.patch.object(cli_progress.os, "name", "nt"), mock.patch.object(
            cli_progress, "_enable_windows_virtual_terminal", return_value=True
        ):
            enabled = cli_progress.should_enable_cli_ui(force=False, disable=False, stream=stream)

        self.assertTrue(enabled)

    def test_force_ui_still_disables_when_windows_vt_init_fails(self) -> None:
        stream = _FakeStream(is_tty=True)

        with mock.patch.object(cli_progress.os, "name", "nt"), mock.patch.object(
            cli_progress, "_enable_windows_virtual_terminal", return_value=False
        ):
            enabled = cli_progress.should_enable_cli_ui(force=True, disable=False, stream=stream)

        self.assertFalse(enabled)

    def test_render_failure_downgrades_to_plain_output(self) -> None:
        stream = _FailingFirstWriteStream()
        progress = cli_progress.CliProgress(
            enabled=True,
            stream=stream,
            fallback_message="当前终端不支持 CLI UI，已退回普通日志。",
        )

        progress.start_run(mode="deep", title="probe", seed=42)
        progress.log_message("after fallback")

        self.assertFalse(progress.enabled)
        joined = "".join(stream.messages)
        self.assertIn("当前终端不支持 CLI UI，已退回普通日志。", joined)
        self.assertIn("after fallback", joined)


if __name__ == "__main__":
    unittest.main()
