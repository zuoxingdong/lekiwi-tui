from __future__ import annotations

import signal

from lekiwi_tui.framework import runner


class _NoTerminalApp:
    terminal = None


def test_suspend_run_converts_parent_keyboard_interrupt_to_130(monkeypatch):
    class FakePopen:
        def __init__(self, argv, env=None):  # noqa: ANN001
            self.argv = argv
            self.env = env
            self.returncode = None
            self.waits = 0
            self.signals = []

        def wait(self, timeout=None):  # noqa: ANN001
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            self.returncode = 130
            return self.returncode

        def send_signal(self, sig):  # noqa: ANN001
            self.signals.append(sig)

    proc_holder = {}

    def fake_popen(argv, env=None, **kwargs):  # noqa: ANN001, ARG001
        proc = FakePopen(argv, env=env)
        proc_holder["proc"] = proc
        return proc

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    rc = runner.suspend_run(_NoTerminalApp(), ["bash", "scripts/eval.sh"])

    assert rc == 130
    assert proc_holder["proc"].signals == [signal.SIGINT]


def test_suspend_run_keeps_130_even_if_child_exits_cleanly_after_interrupt(monkeypatch):
    class FakePopen:
        def __init__(self, argv, env=None):  # noqa: ANN001
            self.argv = argv
            self.env = env
            self.waits = 0
            self.signals = []

        def wait(self, timeout=None):  # noqa: ANN001
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            return 0

        def send_signal(self, sig):  # noqa: ANN001
            self.signals.append(sig)

    proc_holder = {}

    def fake_popen(argv, env=None, **kwargs):  # noqa: ANN001, ARG001
        proc = FakePopen(argv, env=env)
        proc_holder["proc"] = proc
        return proc

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    rc = runner.suspend_run(_NoTerminalApp(), ["bash", "scripts/eval.sh"])

    assert rc == 130
    assert proc_holder["proc"].signals == [signal.SIGINT]


def test_suspend_run_treats_parent_sigint_signal_as_cancel_when_child_returns_0(monkeypatch):
    class FakePopen:
        def __init__(self, argv, env=None):  # noqa: ANN001
            self.argv = argv
            self.env = env
            self.waits = 0
            self.signals = []

        def wait(self, timeout=None):  # noqa: ANN001
            self.waits += 1
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler)
            handler(signal.SIGINT, None)
            return 0

        def send_signal(self, sig):  # noqa: ANN001
            self.signals.append(sig)

    proc_holder = {}
    old_handler = signal.getsignal(signal.SIGINT)

    def fake_popen(argv, env=None, **kwargs):  # noqa: ANN001, ARG001
        proc = FakePopen(argv, env=env)
        proc_holder["proc"] = proc
        return proc

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    rc = runner.suspend_run(_NoTerminalApp(), ["bash", "scripts/eval.sh"])

    assert rc == 130
    assert proc_holder["proc"].signals == [signal.SIGINT]
    assert signal.getsignal(signal.SIGINT) == old_handler
