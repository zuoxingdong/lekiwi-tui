"""Laptop resource sampling for the status card, and the row it renders.

Two properties matter most and both are about not lying: sampling must never run on the
render path (draw() is called every frame, and nvidia-smi costs tens of milliseconds), and
a field that cannot be read must be OMITTED rather than shown as a zero.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

from lekiwi_tui import sysstat
from lekiwi_tui.sysstat import Sample, SysStat, get_sysstat
from lekiwi_tui.screens.menu import MenuScreen

from conftest import make_ctx


def _text(spans) -> str:
    return "".join(sp.content for sp in spans)


def _screen(sample: Sample | None = None, gpu_name: str = "RTX 4090") -> MenuScreen:
    screen = MenuScreen(MagicMock(), make_ctx(gpu_name=gpu_name))
    if sample is not None:
        stat = SysStat()
        stat.sample = sample
        stat._last_start = time.monotonic()   # keep poll() from kicking a real sample
        screen.ctx.ui_state["sysstat"] = stat
    return screen


# ── the sampler ───────────────────────────────────────────────────────────────
def test_poll_returns_immediately_and_does_not_sample_inline(monkeypatch):
    """The whole point of the module: draw() must not pay for the sample."""
    monkeypatch.setattr(sysstat, "_sample_cpu", lambda: time.sleep(5) or 1.0)
    stat = SysStat()

    started = time.monotonic()
    stat.poll()
    elapsed = time.monotonic() - started

    assert elapsed < 0.2, "poll() blocked on the sample"
    assert stat.sample == Sample(), "no result should be published yet"


def test_poll_is_throttled(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(sysstat, "_sample_cpu", lambda: calls.append(1) or 1.0)
    monkeypatch.setattr(sysstat, "_sample_ram", lambda: None)
    monkeypatch.setattr(sysstat, "_sample_gpu", lambda: None)
    stat = SysStat()

    for _ in range(50):          # what 50 frames of draw() would do
        stat.poll()
    time.sleep(0.3)

    assert len(calls) == 1, f"throttle let {len(calls)} samples through"


def test_a_sample_lands_on_the_object(monkeypatch):
    monkeypatch.setattr(sysstat, "_sample_cpu", lambda: 42.0)
    monkeypatch.setattr(sysstat, "_sample_ram", lambda: (8.0, 32.0))
    monkeypatch.setattr(sysstat, "_sample_gpu", lambda: (17, 2.5, 8.0))
    stat = SysStat()

    stat.poll()
    for _ in range(100):
        if stat.sample.cpu_pct is not None:
            break
        time.sleep(0.01)

    assert stat.sample == Sample(42.0, 8.0, 32.0, 17, 2.5, 8.0)


def test_sampling_degrades_per_field_instead_of_raising(monkeypatch):
    """No /proc and no NVIDIA GPU is a normal machine, not an error."""
    monkeypatch.setattr(sysstat, "_sample_cpu", lambda: None)
    monkeypatch.setattr(sysstat, "_sample_ram", lambda: None)
    monkeypatch.setattr(sysstat, "_sample_gpu", lambda: None)
    stat = SysStat()

    stat._run()

    assert stat.sample == Sample()


def test_gpu_sampler_survives_a_missing_nvidia_smi(monkeypatch):
    def boom(*a, **k):
        raise OSError("nvidia-smi: not found")

    monkeypatch.setattr(sysstat.subprocess, "run", boom)

    assert sysstat._sample_gpu() is None


def test_gpu_sampler_survives_unparseable_output(monkeypatch):
    monkeypatch.setattr(sysstat.subprocess, "run",
                        lambda *a, **k: MagicMock(stdout="No devices were found\n"))

    assert sysstat._sample_gpu() is None


def test_the_sampler_is_shared_through_the_context():
    ctx = make_ctx()

    first = get_sysstat(ctx)

    assert get_sysstat(ctx) is first, "a per-frame new sampler would never beat its throttle"


def test_real_ram_reading_is_plausible():
    """Not mocked: on this Linux box the numbers should actually make sense."""
    ram = sysstat._sample_ram()
    if ram is None:
        return                                  # no /proc — nothing to assert
    used, total = ram

    assert 0 < total < 4096, f"implausible total RAM: {total} GB"
    assert 0 <= used <= total


def test_real_cpu_reading_is_a_percentage():
    cpu = sysstat._sample_cpu()
    if cpu is None:
        return

    assert 0.0 <= cpu <= 100.0


# ── the rendered row ──────────────────────────────────────────────────────────
def test_machine_row_shows_cpu_ram_gpu_and_vram():
    screen = _screen(Sample(34.0, 12.4, 31.0, 18, 2.1, 8.0))

    text = _text(screen._machine_spans())

    assert text.startswith("laptop  "), "the row must say whose numbers these are"
    assert "cpu 34%" in text
    assert "ram 12.4/31 GB" in text
    assert "RTX 4090 18%" in text, "the GPU name doubles as the label"
    assert "vram 2.1/8 GB" in text


def test_machine_row_omits_fields_it_could_not_read():
    screen = _screen(Sample(cpu_pct=55.0), gpu_name="")

    text = _text(screen._machine_spans())

    assert "cpu 55%" in text
    assert "ram" not in text
    assert "vram" not in text


def test_machine_row_says_so_when_nothing_is_readable():
    screen = _screen(Sample(), gpu_name="")

    text = _text(screen._machine_spans())

    assert "unavailable" in text, "an empty row would look like a rendering bug"


def test_machine_row_keeps_the_gpu_name_when_utilisation_is_unavailable():
    screen = _screen(Sample(cpu_pct=10.0), gpu_name="RTX 4090")

    assert "RTX 4090" in _text(screen._machine_spans())


def test_load_colour_escalates_with_load():
    from lekiwi_tui.framework import theme

    assert MenuScreen._load_style(10) is theme.OK_STYLE
    assert MenuScreen._load_style(75) is theme.WARN_STYLE
    assert MenuScreen._load_style(95) is theme.ERR_STYLE


def test_the_status_card_has_three_rows_and_the_middle_one_is_the_laptop():
    import pyratatui as p

    screen = _screen(Sample(34.0, 12.4, 31.0, 18, 2.1, 8.0))
    captured: list[tuple[str, list]] = []
    original = MenuScreen._draw_panel
    MenuScreen._draw_panel = lambda self, f, a, t, ls: captured.append((t, ls))
    try:
        screen.draw(MagicMock(), p.Rect(0, 0, 100, 40))
    finally:
        MenuScreen._draw_panel = original

    title, lines = captured[0]
    assert title == "ROBOT"
    assert len(lines) == 3
    assert _text(lines[1].spans).startswith("laptop")
