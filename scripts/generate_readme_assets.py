#!/usr/bin/env python3
"""Generate README hero PNG and short workflow GIF.

These are faithful static renders of the current terminal style: plain dark terminal,
runtime chips, colorful action icons, keycap footer hints, and focused-row highlighting.
They are not screenshots from a live TTY, but they intentionally avoid decorative window
chrome so the result stays close to what Ghostty shows. Values are public placeholders,
not local machine paths or private checkpoint names.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lekiwi_tui.app_registry import ACTIONS

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
HERO = ASSETS / "lekiwi-tui-hero.png"
GIF = ASSETS / "lekiwi-tui-dry-run.gif"

FONT_REG = "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf"
KEY_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

W, H = 1180, 760
PAD_X = 16
LINE_H = 22
FONT_SIZE = 17
EMOJI_SIZE = 18

BG = "#0b101f"
SURFACE = "#111827"
PANEL = "#1f2937"
TEXT = "#e5edf7"
MUTED = "#94a3b8"
ACCENT = "#38bdf8"
SAND = "#fbbf24"
SUCCESS = "#34d399"
WARNING = "#f59e0b"
ERROR = "#fb7185"
HAIRLINE = "#334155"
PURPLE = "#a78bfa"
HIGHLIGHT_BG = "#0e354a"

PUBLIC_HOST = "robot-pi"
PUBLIC_ENV = "lerobot"
PUBLIC_GPU = "CUDA"
PUBLIC_POLICY = "models/lekiwi-policy/checkpoints/latest/pretrained_model"
PUBLIC_TASK = "Pick up the object and place it in the tray"


def font(bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, FONT_SIZE)


FONT = font(False)
BOLD = font(True)
KEY = ImageFont.truetype(KEY_FONT, FONT_SIZE)
EMOJI = ImageFont.truetype(EMOJI_FONT, 109)


class Canvas:
    def __init__(self) -> None:
        self.img = Image.new("RGB", (W, H), BG)
        self.d = ImageDraw.Draw(self.img)
        self._emoji_cache: dict[str, Image.Image] = {}

    def text(self, xy: tuple[int, int], value: str, fill: str = TEXT, *, bold: bool = False) -> None:
        self.d.text(xy, value, font=BOLD if bold else FONT, fill=fill)

    def rect(self, xy: tuple[int, int, int, int], fill: str, outline: str | None = None) -> None:
        self.d.rectangle(xy, fill=fill, outline=outline)

    def line(self, xy: tuple[int, int, int, int], fill: str = HAIRLINE, width: int = 1) -> None:
        self.d.line(xy, fill=fill, width=width)

    def emoji(self, xy: tuple[int, int], glyph: str) -> None:
        glyph = glyph.replace("\ufe0f", "")
        if glyph not in self._emoji_cache:
            raw = Image.new("RGBA", (150, 150), (0, 0, 0, 0))
            draw = ImageDraw.Draw(raw)
            draw.text((0, -8), glyph, font=EMOJI, embedded_color=True)
            bbox = raw.getbbox()
            if bbox is None:
                tile = Image.new("RGBA", (EMOJI_SIZE, EMOJI_SIZE), (0, 0, 0, 0))
            else:
                tile = raw.crop(bbox)
                tile.thumbnail((EMOJI_SIZE, EMOJI_SIZE), Image.Resampling.LANCZOS)
            self._emoji_cache[glyph] = tile
        tile = self._emoji_cache[glyph]
        self.img.paste(tile, xy, tile)

    def chip(self, x: int, y: int, label: str, value: str, color: str = SAND) -> int:
        label_w = int(self.d.textlength(f" {label} ", font=FONT))
        value_w = int(self.d.textlength(f"{value} ", font=BOLD))
        self.rect((x, y, x + label_w + value_w, y + LINE_H - 2), PANEL)
        self.text((x + 2, y + 1), label, MUTED)
        self.text((x + label_w, y + 1), value, color, bold=True)
        return x + label_w + value_w + 10

    def runtime_chips(self, y: int, *, mode: str = "REAL") -> None:
        x = PAD_X
        x = self.chip(x, y, "host", PUBLIC_HOST)
        x = self.chip(x, y, "env", PUBLIC_ENV)

        label = " GPU "
        dot = "● "
        gpu = f"{PUBLIC_GPU} "
        label_w = int(self.d.textlength(label, font=FONT))
        dot_w = int(self.d.textlength(dot, font=BOLD))
        gpu_w = int(self.d.textlength(gpu, font=FONT))
        self.rect((x, y, x + label_w + dot_w + gpu_w, y + LINE_H - 2), PANEL)
        self.text((x + 2, y + 1), "GPU", MUTED)
        self.text((x + label_w, y + 1), dot, SUCCESS, bold=True)
        self.text((x + label_w + dot_w, y + 1), PUBLIC_GPU, TEXT)
        x += label_w + dot_w + gpu_w + 10

        self.chip(x, y, "mode", mode, SUCCESS if mode == "REAL" else WARNING)

    def header(self, subtitle: str) -> None:
        self.text((PAD_X, 12), "◆ LEKIWI", ACCENT, bold=True)
        self.text((PAD_X + 104, 12), subtitle, MUTED)
        self.line((0, 39, W, 39), ACCENT)

    def keycap(self, x: int, y: int, key: str, label: str) -> int:
        key_txt = f" {key} "
        key_w = int(self.d.textlength(key_txt, font=KEY))
        self.rect((x, y, x + key_w, y + LINE_H - 2), PANEL)
        self.d.text((x + 2, y + 1), key, font=KEY, fill=ACCENT)
        x += key_w + 5
        self.text((x, y + 1), label, MUTED)
        return x + int(self.d.textlength(label, font=FONT)) + 18

    def fit_end(self, value: str, max_px: int, *, bold: bool = False) -> str:
        active = BOLD if bold else FONT
        if self.d.textlength(value, font=active) <= max_px:
            return value
        mark = "..."
        if self.d.textlength(mark, font=active) >= max_px:
            return ""
        lo, hi = 0, len(value)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.d.textlength(value[:mid] + mark, font=active) <= max_px:
                lo = mid
            else:
                hi = mid - 1
        return value[:lo] + mark

    def fit_middle(self, value: str, max_px: int, *, bold: bool = False) -> str:
        active = BOLD if bold else FONT
        if self.d.textlength(value, font=active) <= max_px:
            return value
        mark = "..."
        if self.d.textlength(mark, font=active) >= max_px:
            return ""

        final_segment = value.rsplit("/", 1)[-1]
        tail_target = min(max(len(final_segment), 8), max(8, len(value) // 2))
        best = self.fit_end(value, max_px, bold=bold)
        for keep in range(len(value) - 1, 0, -1):
            tail = min(tail_target, keep - 5)
            if tail <= 0:
                continue
            head = keep - tail
            candidate = value[:head] + mark + value[-tail:]
            if self.d.textlength(candidate, font=active) <= max_px:
                return candidate
        return best


def section(c: Canvas, y: int, label: str) -> None:
    c.text((PAD_X, y), label, PURPLE, bold=True)
    c.line((PAD_X + 72, y + 13, PAD_X + 470, y + 13), HAIRLINE)


def action_row(c: Canvas, y: int, action_id: str, *, selected: bool = False) -> None:
    action = next(a for a in ACTIONS if a.id == action_id)
    x0, x1 = PAD_X, 618
    if selected:
        c.rect((x0 - 10, y - 1, x1, y + LINE_H - 1), HIGHLIGHT_BG)
        c.rect((x0 - 10, y - 1, x0 - 5, y + LINE_H - 1), ACCENT)
        c.text((x0, y), "▌", ACCENT, bold=True)
    icon_x = PAD_X + 23
    c.emoji((icon_x, y + 1), action.icon)
    label_color = ACCENT if selected else TEXT
    hint_color = TEXT if selected else MUTED
    c.text((PAD_X + 62, y), f"{action.label:<13}", label_color, bold=selected)
    c.text((PAD_X + 212, y), action.hint, hint_color)


def footer(c: Canvas, y: int, *, eval_page: bool = False) -> None:
    c.line((0, y - 12, W, y - 12), HAIRLINE)
    pairs = (
        [("↑↓/jk", "move"), ("←→/hl", "change"), ("↵", "edit/run"), ("s", "run"), ("q", "back")]
        if eval_page
        else [("↑↓/jk", "move"), ("↵", "select"), ("1-9", "jump"), ("d", "preview"), ("q", "quit")]
    )
    x = PAD_X
    for key, label in pairs:
        x = c.keycap(x, y, key, label)


def draw_menu(selected: str = "record", *, mode: str = "REAL") -> Image.Image:
    c = Canvas()
    c.header("mobile-manipulator control")
    c.runtime_chips(52, mode=mode)
    y = 96
    groups = [
        ("HOST", ["host-launch", "host-kill"]),
        ("COLLECT", ["teleop", "record", "replay", "view"]),
        ("LEARN", ["train", "eval"]),
        ("SETUP", ["setup-pi", "sync", "calibrate", "robot-config", "settings"]),
    ]
    for title, ids in groups:
        section(c, y, title)
        y += 28
        for action_id in ids:
            action_row(c, y, action_id, selected=action_id == selected)
            y += 24
        y += 20
    footer(c, H - 46)
    return c.img


def form_row(
    c: Canvas,
    y: int,
    label: str,
    value: str,
    hint: str = "",
    *,
    selected: bool = False,
    clip: str = "end",
) -> None:
    label_x = PAD_X + 20
    value_x = PAD_X + 190
    hint_x = PAD_X + 650
    if selected:
        c.rect((PAD_X - 10, y - 1, W - PAD_X, y + LINE_H - 1), HIGHLIGHT_BG)
        c.rect((PAD_X - 10, y - 1, PAD_X - 5, y + LINE_H - 1), ACCENT)
        c.text((PAD_X, y), "▌", ACCENT, bold=True)
    label_color = ACCENT if selected else MUTED
    value_color = ACCENT if selected else TEXT
    value_max = (hint_x - value_x - 18) if hint else (W - PAD_X - value_x)
    clipped = c.fit_middle(value, value_max, bold=selected) if clip == "middle" else c.fit_end(
        value,
        value_max,
        bold=selected,
    )
    c.text((label_x, y), f"{label:<15}", label_color, bold=selected)
    c.text((value_x, y), clipped, value_color, bold=selected)
    if hint:
        c.text((hint_x, y), c.fit_end(hint, W - PAD_X - hint_x), MUTED)


def draw_eval(*, mode: str = "PREVIEW", selected: str = "policy") -> Image.Image:
    c = Canvas()
    c.header("run policy")
    c.runtime_chips(52, mode=mode)
    policy = f"{PUBLIC_POLICY}  default"
    form_row(
        c,
        98,
        "Policy",
        policy,
        "enter to pick a checkpoint",
        selected=selected == "policy",
        clip="middle",
    )
    form_row(c, 124, "Task", PUBLIC_TASK, "enter to edit task")
    form_row(c, 150, "Backend", "‹ sync ›", "one policy forward per control tick")
    form_row(c, 176, "Duration", "saved default", "0 = saved default")
    form_row(c, 202, "Display", "‹ off ›", "off lowers CPU")
    form_row(c, 238, "▶ Run policy", "validate preflight and launch rollout", selected=selected == "start")

    y = 316
    section(c, y, "RUN SUMMARY")
    rows = [
        ("Policy", f"{PUBLIC_POLICY} default", SAND),
        ("Task", PUBLIC_TASK, TEXT),
        ("Backend", "sync · one policy forward per control tick", TEXT),
        ("Duration", "saved default", TEXT),
        ("Display", "off · lower CPU", TEXT),
        ("Device", "CUDA available; CPU fallback if needed", TEXT),
        ("Host", "required; start Pi host in another terminal", SAND),
        ("Mode", "PREVIEW · wrapper prints argv" if mode == "PREVIEW" else "REAL · controls robot", WARNING if mode == "PREVIEW" else SUCCESS),
    ]
    y += 30
    for label, value, color in rows:
        c.text((PAD_X + 20, y), f"{label:<10}", MUTED)
        c.text(
            (PAD_X + 132, y),
            c.fit_end(value, W - PAD_X - (PAD_X + 132)),
            color,
            bold=color in (SAND, WARNING, SUCCESS),
        )
        y += 24
    footer(c, H - 46, eval_page=True)
    return c.img


def draw_preview() -> Image.Image:
    c = Canvas()
    c.header("run policy")
    c.runtime_chips(52, mode="PREVIEW")
    c.text((PAD_X, 100), "PREVIEW argv", PURPLE, bold=True)
    c.line((PAD_X + 132, 113, W - PAD_X, 113), HAIRLINE)
    lines = [
        "$ bash scripts/eval.sh --dry-run",
        f"--policy {PUBLIC_POLICY}",
        "--backend sync",
        "--duration 0",
        "--display off",
        f"--gpu {PUBLIC_GPU}",
    ]
    y = 138
    for i, line in enumerate(lines):
        text = c.fit_middle(line, W - PAD_X - (PAD_X + 20)) if i == 1 else c.fit_end(
            line,
            W - PAD_X - (PAD_X + 20),
            bold=i == 0,
        )
        c.text((PAD_X + 20, y), text, SAND if i == 0 else TEXT, bold=i == 0)
        y += 28
    c.text((PAD_X + 20, y + 24), "No robot command runs while preview mode is on.", MUTED)
    footer(c, H - 46, eval_page=True)
    return c.img


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    hero = draw_menu("record", mode="REAL")
    hero.save(HERO)
    frames = [
        draw_menu("record", mode="REAL"),
        draw_menu("eval", mode="PREVIEW"),
        draw_eval(mode="PREVIEW", selected="policy"),
        draw_eval(mode="PREVIEW", selected="start"),
        draw_preview(),
    ]
    frames[0].save(
        GIF,
        save_all=True,
        append_images=frames[1:],
        duration=[900, 900, 1100, 1100, 1400],
        loop=0,
        optimize=True,
    )
    print(HERO)
    print(GIF)


if __name__ == "__main__":
    main()
