#!/usr/bin/env python3
"""Generate the README workflow GIF (one asset; its first frame is the hero).

SYNTHETIC BY DESIGN, for sanitisation: nothing here is captured from a live terminal, so a
recording can never carry a hostname, an IP, a home path, a private checkpoint name — or, in
the camera-preview frame, a photograph of somebody's room. Every value is a PUBLIC_* constant
below and every camera tile is drawn from shapes.

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
GIF = ASSETS / "lekiwi-tui-dry-run.gif"

FONT_REG = "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf"
KEY_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
EMOJI_FONT = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"

W, H = 1180, 668
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
PUBLIC_ROBOT = "lekiwi_pincopen"
PUBLIC_POLICY = "models/lekiwi-policy/checkpoints/latest/pretrained_model"
PUBLIC_TASK = "Pick up the object and place it in the tray"
PUBLIC_LEROBOT = "0.6.1"
PUBLIC_DATASET = "local/demo-plate"
#: Laptop vitals for the status card. Plausible, and nothing here identifies a machine.
PUBLIC_VITALS = "cpu 21%   ram 9.4/31 GB   RTX GPU 12%   vram 1.2/8 GB"


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

        # the LIVE robot chip: type + green dot + remaining session time
        label = " robot "
        dot = "● "
        val = f"{PUBLIC_ROBOT} "
        left = "27:41 "
        label_w = int(self.d.textlength(label, font=FONT))
        val_w = int(self.d.textlength(val, font=BOLD))
        dot_w = int(self.d.textlength(dot, font=BOLD))
        left_w = int(self.d.textlength(left, font=FONT))
        self.rect((x, y, x + label_w + val_w + dot_w + left_w, y + LINE_H - 2), PANEL)
        self.text((x + 2, y + 1), "robot", MUTED)
        self.text((x + label_w, y + 1), val, SAND, bold=True)
        self.text((x + label_w + val_w, y + 1), dot, SUCCESS, bold=True)
        self.text((x + label_w + val_w + dot_w, y + 1), "27:41", MUTED)
        x += label_w + val_w + dot_w + left_w + 10

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


# Digit badges cover the daily-driver rows only (menu.py _JUMPABLE rule).
JUMPABLE = [a.id for a in ACTIONS if a.section != "SETUP"]


# ── the current chrome: cards, a status card, and the hint slot ───────────────


def section(c: Canvas, y: int, label: str) -> None:
    """A quiet group header with a hairline, as the screens draw it."""
    c.text((PAD_X, y), label, PURPLE, bold=True)
    c.line((PAD_X + 130, y + 13, W - PAD_X, y + 13), HAIRLINE)


def footer(c: Canvas, y: int, *, eval_page: bool = False) -> None:
    """Back-compat wrapper for the frames that predate hint_row."""
    keys = ([("↑↓/jk", "move"), ("←→", "change"), ("↵", "edit/run"), ("s", "run"), ("q", "back")]
            if eval_page else
            [("↑↓/jk", "move"), ("↵", "select"), ("d", "preview"), ("q", "quit")])
    hint_row(c, y, "", keys)


def card(c: Canvas, box: tuple[int, int, int, int], title: str) -> None:
    """A titled panel: hairline border with the title breaking the top edge."""
    x0, y0, x1, y1 = box
    c.rect((x0, y0, x1, y1), SURFACE, outline=HAIRLINE)
    c.text((x0 + 12, y0 - 9), f" {title} ", PURPLE, bold=True)


def status_card(c: Canvas, y: int, *, mode: str = "REAL") -> int:
    """The ROBOT card: is the robot reachable, can this laptop take another run, and which
    hardware/env/lerobot is about to be driven."""
    box = (PAD_X, y, W - PAD_X, y + 3 * LINE_H + 18)
    card(c, box, "ROBOT")
    row = y + 12
    c.text((PAD_X + 14, row), "●", SUCCESS, bold=True)
    c.text((PAD_X + 34, row), "host up", SUCCESS)
    c.text((PAD_X + 130, row), "session 27:41 left", MUTED)
    x = W - PAD_X - 130
    c.rect((x, row - 1, W - PAD_X - 14, row + LINE_H - 3), PANEL)
    c.text((x + 8, row), "mode", MUTED)
    c.text((x + 58, row), mode, SUCCESS if mode == "REAL" else WARNING, bold=True)
    row += LINE_H
    c.text((PAD_X + 14, row), "laptop", MUTED)
    c.text((PAD_X + 104, row), PUBLIC_VITALS, TEXT)
    row += LINE_H
    c.text((PAD_X + 14, row), "robot", MUTED)
    c.text((PAD_X + 104, row), PUBLIC_ROBOT, TEXT)
    c.text((PAD_X + 320, row), "env", MUTED)
    c.text((PAD_X + 368, row), PUBLIC_ENV, TEXT)
    c.text((PAD_X + 520, row), "lerobot", MUTED)
    c.text((PAD_X + 608, row), PUBLIC_LEROBOT, TEXT)
    return box[3] + 26


def card_row(c: Canvas, x: int, y: int, width: int, action_id: str, *, selected: bool) -> int:
    """One action inside a card: icon, digit, label, then its description underneath."""
    action = next(a for a in ACTIONS if a.id == action_id)
    if selected:
        c.rect((x + 4, y - 2, x + width - 6, y + LINE_H - 2), HIGHLIGHT_BG)
        c.rect((x + 4, y - 2, x + 8, y + LINE_H - 2), ACCENT)
    c.emoji((x + 16, y + 1), action.icon)
    if action_id in JUMPABLE:
        c.text((x + 44, y), str(JUMPABLE.index(action_id) + 1), ACCENT, bold=selected)
    c.text((x + 66, y), action.label, ACCENT if selected else TEXT, bold=selected)
    c.text((x + 66, y + LINE_H - 2), c.fit_end(action.hint, width - 84), MUTED)
    return y + 2 * LINE_H - 2


def hint_row(c: Canvas, y: int, hint: str, keys: list[tuple[str, str]]) -> None:
    """The footer hint slot: the sentence on the left, keycaps right-aligned."""
    c.line((0, y - 14, W, y - 14), HAIRLINE)
    width = sum(int(c.d.textlength(f" {k} ", font=KEY)) + int(c.d.textlength(lab, font=FONT)) + 23
                for k, lab in keys)
    x = W - PAD_X - width
    # the sentence gets whatever the keycaps leave; the real screen sheds labels here, and an
    # asset that overlaps its own text is worse than one that says less
    c.text((PAD_X, y), c.fit_end(hint, max(0, x - PAD_X - 24)), MUTED)
    for key, label in keys:
        x = c.keycap(x, y, key, label)


def draw_menu(selected: str = "record", *, mode: str = "REAL") -> Image.Image:
    """The card-grid menu: status card, then four cards two across, then the SETUP strip."""
    c = Canvas()
    c.header("mobile-manipulator control")
    y = status_card(c, 56, mode=mode)

    # Membership comes from the registry (menu.py's own source), so a new action appears here
    # instead of being silently missing from the asset — which is exactly what happened when
    # this was a hand-written map and `edit-dataset` did not match the id it guessed.
    grid = [["HOST", "COLLECT"], ["DATA", "LEARN"]]
    members: dict[str, list[str]] = {}
    for a in ACTIONS:
        members.setdefault(getattr(a, "card", None) or a.section, []).append(a.id)
    known = {a.id for a in ACTIONS}
    col_w = (W - 2 * PAD_X - 18) // 2
    for row_titles in grid:
        rows = max(len(members[t]) for t in row_titles)
        height = rows * (2 * LINE_H - 2) + 22
        for i, title in enumerate(row_titles):
            x0 = PAD_X + i * (col_w + 18)
            card(c, (x0, y, x0 + col_w, y + height), title)
            ry = y + 14
            for action_id in members[title]:
                if action_id not in known:      # a card member that no longer exists
                    continue
                ry = card_row(c, x0, ry, col_w, action_id, selected=action_id == selected)
        y += height + 26

    strip = [a.id for a in ACTIONS if a.section == "SETUP"]
    card(c, (PAD_X, y, W - PAD_X, y + LINE_H + 16), "SETUP")
    x = PAD_X + 16
    for action_id in strip:
        action = next(a for a in ACTIONS if a.id == action_id)
        c.emoji((x, y + 13), action.icon)
        c.text((x + 26, y + 12), action.label, TEXT)
        x += 26 + int(c.d.textlength(action.label, font=FONT)) + 28
    hint_row(c, H - 40, "↓ walks a column · a digit runs that action",
             [("↑↓/jk", "move"), ("←→", "column"), ("↵", "select"), ("d", "preview"), ("q", "quit")])
    return c.img


def setting_row(c: Canvas, y: int, label: str, value: str, hint: str = "", *,
                selected: bool = False, kind: str = "plain") -> int:
    """One setting per row. `stepper` renders ‹ value ›, `toggle` renders two pills."""
    if selected:
        c.rect((PAD_X - 6, y - 2, W - PAD_X, y + LINE_H - 2), HIGHLIGHT_BG)
        c.rect((PAD_X - 6, y - 2, PAD_X - 2, y + LINE_H - 2), ACCENT)
    c.text((PAD_X + 16, y), label, ACCENT if selected else MUTED, bold=selected)
    vx = PAD_X + 190
    if kind == "stepper":
        c.text((vx, y), "‹", MUTED)
        c.text((vx + 20, y), value, ACCENT if selected else TEXT, bold=True)
        c.text((vx + 26 + int(c.d.textlength(value, font=BOLD)), y), "›", MUTED)
    elif kind == "toggle":
        on, off = value.split("|")
        w_on = int(c.d.textlength(f" {on} ", font=BOLD))
        c.rect((vx, y - 1, vx + w_on, y + LINE_H - 3), PANEL)
        c.text((vx + 6, y), on, SUCCESS, bold=True)
        c.text((vx + w_on + 14, y), off, MUTED)
    else:
        c.text((vx, y), c.fit_middle(value, 640, bold=selected), ACCENT if selected else TEXT,
               bold=selected)
    if hint:
        c.text((PAD_X + 700, y), c.fit_end(hint, W - PAD_X - (PAD_X + 700)), MUTED)
    return y + LINE_H + 4


def header_note(c: Canvas, text: str, color: str = MUTED, *, bold: bool = False) -> None:
    """Right-aligned header text, measured rather than guessed."""
    c.text((W - PAD_X - int(c.d.textlength(text, font=BOLD if bold else FONT)), 12), text, color,
           bold=bold)


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
        ("Host", "required; Start host from the menu (q keeps it running)", SAND),
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


def draw_teleop() -> Image.Image:
    """Teleoperate: the reference form. Duration / FPS / Display, then Start with the plan
    sentence the screen actually renders."""
    c = Canvas()
    c.header("teleoperate")
    header_note(c, f"{PUBLIC_ROBOT} · leader arm on /dev/ttyACM0")
    y = 70
    c.text((PAD_X, y), "SESSION", PURPLE, bold=True)
    c.line((PAD_X + 110, y + 13, W - PAD_X, y + 13), HAIRLINE)
    y += 30
    y = setting_row(c, y, "Duration", "0", "0 = drive until you press Ctrl+C", kind="stepper")
    y = setting_row(c, y, "FPS", "30", "the robot's control-loop rate", kind="stepper",
                    selected=True)
    y = setting_row(c, y, "Display", "off|on", "mirror the cameras in a window (off lowers CPU)",
                    kind="toggle")
    y += 18
    c.rect((PAD_X - 6, y - 2, W - PAD_X, y + LINE_H - 2), HIGHLIGHT_BG)
    c.rect((PAD_X - 6, y - 2, PAD_X - 2, y + LINE_H - 2), SUCCESS)
    c.text((PAD_X + 16, y), "▶ Start", SUCCESS, bold=True)
    c.text((PAD_X + 190, y), "leader arm + wasd·zx base · no recording · full-TTY session", TEXT)
    y += LINE_H + 26
    c.text((PAD_X + 16, y), "The leader arm drives the follower; wasd turns the base, zx spins it,", MUTED)
    c.text((PAD_X + 16, y + LINE_H), "r/f raise and lower the speed. Ctrl+C ends the session.", MUTED)
    hint_row(c, H - 40, "the focused row explains itself here · ←→ changes it",
             [("↑↓/jk", "move"), ("←→", "change"), ("↵", "edit"), ("s", "start"), ("q", "back")])
    return c.img


def draw_record_new() -> Image.Image:
    """Record on the shared chrome: one setting per row, steppers, pill toggles."""
    c = Canvas()
    c.header("record")
    header_note(c, f"{PUBLIC_DATASET} · 60 episodes · 1.4 GB")
    y = 70
    c.text((PAD_X, y), "DATASET", PURPLE, bold=True)
    c.line((PAD_X + 110, y + 13, W - PAD_X, y + 13), HAIRLINE)
    y += 30
    y = setting_row(c, y, "Name", "demo-plate", "saves under datasets/<name>")
    y = setting_row(c, y, "Task", PUBLIC_TASK, "enter to edit")
    y = setting_row(c, y, "Episodes", "10", "takes to record before it stops", kind="stepper")
    y = setting_row(c, y, "Episode time", "60 s", "recording time per episode · ←→ ±5",
                    kind="stepper", selected=True)
    y = setting_row(c, y, "Reset time", "8 s", "pause between takes, to reset the scene",
                    kind="stepper")
    y += 12
    c.text((PAD_X, y), "CAPTURE", PURPLE, bold=True)
    c.line((PAD_X + 110, y + 13, W - PAD_X, y + 13), HAIRLINE)
    y += 30
    y = setting_row(c, y, "Resume", "append|fresh", "60 recorded — Start appends after them",
                    kind="toggle")
    y = setting_row(c, y, "Streaming", "on|off", "encode while recording, for faster saves",
                    kind="toggle")
    y = setting_row(c, y, "Display", "off|on", "live camera view (off lowers CPU)", kind="toggle")
    y = setting_row(c, y, "Encoder", "h264_nvenc · gpu", "the configured rgb_encoder vcodec")
    y += 16
    c.rect((PAD_X - 6, y - 2, W - PAD_X, y + LINE_H - 2), HIGHLIGHT_BG)
    c.rect((PAD_X - 6, y - 2, PAD_X - 2, y + LINE_H - 2), SUCCESS)
    c.text((PAD_X + 16, y), "▶ Start", SUCCESS, bold=True)
    c.text((PAD_X + 190, y), "10 takes × 60 s, appending after episode 60 · arrows end a take",
           TEXT)
    y += LINE_H + 26
    c.text((PAD_X + 16, y), "→ ends a take and starts the next, ← re-records the last one,", MUTED)
    c.text((PAD_X + 16, y + LINE_H), "Esc stops early. The dataset is written as you go.", MUTED)
    hint_row(c, H - 40, "one setting per row · ←→ changes the focused one",
             [("↑↓/jk", "move"), ("←→", "change"), ("↵", "edit"), ("s", "start"), ("q", "back")])
    return c.img


def draw_record() -> Image.Image:
    c = Canvas()
    c.header("record")
    c.runtime_chips(52, mode="REAL")
    # destination + dataset panel (the chips idiom): what is on disk, what Start will do
    x = c.chip(PAD_X, 96, "repo", "local/lekiwi-demo")
    c.text((x + 6, 97), "~/robots/datasets/lekiwi-demo", MUTED)
    x = PAD_X
    x = c.chip(x, 122, "episodes", "60")
    x = c.chip(x, 122, "length", "24.7 min")
    x = c.chip(x, 122, "size", "1.4 GB")
    x = c.chip(x, 122, "updated", "16:53")
    c.text((PAD_X, 150), "resume on — will append after episode 60", SUCCESS)
    c.text((PAD_X, 174), "● host live — ready to record", SUCCESS, bold=True)
    form_row(c, 214, "Dataset", "lekiwi-demo", "saves under datasets/<name>")
    form_row(c, 240, "Task", "Pick up the cube and place it in the basket.", "enter to edit")
    form_row(c, 266, "Episodes", "10", "←→ ±1 · enter to type")
    form_row(c, 292, "View", "‹ hud ›", "in-page log + episode HUD", selected=True)
    form_row(c, 328, "▶ Start", "validate dataset and launch capture")
    footer(c, H - 46, eval_page=True)
    return c.img


def draw_preview() -> Image.Image:
    """PREVIEW mode: `d` from anywhere makes every action print its argv instead of running.
    The same launcher builds it, so what you read is what would have executed."""
    c = Canvas()
    c.header("record")
    header_note(c, "mode PREVIEW", WARNING, bold=True)
    c.text((PAD_X, 70), "PREVIEW argv", PURPLE, bold=True)
    c.line((PAD_X + 150, 83, W - PAD_X, 83), HAIRLINE)
    # What a dry-run really prints: the lerobot argv the launcher would exec, one token per
    # line. Paths are shown relative — the live output is absolute, and an absolute path is
    # exactly the kind of thing this asset must never carry.
    lines = [
        "$ scripts/record.sh --dry-run --name demo-plate --episodes 10",
        "  python scripts/lerobot_record_kbd.py",
        "  --config_path .lekiwi-cache/record.yaml",
        "  --dataset.repo_id=local/demo-plate",
        "  --dataset.root=datasets/demo-plate",
        f"  --dataset.single_task='{PUBLIC_TASK}'",
        "  --dataset.num_episodes=10  --dataset.episode_time_s=60",
        "  --dataset.streaming_encoding=true  --display_data=false",
    ]
    y = 108
    for i, line in enumerate(lines):
        c.text((PAD_X + 20, y), c.fit_end(line, W - PAD_X - (PAD_X + 20), bold=i == 0),
               SAND if i == 0 else TEXT, bold=i == 0)
        y += 26
    y += 16
    c.text((PAD_X + 20, y), "Nothing reaches the robot while preview is on, so a launcher change",
           MUTED)
    c.text((PAD_X + 20, y + LINE_H), "can be read before it is trusted. Press d again to arm.",
           MUTED)
    hint_row(c, H - 40, "d toggles preview from any screen",
             [("d", "real/preview"), ("q", "back")])
    return c.img


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    frames = [
        draw_menu("teleop", mode="REAL"),      # hero: the card grid + status card
        draw_teleop(),
        draw_menu("record", mode="REAL"),
        draw_record_new(),
        draw_preview(),
    ]
    frames[0].save(
        GIF,
        save_all=True,
        append_images=frames[1:],
        duration=[1600, 1600, 1000, 1800, 1600],
        loop=0,
        optimize=True,
    )
    print(GIF)


if __name__ == "__main__":
    main()
