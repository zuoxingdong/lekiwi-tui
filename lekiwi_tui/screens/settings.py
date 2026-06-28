"""settings.py — SettingsScreen: edit the _launcher knobs -> lekiwi.yaml (port).

Edits land in a WORKING COPY (Config.copy()); nothing touches the live config or lekiwi.yaml
until Save, which round-trips via Config.save (lossless ruamel: anchors/merges/comments
survive). The POLICY_PATH chooser is the four-way Auto / checkpoint / Browse / Type chain
(ConfirmModalState -> DirPicker | PromptModalState, an Invoke async flow). `e` opens
lekiwi.yaml in $EDITOR (suspend + guarded reload). q discards (with a confirm when dirty).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyratatui import Constraint, Direction, Layout, Line, Paragraph, Span, Style, Text

from .. import CFG_FILE
from ..config import CONFIG_SPEC, Config, resolve_editor
from ..framework import theme
from ..framework.events import DOWN, ENTER, ESC, LEFT, RIGHT, UP, Key
from ..framework.modals import ConfirmModalState, PromptModalState
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from ..policies import discover_policies
from ..widgets.pickers import DirPicker
from .chrome import keycap_hint_line, option_line

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

#: The run_headless hook name used by direct no-TTY CLI dispatch.
HEADLESS_HOOK = "run_headless"

RULE = "─" * 54
_LABEL_W = 18
_SECTION_BEFORE = {"LAPTOP_ENV": "LAPTOP", "LEKIWI_HOST": "ROBOT HOST", "POLICY_PATH": "EVAL"}
_AUTO = "Auto - use the newest checkpoint under POLICY_ROOT"
_BROWSE = "Browse directories…"
_TYPE = "Type a path or model repo id…"

# "Edit config" lead-in of the sub-line is bold TEXT (bash 1040 `_sub_line`): there is no
# theme constant for it, so compose it here rather than mutate a shared theme.* style.
_SUB_LEAD_STYLE = Style().fg(theme.TEXT).bold()


def _collapse_home(value: str) -> str:
    home = str(Path.home())
    if value == home:
        return "~"
    if value.startswith(home + "/"):
        return "~" + value[len(home):]
    return value


def _expand_home(value: str) -> str:
    if value == "~":
        return str(Path.home())
    if value.startswith("~/"):
        return str(Path.home()) + value[1:]
    return value


def _enum_options(kind: str) -> list[str]:
    return kind[len("enum:"):].split(",")


def _cycle_enum(cur: str, kind: str, step: int) -> str:
    opts = _enum_options(kind)
    n = len(opts)
    idx = next((i for i, o in enumerate(opts) if o == cur), 0)
    return opts[(idx + step + n) % n]


class SettingsScreen(ScreenState):
    """Settings editor over CONFIG_SPEC; edits land in a working copy until Save."""

    title = "settings"

    def __init__(self, app: "App", ctx: "Context", *, cfg_path: Path = CFG_FILE) -> None:
        self.app = app
        self.ctx = ctx
        self._cfg_path = cfg_path
        self._work: dict[str, str] = dict(ctx.cfg.values)
        self._env_set = set(ctx.cfg.env_set)
        self._n = len(CONFIG_SPEC)
        self.cursor = 0           # 0..n-1 fields, n = Save row
        self.dirty = False
        self.message = ""
        self._msg_ok = False

    # ── input ─────────────────────────────────────────────────────────────────
    def handle_key(self, key: "Key") -> Any:
        name = key.name
        if name in (ESC, "q"):
            if not self.dirty:
                return Pop()
            return Invoke(self._confirm_back)
        if name == "e":
            return Invoke(self._edit_yaml)
        if name in (UP, "k"):
            self.message = ""; self.cursor = (self.cursor - 1) % (self._n + 1); return Nothing
        if name in (DOWN, "j"):
            self.message = ""; self.cursor = (self.cursor + 1) % (self._n + 1); return Nothing
        if name in (LEFT, "h", RIGHT, "l"):
            self._adjust(-1 if name in (LEFT, "h") else 1); return Nothing
        if name == ENTER:
            return self._activate()
        return Nothing

    def _adjust(self, step: int) -> None:
        self.message = ""
        if self.cursor >= self._n:
            return
        f = CONFIG_SPEC[self.cursor]
        if f.kind.startswith("enum:"):
            self._work[f.key] = _cycle_enum(self._work[f.key], f.kind, step); self.dirty = True
        elif f.kind == "int":
            s = step * 60 if f.key == "CONNECTION_TIME" else step
            cur = int(self._work[f.key]) if str(self._work[f.key]).isdigit() else 0
            self._work[f.key] = str(max(0, cur + s)); self.dirty = True

    def _activate(self) -> Any:
        self.message = ""
        if self.cursor >= self._n:
            self._save(); return Nothing
        f = CONFIG_SPEC[self.cursor]
        if f.kind.startswith("enum:"):
            self._work[f.key] = _cycle_enum(self._work[f.key], f.kind, 1); self.dirty = True
            return Nothing
        if f.key == "POLICY_PATH":
            return Invoke(self._pick_policy)
        if f.kind == "path":
            return Invoke(lambda: self._browse_path(f.key))
        return Invoke(lambda: self._edit_text(f.key, f.kind))

    # ── async flows ─────────────────────────────────────────────────────────────
    async def _confirm_back(self) -> None:
        choice = await self.app.run_modal(ConfirmModalState(
            "Discard unsaved changes?", ["Discard", "Cancel"]))
        if choice == "Discard":
            self.app.notify("Changes discarded; lekiwi.yaml was not written.")
            self.app.pop()

    async def _browse_path(self, key: str) -> None:
        result = await self.app.run_modal(DirPicker(self._work[key], f"Pick directory for {key}"))
        if result is not None and result != self._work[key]:
            self._work[key] = result; self.dirty = True

    async def _edit_text(self, key: str, kind: str) -> None:
        ans = await self.app.run_modal(PromptModalState(
            key, value=self._work[key], hint="⏎ apply (blank keeps current) · esc cancel"))
        if ans is None or not ans:
            return
        ans = _expand_home(ans)
        if kind == "int" and not ans.isdigit():
            self.message = f"✗ {key} must be a whole number"; self._msg_ok = False
        elif ans != self._work[key]:
            self._work[key] = ans; self.dirty = True

    async def _pick_policy(self) -> None:
        root = Path(self._work.get("POLICY_ROOT") or "")
        rels: list[str] = []
        for p in discover_policies(root):
            try:
                rels.append(str(p.relative_to(root)))
            except ValueError:
                rels.append(str(p))
        choice = await self.app.run_modal(ConfirmModalState(
            f"Default policy checkpoint - root: {_collapse_home(str(root))}",
            [_AUTO, *rels, _BROWSE, _TYPE]))
        if choice is None:
            return
        sel: str | None = None
        if choice == _AUTO:
            sel = ""
        elif choice in rels:
            sel = choice
        elif choice == _BROWSE:
            picked = await self.app.run_modal(DirPicker(
                root, "Pick checkpoint directory (needs config.json + model.safetensors)"))
            if picked is not None:
                sel = picked
                prefix = str(root) + "/"
                if sel.startswith(prefix):
                    sel = sel[len(prefix):]
        elif choice == _TYPE:
            ans = await self.app.run_modal(PromptModalState(
                "Policy path",
                hint="relative path under POLICY_ROOT · absolute path · model repo id; ⏎ keeps current"))
            if ans:
                sel = _expand_home(ans)
        if sel is not None and sel != self._work["POLICY_PATH"]:
            self._work["POLICY_PATH"] = sel; self.dirty = True

    async def _edit_yaml(self) -> None:
        await self.app.suspend([resolve_editor(), str(self._cfg_path)])
        try:
            new_cfg = Config.load(self._cfg_path)
        except Exception as exc:  # noqa: BLE001 - never crash on a hand-saved bad file
            self.app.notify(f"lekiwi.yaml not reloaded (invalid): {exc}", "error")
            return
        self.ctx.cfg = new_cfg
        self._work = dict(new_cfg.values)
        self._env_set = set(new_cfg.env_set)
        self.dirty = False
        self.message = f"✓ reloaded {_collapse_home(str(self._cfg_path))}"; self._msg_ok = True

    # ── save (sync) ─────────────────────────────────────────────────────────────
    def _save(self) -> None:
        new_cfg = Config(values=dict(self._work), env_set=set(self._env_set))
        try:
            new_cfg.save(self._cfg_path)
        except OSError:
            self.message = f"✗ could not write {_collapse_home(str(self._cfg_path))}"; self._msg_ok = False
            return
        self.ctx.cfg.values.update(self._work)
        self.dirty = False
        self.message = f"✓ saved to {_collapse_home(str(self._cfg_path))}"; self._msg_ok = True

    # ── view ────────────────────────────────────────────────────────────────────
    def _display_value(self, key: str, kind: str) -> str:
        val = self._work[key]
        if kind.startswith("enum:"):
            if not val:
                val = _enum_options(kind)[0]
            return theme.choice(val)
        disp = _collapse_home(val)
        if not disp:
            return "auto (newest under POLICY_ROOT)" if key == "POLICY_PATH" else "(empty)"
        return disp

    def draw(self, frame: Any, area: Any) -> None:
        frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
        # Rows: 0 header · 1 heavy rule · 2 sub-line (config path) · 3 body · 4 info/active-field
        # hint · 5 message (conditional) · 6 key-legend. The sub-line + info row are the two
        # NEW rows; body stays the flexible fill, message + legend keep their behavior.
        rows = (Layout().direction(Direction.Vertical).constraints(
            [Constraint.length(1), Constraint.length(1), Constraint.length(1),
             Constraint.fill(1), Constraint.length(1), Constraint.length(1),
             Constraint.length(1)]).split(area))
        hdr = [Span(f"{theme.title_mark()} LEKIWI", theme.TITLE_STYLE), Span("  settings", theme.SUBTITLE_STYLE)]
        if self.dirty:
            hdr.append(Span(f"   {theme.status_dot()} unsaved", theme.STATUS_VALUE_STYLE))
        frame.render_widget(Paragraph(Text([Line(hdr)])).style(theme.BASE_STYLE), rows[0])
        frame.render_widget(Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1])
        frame.render_widget(self._sub(), rows[2])
        frame.render_widget(self._body(rows[3].width), rows[3])
        frame.render_widget(self._info(), rows[4])
        if self.message:
            style = theme.OK_STYLE if self._msg_ok else theme.ERR_STYLE
            frame.render_widget(Paragraph(Text([Line([Span(self.message, style)])])
                                          ).style(theme.BASE_STYLE), rows[5])
        frame.render_widget(self._hint(), rows[6])

    def _body(self, width: int = 120) -> Paragraph:
        lines: list[Line] = []
        for i, f in enumerate(CONFIG_SPEC):
            section = _SECTION_BEFORE.get(f.key)
            if section:
                lines.append(Line([Span(f"{section} ", theme.SECTION_STYLE),
                                   Span(theme.rule(max(4, 44 - len(section))), theme.BORDER_STYLE)]))
            active = i == self.cursor
            line = option_line(
                f.key,
                self._display_value(f.key, f.kind),
                focused=active,
                label_width=_LABEL_W,
                width=width,
                clip="middle" if f.kind == "path" else "end",
            )
            if f.key in self._env_set:
                line.push_span(Span(
                    "   [env]",
                    theme.HIGHLIGHT_MUTED_STYLE if active else theme.STATUS_VALUE_STYLE,
                ))
            lines.append(line)
        save_active = self.cursor >= self._n
        lines.append(Line([Span("", theme.BASE_STYLE)]))
        lines.append(option_line(
            f"{theme.action_icon('💾')} Save settings",
            "write launcher settings",
            focused=save_active,
            label_width=22,
            width=width,
            label_unfocused_style=theme.OK_STYLE,
        ))
        return Paragraph(Text(lines)).style(theme.BASE_STYLE)

    def _sub(self) -> Paragraph:
        # "Edit config  → <yaml> · env vars at launch override" (bash 1040 _sub_line). The
        # lead-in is bold TEXT; the rest (arrow, path, caveat) is muted.
        return Paragraph(Text([Line([
            Span("Launcher settings", _SUB_LEAD_STYLE),
            Span("  → ", theme.MUTED_STYLE),
            Span(_collapse_home(str(self._cfg_path)), theme.MUTED_STYLE),
            Span("  ·  env vars set at launch override these values", theme.MUTED_STYLE),
        ])])).style(theme.BASE_STYLE)

    def _info(self) -> Paragraph:
        # The active field's hint (muted), or the Save-row note. When the active field is
        # env-overridden, prepend the [env override] provenance caveat (SAND) (bash 1058-1064
        # _info_text). Reads self._env_set — kept in sync by _edit_yaml — to match _body.
        if self.cursor >= self._n:
            return Paragraph(Text([Line([Span(
                "writes launcher settings to lekiwi.yaml · launch env vars still win",
                theme.MUTED_STYLE)])])).style(theme.BASE_STYLE)
        f = CONFIG_SPEC[self.cursor]
        spans: list[Span] = []
        if f.key in self._env_set:
            spans.append(Span("[env override] ", theme.STATUS_VALUE_STYLE))
            spans.append(Span("an env var set this; it wins until unset  ·  ", theme.MUTED_STYLE))
        spans.append(Span(f.hint, theme.MUTED_STYLE))
        return Paragraph(Text([Line(spans)])).style(theme.BASE_STYLE)

    def _hint(self) -> Paragraph:
        return keycap_hint_line([
            ("↑↓/jk", "move"),
            ("←→", "adjust"),
            ("⏎", "edit/browse/save"),
            ("e", "edit file"),
            ("q", "back"),
        ])


def run_headless(ctx, extra: list[str]) -> int:  # noqa: ANN001
    """No-TTY settings dispatch (bash do_settings 1150-1157): print the effective config as
    plain ``KEY=value``, env-overridden keys annotated with a trailing ``\\t# env override``.
    A leading comment header names the yaml path (~-collapsed). No save, so there is no
    real-file-write risk. Returns 0.

    Ported from the Textual ``run_headless(app, extra)``; this port threads config through
    ``ctx`` (there is no ``app.cfg``), so it takes ``ctx`` and reads ``ctx.cfg.values`` /
    ``ctx.cfg.env_set``. ``__main__`` calls this hook for direct no-TTY settings dispatch."""
    cfg = ctx.cfg
    print(f"# effective lekiwi config (env > {_collapse_home(str(CFG_FILE))} > built-in defaults)")
    for f in CONFIG_SPEC:
        val = cfg.values[f.key]
        if f.key in cfg.env_set:
            print(f"{f.key}={val}\t# env override")
        else:
            print(f"{f.key}={val}")
    return 0


__all__ = ["SettingsScreen", "run_headless", "HEADLESS_HOOK"]
