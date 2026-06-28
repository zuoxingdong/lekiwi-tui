"""CLI entry — ``lekiwi [action] [extra…]`` or ``python -m lekiwi_tui``.

Mirrors the Textual app's ``__main__`` dispatch:
  * no args            → launch the immediate-mode menu (needs a TTY).
  * action id / alias  → run that action directly (a screen opens it; a suspend action
                         runs without the TUI loop, dry-run safe).
  * -h / --help        → print the action list + aliases.
  * unknown action     → error.

App bootstrap order (the App, its root screen, and the Dispatcher reference each other):
build the Dispatcher and the root screen first (a screen never uses ``app`` in ``__init__``),
construct the App with the dispatcher's hook, then bind the App back into both. See
:func:`_make_app`.
"""
from __future__ import annotations

import asyncio
import difflib
import importlib
import os
import sys

from . import CFG_FILE, EXAMPLE_CFG_FILE, ROOT
from .app_registry import ACTIONS, ACTIONS_BY_ID, ALIASES, resolve_action
from .context import load_context
from .dispatch import Dispatcher
from .framework.app import App
from .screens.menu import MenuScreen


def _make_app(ctx, root):
    """Wire a root :class:`ScreenState` into a fully-bound App + Dispatcher.

    The root is built with ``app=None`` (screens must not touch ``app`` in ``__init__``);
    we inject the App into both the dispatcher and the root just before returning, breaking
    the App↔root↔dispatcher construction cycle.
    """
    disp = Dispatcher(ctx)
    app = App(root, run_action=disp.run_action, fps=30.0)
    disp.bind(app)
    # Give the root its app handle now that the App exists (unused during __init__).
    try:
        root.app = app
    except AttributeError:
        pass
    return app, disp


def print_help() -> None:
    """Print the action list + usage."""
    from .framework import theme

    print(
        "lekiwi-tui — control center for the LeKiwi (lerobot) workflow.\n"
        "\n"
        "  lekiwi                                    # interactive menu\n"
        "  lekiwi teleop                             # jump straight to an action\n"
        "  python -m lekiwi_tui                      # Python module entry point\n"
        "  python -m lekiwi_tui host-kill            # stop the running Pi host\n"
        "\n"
        "Actions:"
    )
    for a in ACTIONS:
        print(f"  {theme.action_icon(a.icon)}  {a.id:<12} {a.hint}")
    print(
        "\nAliases: host-launch=l/launch  host-kill=k/kill  teleop=t  record=r  "
        "replay=p  view=v/viz  calibrate=c  eval=e/rollout  robot-config=y/robot  "
        "settings=s/config  setup-pi=pi/provision  sync=rsync/push\n"
        "\n--dry-run / -n : PREVIEW mode — print each command instead of running it.\n"
        "Default is REAL execution (run from your lerobot env so the scripts can import\n"
        "lerobot). Toggle live with 'd' in the menu.\n"
        "\nConfig precedence: env var at launch > lekiwi.yaml > built-in default."
        f"\nWorkspace root: {ROOT}"
    )


def _action_suggestions(name: str) -> list[str]:
    """Return close action/alias suggestions for an unknown CLI action."""
    ranked: list[tuple[float, str]] = []

    for action_id in ACTIONS_BY_ID:
        parts = action_id.split("-")
        score = max(
            difflib.SequenceMatcher(None, name, action_id).ratio(),
            *(difflib.SequenceMatcher(None, name, part).ratio() for part in parts),
        )
        if score >= 0.55:
            ranked.append((score, action_id))

    for alias, action_id in ALIASES.items():
        score = difflib.SequenceMatcher(None, name, alias).ratio()
        if len(alias) > 1 and score >= 0.7:
            ranked.append((score, f"{alias} ({action_id})"))

    out: list[str] = []
    seen: set[str] = set()
    for _, suggestion in sorted(ranked, key=lambda item: (-item[0], len(item[1]), item[1])):
        if suggestion in seen:
            continue
        seen.add(suggestion)
        out.append(suggestion)
        if len(out) == 3:
            break
    return out


def _workspace_ready(*, require_private_config: bool) -> bool:
    """Validate the project workspace before any action that needs config/scripts."""
    scripts = ROOT / "scripts"
    missing: list[str] = []
    if not ROOT.is_dir():
        missing.append(str(ROOT))
    if require_private_config and not CFG_FILE.exists():
        missing.append(str(CFG_FILE))
    if not scripts.is_dir():
        missing.append(str(scripts))
    if not require_private_config and not (CFG_FILE.exists() or EXAMPLE_CFG_FILE.exists()):
        missing.append(str(EXAMPLE_CFG_FILE))
    if not missing:
        return True
    print("✗ lekiwi workspace is incomplete.", file=sys.stderr)
    print(f"  resolved root: {ROOT}", file=sys.stderr)
    for path in missing:
        print(f"  missing: {path}", file=sys.stderr)
    print(
        "  Install from the checkout with `python -m pip install -e .`, or set "
        "LEKIWI_ROOT=/path/to/lekiwi-tui.",
        file=sys.stderr,
    )
    if require_private_config and not CFG_FILE.exists() and EXAMPLE_CFG_FILE.exists():
        print(
            "  First-run setup: `cp lekiwi.example.yaml lekiwi.yaml`, then edit "
            "lekiwi.yaml for your robot.",
            file=sys.stderr,
        )
    return False


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # Global --dry-run / -n opt-in: PREVIEW the argv each action would run instead of
    # executing it. Default is real execution (like the Textual app).
    if "--dry-run" in args or "-n" in args:
        from .framework import runner
        runner.DRY_RUN = True
        args = [a for a in args if a not in ("--dry-run", "-n")]
    is_tty = sys.stdin.isatty() and sys.stdout.isatty()

    # ── no args → the interactive menu (needs a TTY) ──────────────────────────
    if not args:
        if not _workspace_ready(require_private_config=True):
            return 1
        os.chdir(ROOT)  # relative dataset/model paths resolve against the workspace root
        if not is_tty:
            print(
                "The menu needs an interactive terminal. Use: python -m lekiwi_tui <action>",
                file=sys.stderr,
            )
            return 1
        ctx = load_context(is_tty=True)
        app, _ = _make_app(ctx, MenuScreen(None, ctx))
        asyncio.run(app.run())
        return 0

    action_name, *extra = args
    if action_name in ("-h", "--help", "help"):
        print_help()
        return 0

    action = resolve_action(action_name)
    if action is None:
        print(f"✗ unknown action: '{action_name}'", file=sys.stderr)
        matches = _action_suggestions(action_name)
        if matches:
            print(f"  did you mean: {', '.join(matches)}", file=sys.stderr)
        print(
            "  valid: host-launch host-kill teleop record replay view calibrate "
            "train eval setup-pi sync robot-config settings",
            file=sys.stderr,
        )
        return 1

    from .framework import runner

    if not _workspace_ready(require_private_config=not runner.DRY_RUN):
        return 1
    os.chdir(ROOT)  # relative dataset/model paths resolve against the workspace root

    ctx = load_context(is_tty=is_tty)

    # ── suspend action: run it without the TUI loop (suspend_run falls back to a
    #    plain blocking subprocess when there is no live terminal). ─────────────
    if action.kind == "suspend":
        disp = Dispatcher(ctx)
        # A non-running App so the handler has app.notify / app.terminal (None → fallback).
        disp.bind(App(MenuScreen(None, ctx), run_action=disp.run_action))
        return asyncio.run(disp.run_action(action.id, extra))

    # ── screen / stream / host action ──
    module_path, _, cls_name = action.handler.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        print(f"✗ {action.label} not ported yet.", file=sys.stderr)
        return 0
    # No TTY → use the screen's run_headless(ctx, extra) hook if it has one (the original's
    # no-TTY dispatch: settings prints KEY=value, train/eval/record/teleop front the script,
    # robot-config dumps the core values); else it genuinely needs a terminal.
    if not is_tty:
        hook = getattr(module, "run_headless", None)
        if hook is not None:
            return int(hook(ctx, extra))
        print(
            f"✗ {action.label} needs an interactive terminal. Run `python -m lekiwi_tui`.",
            file=sys.stderr,
        )
        return 1
    screen_cls = getattr(module, cls_name)
    app, _ = _make_app(ctx, screen_cls(None, ctx))
    asyncio.run(app.run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
