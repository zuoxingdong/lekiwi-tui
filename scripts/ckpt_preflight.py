#!/usr/bin/env python
"""ckpt_preflight.py — can this checkpoint's config load on the INSTALLED lerobot?

eval.sh runs this once before it execs the rollout. The failure it exists for: a
checkpoint trained on a lerobot that has a config field yours does not (an unreleased
fix, a patch, a policy plugin) dies inside draccus with a ~60-line traceback ending in

    draccus.utils.DecodingError: The fields `attention_backend` are not valid for SmolVLAConfig

and it dies AFTER the TUI form is filled in, the GPU is chosen and the robot is about
to be driven. The information needed to say so is available before any of that: load
the config the same way ``rollout/configs.py`` will, and report what happened in one
readable paragraph instead.

Deliberately conservative about blocking. Exit 2 only when the config genuinely fails
to load; anything that merely means "cannot tell" (a hub repo id rather than a local
dir, no config.json, lerobot not importable from this interpreter) exits 0 quietly and
lets the real launch produce the real error. A preflight that blocks a working launch
is worse than no preflight.

The verdict is cached under ``.lekiwi-cache/preflight/`` keyed by the config's
mtime+size AND the lerobot version, because the import it needs costs ~5s and a
rollout is something you launch over and over. Delete the cache dir to force a
re-check; it invalidates itself when either side changes.

Usage:  ckpt_preflight.py <policy-path>        exit 0 = ok / unknown, 2 = will not load
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".lekiwi-cache" / "preflight"

OK = 0
INCOMPATIBLE = 2


def lerobot_facts() -> tuple[str, str] | None:
    """``(version, path)`` of the importable lerobot, or None when there is none."""
    try:
        import lerobot
    except Exception:
        return None
    return getattr(lerobot, "__version__", "?"), str(Path(lerobot.__file__).parent)


def load_config(policy_dir: str) -> None:
    """Load the checkpoint config exactly as the rollout will, or raise.

    ``import lerobot.policies`` first: the choice classes register on import, and
    without it draccus fails with "Couldn't find a choice class for 'smolvla'" even
    for a perfectly valid checkpoint.
    """
    import lerobot.policies  # noqa: F401  (registers the policy choice classes)
    from lerobot.configs.policies import PreTrainedConfig

    PreTrainedConfig.from_pretrained(policy_dir)


def cache_key(config_file: Path, version: str) -> Path:
    st = config_file.stat()
    stamp = f"{config_file.resolve()}|{st.st_mtime_ns}|{st.st_size}|{version}"
    return CACHE_DIR / hashlib.sha1(stamp.encode()).hexdigest()[:16]


def unknown_fields(message: str) -> list[str]:
    """The field names out of draccus's "The fields `a`, `b` are not valid" message."""
    if "are not valid for" not in message and "is not valid for" not in message:
        return []
    head = message.split(" are not valid for")[0].split(" is not valid for")[0]
    return [chunk.strip(" `") for chunk in head.replace("The fields", "").split(",") if chunk.strip(" `")]


def report(message: str, policy_type: str, version: str, where: str) -> str:
    fields = unknown_fields(message)
    if fields:
        body = (
            "✗ this checkpoint needs config fields the installed lerobot does not have:\n"
            f"    missing:      {', '.join(fields)}\n"
            f"    policy type:  {policy_type or '?'}\n"
            f"    lerobot:      {version}  ({where})\n\n"
            "  It was trained on a lerobot that has them (an unreleased fix, a local patch,\n"
            "  or a policy plugin). Install that version, or retrain against this one."
        )
    else:
        body = (
            "✗ this checkpoint's config does not load on the installed lerobot:\n"
            f"    {message.strip().splitlines()[-1]}\n"
            f"    policy type:  {policy_type or '?'}\n"
            f"    lerobot:      {version}  ({where})"
        )
    return body + "\n\n  Nothing was sent to the robot."


def diagnose(policy: str, facts=lerobot_facts, load=load_config) -> tuple[int, str]:
    """``(exit_code, text)``. Only INCOMPATIBLE means "do not launch"."""
    config_file = Path(policy).expanduser() / "config.json"
    if not config_file.is_file():
        return OK, ""  # a hub repo id, or a dir the launch itself will complain about
    got = facts()
    if got is None:
        return OK, ""  # cannot tell from here; let the real launch speak
    version, where = got

    try:
        marker = cache_key(config_file, version)
    except OSError:
        marker = None
    if marker is not None and marker.exists():
        return OK, ""

    try:
        policy_type = str(json.loads(config_file.read_text()).get("type", ""))
    except Exception:
        policy_type = ""

    try:
        load(str(Path(policy).expanduser()))
    except Exception as exc:  # draccus DecodingError, but anything counts as "will not load"
        return INCOMPATIBLE, report(str(exc), policy_type, version, where)

    if marker is not None:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"{policy_type} ok on lerobot {version}\n")
        except OSError:
            pass  # a read-only checkout still gets the check, just not the shortcut
    return OK, ""


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: ckpt_preflight.py <policy-path>", file=sys.stderr)
        return OK  # never block on being called wrong
    code, text = diagnose(argv[1])
    if text:
        print(text, file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
