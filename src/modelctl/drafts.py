"""Speculative-decoding draft companions for llama.cpp GGUF repositories.

A draft model ships next to the quantizations in the same repository and is
recognised by its basename prefix or a parent directory, for example
``mtp-gemma-4-31B-it.gguf``, ``MTP/mtp-...-Q8_0.gguf``, or
``dspark-Inkling-2-Mini-Q8_0.gguf``. Drafts are companions, never
quantization groups. The companion key stays ``mtp`` for compatibility with
existing manifests and object metadata.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from .errors import ManifestError

# Longest prefix first so ``dflash2-`` is never classified as ``dflash``.
_DRAFT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("dflash2-", "dflash2"),
    ("dflash-", "dflash"),
    ("dspark-", "dspark"),
    ("mtp-", "mtp"),
)
_DRAFT_DIRS = frozenset({"mtp", "dspark", "dflash", "dflash2"})
# llama.cpp --spec-type per draft kind. DFlash has no confirmed value, so it
# is served with --model-draft only unless the manifest records one.
_DEFAULT_SPEC_TYPES = {"mtp": "draft-mtp", "dspark": "draft-dspark"}
_SPEC_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_Q8_0_RE = re.compile(r"(?:^|[-_.])q8_0(?:[-_.]|$)")


def draft_kind(value: str) -> str | None:
    """Return the draft kind of a repository path, or None for non-drafts."""
    path = PurePosixPath(value)
    if path.suffix.lower() != ".gguf":
        return None
    basename = path.name.lower()
    for prefix, kind in _DRAFT_PREFIXES:
        if basename.startswith(prefix):
            return kind
    for part in reversed(path.parts[:-1]):
        if part.lower() in _DRAFT_DIRS:
            return part.lower()
    return None


def is_draft_artifact(value: str) -> bool:
    return draft_kind(value) is not None


def is_imatrix_artifact(value: str) -> bool:
    """Importance-matrix calibration data that some repos ship as ``*.gguf``."""
    return "imatrix" in PurePosixPath(value).name.lower()


def preferred_draft(root_candidates: list[str]) -> str | None:
    """Default among several root-level drafts: the single Q8_0 one."""
    q8 = [
        value
        for value in root_candidates
        if _Q8_0_RE.search(PurePosixPath(value).stem.lower())
    ]
    return q8[0] if len(q8) == 1 else None


def default_spec_type(draft_path: str) -> str | None:
    """The kind default for the ``mtp`` companion's --spec-type.

    Before draft kinds existed, every ``mtp`` companion served with
    ``draft-mtp``. That stays true for any path under an ``mtp/`` directory
    and for paths with no recognised kind (hand-written manifests). Only a
    DSpark draft (``draft-dspark``) or a DFlash draft (no value) outside an
    ``mtp/`` directory gets a different default.
    """
    parents = {part.lower() for part in PurePosixPath(draft_path).parts[:-1]}
    if "mtp" in parents:
        return "draft-mtp"
    kind = draft_kind(draft_path)
    if kind is None:
        return "draft-mtp"
    return _DEFAULT_SPEC_TYPES.get(kind)


def is_valid_spec_type(value: str) -> bool:
    return bool(_SPEC_TYPE_RE.fullmatch(value))


def validate_spec_type(value: str) -> str:
    if not _SPEC_TYPE_RE.fullmatch(value):
        raise ManifestError(
            f"invalid draft spec type {value!r}; use lowercase letters, digits, "
            "and '-' (at most 32 characters)"
        )
    return value


def effective_spec_type(draft_path: str, recorded: object) -> str | None:
    """The --spec-type for a draft: the recorded value, else the kind default.

    Legacy objects carry no recorded value, so an ``mtp-`` draft keeps
    serving with ``draft-mtp`` exactly as before.
    """
    if recorded is None:
        return default_spec_type(draft_path)
    if not isinstance(recorded, str) or not is_valid_spec_type(recorded):
        raise ManifestError("object metadata has an invalid draft_spec_type")
    return recorded


def draft_args(draft: str, resolved: str, recorded: object) -> list[str]:
    """llama.cpp argv for a draft companion.

    ``draft`` is the companion's repository path (it decides the kind, so
    blob-backed cache snapshots classify the same way); ``resolved`` is the
    on-disk file passed to --model-draft.
    """
    spec_type = effective_spec_type(draft, recorded)
    args = ["--model-draft", resolved]
    if spec_type is not None:
        args.extend(["--spec-type", spec_type])
    return args
