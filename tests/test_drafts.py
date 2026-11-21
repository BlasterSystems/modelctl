"""Speculative draft companions: kinds, selection, spec types, and serving.

File lists are trimmed copies of real Hugging Face repositories (2026-11-21):
unsloth/Inkling-2-Mini-GGUF, unsloth/gemma-4-26B-A4B-it-GGUF and
bartowski/google_gemma-4-26B-A4B-it-GGUF.
"""

import shlex
from pathlib import Path

import pytest
import yaml

from modelctl.download_queue import DownloadQueueError, load_download_queue
from modelctl.drafts import (
    default_spec_type,
    draft_args,
    draft_kind,
    effective_spec_type,
    preferred_draft,
)
from modelctl.errors import ManifestError
from modelctl.generation import generate_manifest_document
from modelctl.layout import Layout, atomic_symlink
from modelctl.manifest import parse_manifest
from modelctl.operations import serve_command
from modelctl.validation import ExpectedFile, validate_object, write_metadata

from test_generation import FakeApi

INKLING = [
    "README.md",
    "Inkling-2-Mini-UD-IQ1_M.gguf",
    "Inkling-2-Mini-UD-Q4_K_M.gguf",
    "Inkling-2-Mini-UD-Q4_K_XL.gguf",
    "Inkling-2-Mini-UD-Q8_K_XL.gguf",
    "Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file",
    "Mainline_Support/README.md",
    "dspark-Inkling-2-Mini-Q8_0.gguf",
    "dspark/README.md",
    "dspark/dspark-Inkling-2-Mini-BF16.gguf",
    "mmproj-BF16.gguf",
    "mmproj-F16.gguf",
    "mmproj-F32.gguf",
]
UNSLOTH_GEMMA = [
    "BF16/gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf",
    "BF16/gemma-4-26B-A4B-it-BF16-00002-of-00002.gguf",
    "MTP/mtp-gemma-4-26B-A4B-it-BF16.gguf",
    "MTP/mtp-gemma-4-26B-A4B-it-F16.gguf",
    "MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf",
    "gemma-4-26B-A4B-it-Q8_0.gguf",
    "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
    "imatrix_unsloth.gguf_file",
    "mmproj-BF16.gguf",
    "mmproj-F16.gguf",
    "mmproj-F32.gguf",
    "mtp-gemma-4-26B-A4B-it.gguf",
]
BARTOWSKI_GEMMA = [
    "google_gemma-4-26B-A4B-it-Q4_K_M.gguf",
    "google_gemma-4-26B-A4B-it-Q8_0.gguf",
    "google_gemma-4-26B-A4B-it-bf16/google_gemma-4-26B-A4B-it-bf16-00001-of-00002.gguf",
    "google_gemma-4-26B-A4B-it-bf16/google_gemma-4-26B-A4B-it-bf16-00002-of-00002.gguf",
    "google_gemma-4-26B-A4B-it-imatrix.gguf",
    "mmproj-google_gemma-4-26B-A4B-it-bf16.gguf",
    "mmproj-google_gemma-4-26B-A4B-it-f16.gguf",
    "mtp-google_gemma-4-26B-A4B-it-Q4_0.gguf",
    "mtp-google_gemma-4-26B-A4B-it-Q8_0.gguf",
]


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("mtp-gemma.gguf", "mtp"),
        ("MTP/gemma-4-12B-it-MTP-Q8_0.gguf", "mtp"),
        ("dspark-Inkling-2-Mini-Q8_0.gguf", "dspark"),
        ("dspark/dspark-Inkling-2-Mini-BF16.gguf", "dspark"),
        ("DSPARK-x.gguf", "dspark"),
        ("dflash-Muse-Glimmer-30B-Q4_K_M.gguf", "dflash"),
        ("dflash2-model-Q8_0.gguf", "dflash2"),
        ("DFlash2/model-Q8_0.gguf", "dflash2"),
        ("gemma-4-Q4_K_M.gguf", None),
        ("dspark/README.md", None),
        ("dsparkle-model.gguf", None),
        ("BF16/model-BF16-00001-of-00002.gguf", None),
    ],
)
def test_draft_kind_classifies_prefixes_and_directories(path, kind):
    assert draft_kind(path) == kind


def test_spec_type_defaults_by_kind():
    assert default_spec_type("mtp-x.gguf") == "draft-mtp"
    assert default_spec_type("MTP/x.gguf") == "draft-mtp"
    assert default_spec_type("dspark-x.gguf") == "draft-dspark"
    assert default_spec_type("dspark/dspark-x-BF16.gguf") == "draft-dspark"
    assert default_spec_type("dflash-x.gguf") is None
    assert default_spec_type("dflash2-x.gguf") is None
    # Legacy: unknown names and anything under MTP/ keep draft-mtp.
    assert default_spec_type("draft.gguf") == "draft-mtp"
    assert default_spec_type("MTP/dflash-x.gguf") == "draft-mtp"


def test_effective_spec_type_rejects_unsafe_recorded_values():
    assert effective_spec_type("dflash-x.gguf", "draft-dflash") == "draft-dflash"
    with pytest.raises(ManifestError):
        effective_spec_type("mtp-x.gguf", "--host=0.0.0.0")
    with pytest.raises(ManifestError):
        effective_spec_type("mtp-x.gguf", 3)


def test_draft_args_omit_spec_type_when_unknown():
    assert draft_args("dflash-x.gguf", "/m/dflash-x.gguf", None) == [
        "--model-draft",
        "/m/dflash-x.gguf",
    ]


def test_preferred_draft_is_the_single_q8_0():
    assert (
        preferred_draft(["mtp-m-Q4_0.gguf", "mtp-m-Q8_0.gguf"]) == "mtp-m-Q8_0.gguf"
    )
    assert preferred_draft(["mtp-m-Q4_0.gguf", "mtp-m-F16.gguf"]) is None
    assert preferred_draft(["a-Q8_0.gguf", "b-q8_0.gguf"]) is None


def test_inkling_selects_root_dspark_draft_and_f16_mmproj():
    document = generate_manifest_document(
        "unsloth/Inkling-2-Mini-GGUF",
        quantization="UD-Q4_K_XL",
        api=FakeApi(INKLING),
    )
    assert document["entrypoint"] == "Inkling-2-Mini-UD-Q4_K_XL.gguf"
    assert document["companions"] == {
        "mmproj": "mmproj-F16.gguf",
        "mtp": "dspark-Inkling-2-Mini-Q8_0.gguf",
    }
    # The kind default is derived at serve time, not recorded.
    assert "draft_spec_type" not in document


def test_inkling_dspark_files_are_not_quantization_groups():
    with pytest.raises(ManifestError, match="no GGUF quantization matches 'Q8_0'"):
        generate_manifest_document(
            "unsloth/Inkling-2-Mini-GGUF",
            quantization="Q8_0",
            api=FakeApi(INKLING),
        )


def test_inkling_explicit_subdirectory_draft():
    document = generate_manifest_document(
        "unsloth/Inkling-2-Mini-GGUF",
        quantization="UD-Q4_K_XL",
        mtp="dspark/dspark-Inkling-2-Mini-BF16.gguf",
        api=FakeApi(INKLING),
    )
    assert document["companions"]["mtp"] == "dspark/dspark-Inkling-2-Mini-BF16.gguf"


def test_unsloth_gemma_keeps_root_mtp_draft():
    document = generate_manifest_document(
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        quantization="UD-Q4_K_XL",
        api=FakeApi(UNSLOTH_GEMMA),
    )
    assert document["companions"] == {
        "mmproj": "mmproj-F16.gguf",
        "mtp": "mtp-gemma-4-26B-A4B-it.gguf",
    }


def test_bartowski_gemma_prefers_q8_0_draft_and_skips_imatrix():
    document = generate_manifest_document(
        "bartowski/google_gemma-4-26B-A4B-it-GGUF",
        quantization="Q4_K_M",
        api=FakeApi(BARTOWSKI_GEMMA),
    )
    assert document["companions"] == {
        "mmproj": "mmproj-google_gemma-4-26B-A4B-it-f16.gguf",
        "mtp": "mtp-google_gemma-4-26B-A4B-it-Q8_0.gguf",
    }
    with pytest.raises(ManifestError, match="no GGUF quantization matches 'imatrix'"):
        generate_manifest_document(
            "bartowski/google_gemma-4-26B-A4B-it-GGUF",
            quantization="imatrix",
            api=FakeApi(BARTOWSKI_GEMMA),
        )


def test_several_root_drafts_without_q8_0_still_require_mtp():
    files = ["m-Q4_K_M.gguf", "mtp-m-Q4_0.gguf", "mtp-m-F16.gguf"]
    with pytest.raises(ManifestError, match="pass --mtp"):
        generate_manifest_document("org/m-GGUF", api=FakeApi(files))


def test_draft_spec_type_is_recorded_only_as_an_override():
    files = ["m-Q4_K_M.gguf", "dflash-m-Q8_0.gguf"]
    document = generate_manifest_document(
        "org/m-GGUF", draft_spec_type="draft-dflash", api=FakeApi(files)
    )
    assert document["draft_spec_type"] == "draft-dflash"
    same_as_default = generate_manifest_document(
        "org/m-GGUF",
        quantization="UD-Q4_K_XL",
        draft_spec_type="draft-dspark",
        api=FakeApi(INKLING),
    )
    assert "draft_spec_type" not in same_as_default


@pytest.mark.parametrize("value", ["Draft-MTP", "a b", "x;rm", "-x", "a" * 33])
def test_draft_spec_type_rejects_invalid_values(value):
    with pytest.raises(ManifestError, match="invalid draft spec type"):
        generate_manifest_document(
            "org/m-GGUF",
            draft_spec_type=value,
            api=FakeApi(["m-Q4_K_M.gguf", "mtp-m.gguf"]),
        )


def test_draft_spec_type_requires_a_draft_companion():
    with pytest.raises(ManifestError, match="requires a draft model companion"):
        generate_manifest_document(
            "org/m-GGUF", draft_spec_type="draft-mtp", api=FakeApi(["m-Q4_K_M.gguf"])
        )


def test_manifest_rejects_draft_spec_type_without_mtp_companion():
    with pytest.raises(ManifestError, match="requires an 'mtp' draft companion"):
        parse_manifest(
            {"repo": "org/m", "format": "gguf", "draft_spec_type": "draft-mtp"}, "m"
        )


def test_draft_spec_type_changes_object_identity_only_when_set(tmp_path):
    layout = Layout(tmp_path)
    base = {
        "repo": "org/m",
        "format": "gguf",
        "entrypoint": "m.gguf",
        "companions": {"mtp": "dflash-m.gguf"},
    }
    plain = layout.object_path(parse_manifest(base, "m"), "a" * 40)
    override = layout.object_path(
        parse_manifest({**base, "draft_spec_type": "draft-dflash"}, "m"), "a" * 40
    )
    assert plain != override


def _published(tmp_path: Path, companions: dict, **extra) -> Path:
    layout = Layout(tmp_path)
    layout.prepare()
    manifest = parse_manifest(
        {
            "repo": "org/m",
            "format": "gguf",
            "entrypoint": "m.gguf",
            "runtime": "llama.cpp",
            "companions": companions,
            **extra,
        },
        "m",
    )
    object_path = layout.object_path(manifest, "e" * 40)
    names = ["m.gguf", *companions.values()]
    for name in names:
        (object_path / name).parent.mkdir(parents=True, exist_ok=True)
        (object_path / name).write_bytes(b"gguf")
    expected = [ExpectedFile(name, 4) for name in names]
    write_metadata(object_path, manifest, "e" * 40, expected, "m.gguf")
    validate_object(object_path)
    atomic_symlink(object_path, layout.active_path("m"))
    return object_path


def test_serve_uses_draft_dspark_for_dspark_companion(tmp_path):
    object_path = _published(tmp_path, {"mtp": "dspark-m-Q8_0.gguf"})
    argv = shlex.split(serve_command(tmp_path, "m"))
    assert argv[-4:] == [
        "--model-draft",
        str((object_path / "dspark-m-Q8_0.gguf").resolve()),
        "--spec-type",
        "draft-dspark",
    ]


def test_serve_omits_spec_type_for_dflash_without_override(tmp_path):
    object_path = _published(tmp_path, {"mtp": "dflash-m-Q8_0.gguf"})
    argv = shlex.split(serve_command(tmp_path, "m"))
    assert argv[-2:] == ["--model-draft", str((object_path / "dflash-m-Q8_0.gguf").resolve())]
    assert "--spec-type" not in argv


def test_serve_uses_recorded_override(tmp_path):
    _published(tmp_path, {"mtp": "dflash-m-Q8_0.gguf"}, draft_spec_type="draft-dflash")
    argv = shlex.split(serve_command(tmp_path, "m"))
    assert argv[-2:] == ["--spec-type", "draft-dflash"]


def test_legacy_mtp_directory_object_still_serves_draft_mtp(tmp_path):
    _published(tmp_path, {"mtp": "MTP/gemma-MTP-Q8_0.gguf"})
    argv = shlex.split(serve_command(tmp_path, "m"))
    assert argv[-2:] == ["--spec-type", "draft-mtp"]


@pytest.mark.parametrize(
    "companion", ["draft.gguf", "gemma-draft-Q8_0.gguf", "MTP/dflash-x.gguf", "mtp/dspark-x.gguf"]
)
def test_legacy_mtp_companions_without_a_draft_kind_still_serve_draft_mtp(
    tmp_path, companion
):
    # Regression (review F1): before draft kinds, every mtp companion served
    # with draft-mtp, including hand-written names and anything under MTP/.
    _published(tmp_path, {"mtp": companion})
    argv = shlex.split(serve_command(tmp_path, "m"))
    assert argv[-2:] == ["--spec-type", "draft-mtp"]


def test_tampered_metadata_spec_type_is_rejected(tmp_path):
    import json

    object_path = _published(tmp_path, {"mtp": "mtp-m.gguf"})
    metadata_file = object_path / ".modelctl.json"
    payload = json.loads(metadata_file.read_text())
    payload["draft_spec_type"] = "x --host 0.0.0.0"
    metadata_file.write_text(json.dumps(payload))
    with pytest.raises(Exception, match="draft_spec_type"):
        serve_command(tmp_path, "m")


def test_queue_accepts_draft_spec_type(tmp_path):
    path = tmp_path / "queue.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "downloads": [
                    {
                        "source": "org/m-GGUF",
                        "quantization": "Q4_K_M",
                        "mtp": "dflash-m-Q8_0.gguf",
                        "draft_spec_type": "draft-dflash",
                    }
                ]
            }
        )
    )
    [request] = load_download_queue(path)
    assert request.draft_spec_type == "draft-dflash"


def test_queue_rejects_empty_draft_spec_type(tmp_path):
    path = tmp_path / "queue.yaml"
    path.write_text(
        yaml.safe_dump({"downloads": [{"source": "org/m", "draft_spec_type": ""}]})
    )
    with pytest.raises(DownloadQueueError, match="draft_spec_type"):
        load_download_queue(path)
