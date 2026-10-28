# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weight-free local-checkpoint routing regression; not a full engine test."""

import sys
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from .test_realtime_batched_gpu_history import load_functions

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ADAPTER = load_functions(
    "../../../entrypoints/openai/tts_adapters/moss_tts.py",
    {"_detect_moss_variant", "_get_moss_ref_encoder"},
    "_MossTTSAdapterBase",
)
ADAPTER["cast"] = cast
DETECT = ADAPTER["_detect_moss_variant"]


@pytest.mark.parametrize(
    "hf_config",
    [
        SimpleNamespace(model_type="moss_tts_realtime", architectures=None),
        SimpleNamespace(architectures=["MossTTSRealtime"]),
    ],
    ids=["model-type", "architecture"],
)
@pytest.mark.parametrize("model_path", ["/models/main", "/models/local-checkpoint"])
def test_local_realtime_checkpoint_uses_codec_reference_encoder(monkeypatch, hf_config, model_path):
    # A Realtime codec has no processor.model_config. Passing the processor
    # down the ordinary TTS branch reproduces the deployment's AttributeError.
    codec = object()
    processor = object()
    cache = object()
    adapter = SimpleNamespace(
        engine_client=SimpleNamespace(model_config=SimpleNamespace(model=model_path, hf_config=hf_config)),
        ctx=SimpleNamespace(server=SimpleNamespace(api_server_shared_dir=None)),
        _speaker_cache=cache,
        _get_moss_realtime_components=lambda: (object(), processor, codec),
        _get_moss_processor=lambda: processor,
    )
    adapter._moss_variant = DETECT(adapter)

    factory = load_functions("reference_encoder.py", {"build_reference_encoder"})

    def record_encoder(actual_processor, **kwargs):
        return SimpleNamespace(processor=actual_processor, **kwargs)

    factory.update(MossRealtimeReferenceEncoder=record_encoder, MossReferenceEncoder=record_encoder)
    module_name = "vllm_omni.model_executor.models.moss_tts.reference_encoder"
    module = ModuleType(module_name)
    module.build_reference_encoder = factory["build_reference_encoder"]
    monkeypatch.setitem(sys.modules, module_name, module)

    encoder = ADAPTER["_get_moss_ref_encoder"](adapter)
    assert adapter._moss_variant == "realtime"
    assert encoder.processor is codec
    assert encoder.variant == "realtime"
    assert encoder.n_vq == 16
    assert encoder.sr_target == 24000
    assert encoder.speaker_cache is cache
    assert ADAPTER["_get_moss_ref_encoder"](adapter) is encoder


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("OpenMOSS-Team/MOSS-TTS-Realtime", "realtime"),
        ("OpenMOSS-Team/MOSS-TTS-Local-v1.5", "local"),
        ("OpenMOSS-Team/MOSS-TTSD-v1.0", "ttsd"),
        ("OpenMOSS-Team/MOSS-SoundEffect", "sound_effect"),
        ("OpenMOSS-Team/MOSS-VoiceGenerator", "voice_generator"),
        ("/models/main", "tts"),
    ],
)
@pytest.mark.parametrize("hf_config", [None, SimpleNamespace(model_type="moss_tts", architectures=["MossTTSDelay"])])
def test_existing_name_fallback_is_preserved(model_name, expected, hf_config):
    adapter = SimpleNamespace(
        engine_client=SimpleNamespace(model_config=SimpleNamespace(model=model_name, hf_config=hf_config))
    )
    assert DETECT(adapter) == expected


@pytest.mark.parametrize(
    "engine_client",
    [None, SimpleNamespace(), SimpleNamespace(model_config=None), SimpleNamespace(model_config=SimpleNamespace())],
)
def test_missing_model_metadata_keeps_default_variant(engine_client):
    assert DETECT(SimpleNamespace(engine_client=engine_client)) == "tts"
