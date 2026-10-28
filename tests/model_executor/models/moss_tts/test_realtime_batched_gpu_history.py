# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weight-free CPU/CUDA regression of the production batch/state methods.

Load the actual method ASTs to keep these tests runnable without vLLM/CUDA
extensions. Tensor math and routing execute unchanged with real PyTorch.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
SOURCES = ROOT / "vllm_omni/model_executor/models/moss_tts"


class OmniOutput(NamedTuple):
    text_hidden_states: object
    multimodal_outputs: dict


def load_functions(filename, names, class_name=None):
    tree = ast.parse((SOURCES / filename).read_text())
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    funcs = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(funcs) == len(names)
    mod = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *funcs],
        type_ignores=[],
    )
    ns = dict(torch=torch, F=F, OmniOutput=OmniOutput)
    exec(compile(ast.fix_missing_locations(mod), filename, "exec"), ns)
    return ns


LOCAL = load_functions("modeling_moss_tts_local.py", {"_apply_batched_repetition_penalty", "_sample_token"})
PENALTY = LOCAL["_apply_batched_repetition_penalty"]
MAKE = load_functions("modeling_moss_tts_talker.py", {"make_omni_output"}, "MossTTSRealtimeTalkerForGeneration")[
    "make_omni_output"
]
GENERATE_NS = load_functions("modeling_moss_tts_local.py", {"generate_frame"}, "MossTTSRealtimeLocalTransformer")
GENERATE_NS.update(LOCAL)
GENERATE = GENERATE_NS["generate_frame"]

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_row_local_penalty_padding_duplicates_and_sign(device):
    logits = torch.tensor([[2.0, -3.0, 4.0, -5.0], [2.0, -3.0, 4.0, -5.0]], device=device)
    history = torch.tensor([[0, 0, -1, -1], [1, 3, 1, -1]], device=device)
    actual = PENALTY(logits, history, 2.0)
    expected = torch.tensor([[1.0, -3.0, 4.0, -5.0], [2.0, -6.0, 4.0, -10.0]], device=device)
    torch.testing.assert_close(actual, expected)
    assert logits[0, 0] == 2  # no in-place contamination


@pytest.mark.parametrize("device", DEVICES)
def test_real_generate_frame_batches_like_independent_greedy_rows(device):
    torch.manual_seed(7)
    nq, hidden_size, vocab = 4, 8, 12
    calls = []

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.codec_embedding = torch.nn.ModuleList([torch.nn.Embedding(vocab, hidden_size) for _ in range(nq - 1)])

        def forward(self, embeds, positions):
            calls.append(embeds.shape[0])
            # Causal, batch-independent toy transformer for control-flow testing.
            return embeds.cumsum(dim=1)

    model = SimpleNamespace(config=SimpleNamespace(rvq=nq, hidden_size=hidden_size), model=Decoder().to(device))
    heads = torch.nn.ModuleList([torch.nn.Linear(hidden_size, vocab, bias=False) for _ in range(nq)]).to(device)
    h = torch.randn(3, hidden_size, device=device)
    hist = torch.full((3, nq, 5), -1, dtype=torch.long, device=device)
    hist[0, :, 0] = 1
    hist[1, :, :2] = 2
    hist[2, :, :3] = 3
    batch = GENERATE(model, h, heads, do_sample=False, repetition_penalty=1.1, history_codes=hist)
    assert calls == [3] * nq
    singles = torch.cat(
        [
            GENERATE(model, h[i : i + 1], heads, do_sample=False, repetition_penalty=1.1, history_codes=hist[i : i + 1])
            for i in range(3)
        ]
    )
    torch.testing.assert_close(batch, singles)
    with pytest.raises(ValueError, match="shape"):
        GENERATE(model, h, heads, history_codes=hist[:1])


def bare(device="cpu"):
    calls = []

    def frame(hidden, heads, **kwargs):
        calls.append((hidden.clone(), kwargs["history_codes"].clone()))
        return hidden[:, :1].long().expand(-1, 4).clone()

    obj = SimpleNamespace(
        n_vq=4,
        local_transformer=SimpleNamespace(generate_frame=frame),
        local_lm_heads=[],
        AUDIO_EOS=1026,
        AUDIO_BOS=1025,
        audio_pad_token=1024,
    )
    return obj, calls


def run(obj, hidden, infos, spans=None, eligible=None):
    if spans is None:
        spans = [(i, i + 1) for i in range(len(infos))]
    return MAKE(
        obj, hidden, runtime_additional_information=infos, request_token_spans=spans, request_sample_eligible=eligible
    )


@pytest.mark.parametrize("device", DEVICES)
def test_compaction_mixed_prefill_eos_and_output_placeholders(device):
    obj, calls = bare(device)
    infos = [{}, {}, {"audio_state": {"is_stopping": True}}, {}, {}]
    out = run(
        obj,
        torch.tensor([[1.0], [10.0], [1026.0], [99.0], [98.0], [20.0]], device=device),
        infos,
        spans=[(0, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        eligible=[True, True, True, False, True],
    )
    assert len(calls) == 1
    assert calls[0][0].flatten().tolist() == [10, 1026, 20]
    audio = out.multimodal_outputs["codes"]["audio"]
    assert len(audio) == 5
    assert audio[0].tolist() == [[10] * 4]
    assert audio[4].tolist() == [[20] * 4]
    assert all(audio[i].numel() == 0 for i in (1, 2, 3))
    assert infos[1]["audio_state"]["is_stopping"]
    assert "audio_codes" not in infos[3]


@pytest.mark.parametrize("device", DEVICES)
def test_history_follows_request_on_reorder_cancel_and_new_request(device):
    obj, calls = bare(device)
    a, b, c = {}, {}, {}
    run(obj, torch.tensor([[10.0], [20.0]], device=device), [a, b])
    old_a = a["audio_codes"]["history"]
    run(obj, torch.tensor([[21.0], [11.0]], device=device), [b, a])
    histories = calls[-1][1]
    assert histories[0, :, 0].tolist() == [20] * 4
    assert histories[1, :, 0].tolist() == [10] * 4
    assert old_a[:, 1].tolist() == [-1] * 4  # previous snapshot immutable
    # b is cancelled/removed; a continues; c must have entirely fresh state.
    run(obj, torch.tensor([[12.0], [30.0]], device=device), [a, c])
    assert calls[-1][1][0, :, :2].tolist() == [[10, 11]] * 4
    assert torch.all(calls[-1][1][1] == -1)
    assert c["audio_codes"]["history"].device.type == device
    assert a["audio_codes"]["history"].data_ptr() != c["audio_codes"]["history"].data_ptr()


@pytest.mark.parametrize("device", DEVICES)
def test_history_ring_wrap_and_legacy_bootstrap(device):
    obj, _ = bare(device)
    a, b = {}, {}
    for step in range(65):
        run(obj, torch.tensor([[float(step)], [float(step + 100)]], device=device), [a, b])
    assert sorted(a["audio_codes"]["history"][0].tolist()) == list(range(15, 65))
    assert sorted(b["audio_codes"]["history"][0].tolist()) == list(range(115, 165))
    legacy = {"audio_codes": {"accumulated": torch.arange(60, device=device).view(60, 1).expand(-1, 4)}}
    run(obj, torch.tensor([[60.0]], device=device), [legacy])
    assert sorted(legacy["audio_codes"]["history"][0].tolist()) == list(range(11, 61))


def test_no_active_rows_and_generation_limit():
    obj, calls = bare()
    a = {"audio_state": {"max_new_frames": 2}}
    for _ in range(3):
        out = run(obj, torch.tensor([[1.0]]), [a])
    assert len(calls) == 2
    assert a["audio_state"]["step"] == 2
    assert out.multimodal_outputs == {}
    assert run(obj, torch.ones(1, 1), [{}], eligible=[False]).multimodal_outputs == {}


def test_history_is_declared_gpu_resident():
    source = (SOURCES / "modeling_moss_tts_talker.py").read_text()
    realtime = source.split("class MossTTSRealtimeTalkerForGeneration", 1)[1].split("class ", 1)[0]
    assert '("audio_codes", "history")' in realtime


@pytest.mark.parametrize("device", DEVICES)
def test_sixteen_streams_reordered_and_independently_penalized(device):
    obj, calls = bare(device)
    infos = [{} for _ in range(16)]
    values = torch.arange(1, 17, device=device).float().view(-1, 1)
    run(obj, values, infos)
    assert calls[0][0].shape[0] == 16
    order = [15, 0, 7, 4, 3, 12, 11, 9, 2, 8, 1, 6, 13, 14, 10, 5]
    out = run(obj, values[order] + 100, [infos[i] for i in order])
    assert len(calls) == 2
    assert calls[-1][1][:, 0, 0].tolist() == [i + 1 for i in order]
    audio = out.multimodal_outputs["codes"]["audio"]
    assert [x[0, 0].item() for x in audio] == [i + 101 for i in order]
    logits = torch.ones(16, 200, device=device)
    hist = torch.stack([infos[i]["audio_codes"]["history"] for i in order])
    penalized = PENALTY(logits, hist[:, 0], 2.0)
    expected = logits.clone()
    for row, i in enumerate(order):
        expected[row, i + 1] = 0.5
        expected[row, i + 101] = 0.5
    torch.testing.assert_close(penalized, expected)
