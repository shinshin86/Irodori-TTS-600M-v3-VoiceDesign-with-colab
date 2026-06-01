import gradio as gr
from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    default_runtime_device,
    get_cached_runtime,
    list_available_runtime_precisions,
    save_wav,
)
from datetime import datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

CHECKPOINT = "Aratako/Irodori-TTS-600M-v3-VoiceDesign"
_runtime_cache = None


def _get_runtime():
    global _runtime_cache
    if _runtime_cache is not None:
        return _runtime_cache
    device = default_runtime_device()
    precision = list_available_runtime_precisions(device)[0]
    resolved = hf_hub_download(repo_id=CHECKPOINT, filename="model.safetensors")
    key = RuntimeKey(
        checkpoint=resolved,
        model_device=device,
        codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
        model_precision=precision,
        codec_device=device,
        codec_precision=precision,
        compile_model=False,
        compile_dynamic=False,
    )
    runtime, _ = get_cached_runtime(key)
    _runtime_cache = runtime
    return runtime


def generate(text, caption, ref_wav):
    if not text.strip():
        raise gr.Error("テキストを入力してください")
    runtime = _get_runtime()

    has_ref = bool(ref_wav)
    result = runtime.synthesize(
        SamplingRequest(
            text=text.strip(),
            caption=caption.strip() or None,
            # 参照音声があれば声をクローン（3要素制御）。なければキャプションのみ。
            ref_wav=ref_wav if has_ref else None,
            ref_latent=None,
            no_ref=not has_ref,
            ref_normalize_db=-16.0,
            ref_ensure_max=True,
            num_candidates=1,
            decode_mode="sequential",
            # v3 は Duration Predictor が長さを自動推定するため seconds は指定しない
            seconds=None,
            duration_scale=1.0,
            max_ref_seconds=30.0,
            max_text_len=None,
            max_caption_len=None,
            num_steps=40,
            seed=None,
            cfg_guidance_mode="independent",
            cfg_scale_text=3.0,
            cfg_scale_caption=3.0,
            # 参照音声があるときだけ話者ガイダンスを有効化
            cfg_scale_speaker=5.0 if has_ref else 0.0,
            cfg_scale=None,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=None,
            rescale_k=None,
            rescale_sigma=None,
            context_kv_cache=True,
            speaker_kv_scale=None,
            speaker_kv_min_t=None,
            speaker_kv_max_layers=None,
            trim_tail=True,
        ),
        log_fn=lambda msg: print(msg, flush=True),
    )
    out_dir = Path("gradio_outputs")
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = save_wav(
        out_dir / f"sample_{stamp}.wav",
        result.audios[0].float(),
        result.sample_rate,
    )
    return str(out_path)


with gr.Blocks(title="Irodori-TTS v3 VoiceDesign") as demo:
    gr.Markdown("# Irodori-TTS v3 VoiceDesign")
    gr.Markdown(
        "テキストとキャプションで声質を指定して日本語音声を生成します。\n\n"
        "参照音声をアップロードすると、その声をクローンしつつ"
        "キャプションで話し方を制御できます（3要素制御）。"
    )
    text = gr.Textbox(
        label="Text",
        lines=3,
        placeholder="ここに読み上げたいテキストを入力...",
    )
    caption = gr.Textbox(
        label="Caption / Style Prompt (optional)",
        lines=2,
        placeholder="例: 落ち着いた女性の声で、やわらかく自然に読み上げてください。",
    )
    ref_wav = gr.Audio(
        label="Reference Audio (optional) — 声をクローンしたい音声をアップロード",
        type="filepath",
        sources=["upload", "microphone"],
    )
    btn = gr.Button("Generate", variant="primary")
    audio = gr.Audio(label="Generated Audio", type="filepath", interactive=False)
    btn.click(generate, inputs=[text, caption, ref_wav], outputs=[audio])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=7861)
