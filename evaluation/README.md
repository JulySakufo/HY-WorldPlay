# Evaluation Utilities

This directory mirrors the Light Interaction fixed-prompt evaluation setup for
HY-WorldPlay experiments.

## Included

```text
data/
  refined_prompts_llava16.json
  sampled_200/

scripts/
  batch_video_generation.py      # Batch generation, timing, optional speedup summary
  evaluate_psnr_ssim_lpips.py    # PSNR / SSIM / LPIPS metrics, copied from Light Interaction
```

## Batch Generation And Timing

Run from the HY-WorldPlay root:

```bash
export HY_MODEL_PATH=/path/to/HunyuanVideo-1.5
export HY_AR_DISTILL_ACTION_MODEL_PATH=/path/to/ar_distilled_action_model/diffusion_pytorch_model.safetensors

python evaluation/scripts/batch_video_generation.py \
  --output-root outputs/eval_sdtm \
  --allowed-gpus 0 \
  --method-preset sdtm
```

The default action groups follow Light Interaction:

```text
left_right=left-5, right-5.5
forward_backward=w-5, s-5.5
```

Those durations are interpreted as seconds and converted to HY-WorldPlay latent
pose strings using `seconds * fps / 4`. With the default `253` frames and
`24` fps, `left-5, right-5.5` becomes `left-30, right-33`. If you change
`--num-frames`, keep the action duration aligned with that frame count or pass
a latent-count pose with `--action-unit latents`.

Per-video timings are written next to each generated video:

```text
outputs/eval_sdtm/left_right/<video_id>/generation_time.json
```

Aggregate timing files are written to:

```text
outputs/eval_sdtm/timing/generation_times.csv
outputs/eval_sdtm/timing/generation_time_summary.csv
```

To run a baseline without SDTM token merging:

```bash
python evaluation/scripts/batch_video_generation.py \
  --output-root outputs/eval_baseline \
  --allowed-gpus 0 \
  --method-preset baseline
```

To compute speedup after a baseline run:

```bash
python evaluation/scripts/batch_video_generation.py \
  --output-root outputs/eval_sdtm \
  --allowed-gpus 0 \
  --method-preset sdtm \
  --baseline-timing-csv outputs/eval_baseline/timing/generation_times.csv
```

This writes:

```text
outputs/eval_sdtm/timing/speedup_vs_baseline.csv
outputs/eval_sdtm/timing/speedup_summary.csv
```

Extra `hyvideo/generate.py` arguments can be appended to the command. Later
arguments override the method preset, for example:

```bash
python evaluation/scripts/batch_video_generation.py \
  --output-root outputs/eval_sdtm_ratio_03 \
  --allowed-gpus 0 \
  --method-preset sdtm \
  --sdtm_ratio 0.3 --sdtm_verbose true
```

## PSNR / SSIM / LPIPS

Mutual comparison against a baseline:

```bash
python evaluation/scripts/evaluate_psnr_ssim_lpips.py \
  --run-mutual \
  --ref-dir outputs/eval_baseline/left_right \
  --test-dir outputs/eval_sdtm/left_right \
  --output-dir evaluation_results \
  --tag sdtm_left_right \
  --mutual-window 30
```

Self-consistency on a return trajectory:

```bash
python evaluation/scripts/evaluate_psnr_ssim_lpips.py \
  --run-self \
  --test-dir outputs/eval_sdtm/forward_backward \
  --output-dir evaluation_results \
  --tag sdtm_forward_backward \
  --one-way-sec 5 \
  --self-window 50
```
