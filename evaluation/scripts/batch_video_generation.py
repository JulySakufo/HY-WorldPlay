#!/usr/bin/env python3
# Copyright 2026 Jiacheng Lu and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Batch generation and timing on the fixed Light Interaction evaluation set.

The sample set uses second-style action groups by default. This wrapper converts
those groups to HY-WorldPlay latent-count pose strings before calling
``hyvideo/generate.py``.
"""

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


DEFAULT_ACTIONS = {
    "left_right": "left-5, right-5.5",
    "forward_backward": "w-5, s-5.5",
}

METHOD_PRESET_ARGS = {
    "baseline": [
        "--use_sdtm",
        "false",
    ],
    "sdtm": [
        "--use_sdtm",
        "true",
        "--sdtm_ratio",
        "0.2",
        "--sdtm_deviation",
        "0.05",
        "--sdtm_sx",
        "4",
        "--sdtm_sy",
        "5",
        "--sdtm_switch_step",
        "1",
        "--sdtm_protect_steps_frequency",
        "-1",
        "--sdtm_auto_window",
        "true",
        "--sdtm_verbose",
        "false",
    ],
}

TIMING_COLUMNS = [
    "action",
    "video_id",
    "filename",
    "status",
    "returncode",
    "duration_wall_s",
    "gpu_id",
    "method_preset",
    "pose",
    "pose_source",
    "seed",
    "num_frames",
    "num_steps",
    "output_dir",
    "video_path",
    "started_at",
    "ended_at",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    repo_root = Path(__file__).resolve().parents[2]
    default_model_path = os.environ.get("HY_MODEL_PATH")
    default_action_ckpt = (
        os.environ.get("HY_AR_DISTILL_ACTION_MODEL_PATH")
        or os.environ.get("AR_DISTILL_ACTION_MODEL_PATH")
    )

    parser.add_argument("--prompt-json", default="evaluation/data/refined_prompts_llava16.json")
    parser.add_argument("--hy-worldplay-root", default=str(repo_root))
    parser.add_argument("--model-path", default=default_model_path)
    parser.add_argument("--action-ckpt", default=default_action_ckpt)
    parser.add_argument("--output-root", required=True, help="Directory where generated videos are written.")
    parser.add_argument("--torchrun", default="torchrun")
    parser.add_argument("--pythonpath", default=None, help="Optional PYTHONPATH prefix.")
    parser.add_argument("--actions", nargs="*", default=[f"{k}={v}" for k, v in DEFAULT_ACTIONS.items()])
    parser.add_argument("--action-unit", choices=["seconds", "latents"], default="seconds")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--temporal-compression", type=int, default=4)
    parser.add_argument("--allowed-gpus", default="", help="Comma-separated GPU ids. Empty means all visible GPUs.")
    parser.add_argument("--vram-threshold-mb", type=int, default=20)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--poll-interval-sec", type=int, default=10)
    parser.add_argument("--master-port-base", type=int, default=29540)
    parser.add_argument("--num-frames", type=int, default=253)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--method-preset", choices=sorted(METHOD_PRESET_ARGS), default="sdtm")
    parser.add_argument("--model-type", choices=["ar", "bi"], default="ar")
    parser.add_argument("--few-step", choices=["true", "false"], default="true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument(
        "--baseline-timing-csv",
        default=None,
        help="Optional generation_times.csv from a baseline run. Writes speedup_vs_baseline.csv.",
    )

    args, extra_generate_args = parser.parse_known_args()
    if extra_generate_args and extra_generate_args[0] == "--":
        extra_generate_args = extra_generate_args[1:]
    args.extra_generate_args = extra_generate_args

    if not args.model_path:
        parser.error("--model-path or HY_MODEL_PATH is required.")
    if not args.action_ckpt:
        parser.error("--action-ckpt or HY_AR_DISTILL_ACTION_MODEL_PATH is required.")
    return args


def parse_actions(items):
    actions = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Action must use name=pose format: {item}")
        name, pose = item.split("=", 1)
        name = name.strip()
        pose = pose.strip()
        if not name or not pose:
            raise ValueError(f"Action must use name=pose format: {item}")
        actions[name] = pose
    return actions


def parse_pose_commands(pose):
    commands = []
    for raw in pose.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "-" not in raw:
            raise ValueError(f"Invalid pose command: {raw}")
        action, value = raw.rsplit("-", 1)
        commands.append((action.strip(), float(value.strip())))
    if not commands:
        raise ValueError(f"Empty pose string: {pose}")
    return commands


def video_motion_latents(num_frames, temporal_compression):
    if (num_frames - 1) % temporal_compression != 0:
        raise ValueError(
            f"--num-frames must satisfy (num_frames - 1) % {temporal_compression} == 0."
        )
    return (num_frames - 1) // temporal_compression


def distribute_seconds_to_latents(commands, fps, temporal_compression):
    if any(value < 0 for _, value in commands):
        raise ValueError("Action durations must be non-negative.")
    total_seconds = sum(value for _, value in commands)
    if total_seconds <= 0:
        raise ValueError("Total action duration must be positive.")

    raw_counts = [value * fps / temporal_compression for _, value in commands]
    target_motion_latents = int(round(sum(raw_counts)))
    counts = [math.floor(value) for value in raw_counts]
    remaining = target_motion_latents - sum(counts)
    order = sorted(
        range(len(raw_counts)),
        key=lambda idx: raw_counts[idx] - counts[idx],
        reverse=True,
    )
    for idx in order[:remaining]:
        counts[idx] += 1
    return counts


def convert_pose_for_hy_worldplay(pose, action_unit, num_frames, fps, temporal_compression):
    if pose.endswith(".json"):
        return pose

    commands = parse_pose_commands(pose)
    target_motion_latents = video_motion_latents(num_frames, temporal_compression)
    if action_unit == "seconds":
        counts = distribute_seconds_to_latents(commands, fps, temporal_compression)
        actual = sum(counts)
        if actual != target_motion_latents:
            implied_frames = actual * temporal_compression + 1
            raise ValueError(
                f"Pose {pose!r} implies {actual} motion latents "
                f"({implied_frames} frames at {fps} fps), but --num-frames "
                f"{num_frames} requires {target_motion_latents}. Adjust --num-frames "
                "or pass a latent-count pose with --action-unit latents."
            )
    else:
        for _, value in commands:
            if not value.is_integer():
                raise ValueError(f"Latent-count poses must use integer durations: {pose!r}")
        counts = [int(value) for _, value in commands]
        actual = sum(counts)
        if actual != target_motion_latents:
            raise ValueError(
                f"Pose {pose!r} uses {actual} motion latents, but --num-frames "
                f"{num_frames} requires {target_motion_latents}."
            )

    return ", ".join(f"{action}-{count}" for (action, _), count in zip(commands, counts))


def resolve_path(root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(root, path)


def safe_stem(filename):
    return Path(filename).stem.replace(" ", "_")


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def free_gpus(threshold_mb, allowed):
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True).strip()
    available = []
    for line in output.splitlines():
        if not line.strip():
            continue
        gpu_id, used = [int(x.strip()) for x in line.split(",")]
        if allowed and gpu_id not in allowed:
            continue
        if used < threshold_mb:
            available.append(gpu_id)
    return available


def has_generated_video(out_dir):
    out_dir = Path(out_dir)
    return (out_dir / "gen.mp4").exists() or (out_dir / "gen_sr.mp4").exists()


def build_command(args, task, image_path, out_dir, pose):
    cmd = [
        args.torchrun,
        "--nproc_per_node=1",
        f"--master_port={args.master_port}",
        "hyvideo/generate.py",
        "--prompt",
        task["refined_prompt"],
        "--image_path",
        str(image_path),
        "--resolution",
        "480p",
        "--aspect_ratio",
        "16:9",
        "--video_length",
        str(args.num_frames),
        "--seed",
        str(args.seed),
        "--rewrite",
        "false",
        "--sr",
        "false",
        "--save_pre_sr_video",
        "--output_path",
        str(out_dir),
        "--model_path",
        args.model_path,
        "--action_ckpt",
        args.action_ckpt,
        "--few_step",
        args.few_step,
        "--pose",
        pose,
        "--num_inference_steps",
        str(args.num_steps),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--model_type",
        args.model_type,
        "--use_vae_parallel",
        "false",
        "--use_sageattn",
        "false",
        "--use_fp8_gemm",
        "false",
        "--transformer_resident_ar_rollout",
        "true",
    ]
    cmd.extend(METHOD_PRESET_ARGS[args.method_preset])
    cmd.extend(args.extra_generate_args)
    return cmd


def write_json_atomic(path, payload):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def run_one(gpu_id, task, args, action_name, pose_source, pose):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    pythonpath_entries = []
    if args.pythonpath:
        pythonpath_entries.append(args.pythonpath)
    pythonpath_entries.append(str(args.hy_worldplay_root))
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    prompt_path = Path(args.prompt_json).resolve()
    image_path = Path(prompt_path.parent.parent, task["image_path"]).resolve()
    out_dir = Path(args.output_root, action_name, safe_stem(task["filename"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    args.master_port = args.master_port_base + gpu_id
    cmd = build_command(args, task, image_path, out_dir, pose)

    stdout_path = out_dir / "generate_stdout.log"
    stderr_path = out_dir / "generate_stderr.log"
    timing_path = out_dir / "generation_time.json"
    video_path = out_dir / "gen.mp4"

    started_at = iso_now()
    start = time.perf_counter()
    returncode = -1
    error = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_f:
            result = subprocess.run(
                cmd,
                cwd=args.hy_worldplay_root,
                env=env,
                text=True,
                stdout=stdout_f,
                stderr=stderr_f,
                check=False,
            )
        returncode = result.returncode
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    duration = time.perf_counter() - start
    ended_at = iso_now()

    status = "ok" if returncode == 0 and video_path.exists() else "failed"
    record = {
        "action": action_name,
        "video_id": safe_stem(task["filename"]),
        "filename": task["filename"],
        "status": status,
        "returncode": returncode,
        "duration_wall_s": duration,
        "gpu_id": gpu_id,
        "method_preset": args.method_preset,
        "pose": pose,
        "pose_source": pose_source,
        "seed": args.seed,
        "num_frames": args.num_frames,
        "num_steps": args.num_steps,
        "output_dir": str(out_dir),
        "video_path": str(video_path),
        "started_at": started_at,
        "ended_at": ended_at,
        "command": cmd,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    if error:
        record["error"] = error
    write_json_atomic(timing_path, record)

    if status == "ok":
        logging.info("[GPU %s] done %s/%s in %.1fs", gpu_id, action_name, task["filename"], duration)
    else:
        logging.error("[GPU %s] failed %s/%s in %.1fs", gpu_id, action_name, task["filename"], duration)


def run_action(action_name, pose_source, tasks, args, allowed):
    pose = convert_pose_for_hy_worldplay(
        pose_source,
        args.action_unit,
        args.num_frames,
        args.fps,
        args.temporal_compression,
    )

    pending = []
    for task in tasks:
        out_dir = Path(args.output_root, action_name, safe_stem(task["filename"]))
        if args.skip_existing and has_generated_video(out_dir):
            continue
        pending.append(task)

    print(f"[{action_name}] pose={pose!r} pending {len(pending)} / {len(tasks)}")
    active = {}
    while pending or active:
        finished = [gpu for gpu, proc in active.items() if not proc.is_alive()]
        for gpu in finished:
            active[gpu].join()
            del active[gpu]

        idle = [gpu for gpu in free_gpus(args.vram_threshold_mb, allowed) if gpu not in active]
        while idle and pending and len(active) < args.max_concurrent:
            gpu_id = idle.pop(0)
            task = pending.pop(0)
            proc = mp.Process(target=run_one, args=(gpu_id, task, args, action_name, pose_source, pose))
            proc.start()
            active[gpu_id] = proc
            print(f"[dispatch] gpu={gpu_id} action={action_name} file={task['filename']}")
            if idle and pending:
                time.sleep(4)

        if pending:
            time.sleep(args.poll_interval_sec)


def load_timing_records(output_root):
    rows = []
    for path in sorted(Path(output_root).glob("*/*/generation_time.json")):
        try:
            with path.open(encoding="utf-8") as f:
                record = json.load(f)
        except json.JSONDecodeError:
            continue
        record["duration_wall_s"] = float(record.get("duration_wall_s", 0.0))
        rows.append(record)
    return rows


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def collect_timings(output_root):
    records = load_timing_records(output_root)
    timing_dir = Path(output_root, "timing")
    timing_dir.mkdir(parents=True, exist_ok=True)
    write_csv(timing_dir / "generation_times.csv", records, TIMING_COLUMNS)

    summary_rows = []
    actions = sorted({row.get("action") for row in records if row.get("action")})
    for action in actions + (["ALL"] if actions else []):
        group = records if action == "ALL" else [row for row in records if row.get("action") == action]
        ok = [row for row in group if row.get("status") == "ok"]
        failed = [row for row in group if row.get("status") != "ok"]
        durations = [row["duration_wall_s"] for row in ok]
        summary_rows.append(
            {
                "action": action,
                "count_ok": len(ok),
                "count_failed": len(failed),
                "mean_duration_wall_s": mean(durations) if durations else "",
                "median_duration_wall_s": median(durations) if durations else "",
                "total_duration_wall_s": sum(durations) if durations else "",
            }
        )
    write_csv(
        timing_dir / "generation_time_summary.csv",
        summary_rows,
        [
            "action",
            "count_ok",
            "count_failed",
            "mean_duration_wall_s",
            "median_duration_wall_s",
            "total_duration_wall_s",
        ],
    )
    return records


def load_timing_csv(path):
    rows = {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            key = (row.get("action"), row.get("video_id"))
            rows[key] = row
    return rows


def write_speedup_csv(current_records, baseline_timing_csv, output_root):
    baseline = load_timing_csv(baseline_timing_csv)
    rows = []
    for record in current_records:
        if record.get("status") != "ok":
            continue
        key = (record.get("action"), record.get("video_id"))
        base = baseline.get(key)
        if not base:
            continue
        baseline_s = float(base["duration_wall_s"])
        test_s = float(record["duration_wall_s"])
        rows.append(
            {
                "action": record.get("action"),
                "video_id": record.get("video_id"),
                "baseline_duration_wall_s": baseline_s,
                "test_duration_wall_s": test_s,
                "speedup": baseline_s / test_s if test_s > 0 else "",
            }
        )

    timing_dir = Path(output_root, "timing")
    write_csv(
        timing_dir / "speedup_vs_baseline.csv",
        rows,
        [
            "action",
            "video_id",
            "baseline_duration_wall_s",
            "test_duration_wall_s",
            "speedup",
        ],
    )

    if rows:
        baseline_mean = mean(row["baseline_duration_wall_s"] for row in rows)
        test_mean = mean(row["test_duration_wall_s"] for row in rows)
        summary = [
            {
                "matched_videos": len(rows),
                "mean_baseline_duration_wall_s": baseline_mean,
                "mean_test_duration_wall_s": test_mean,
                "speedup_of_means": baseline_mean / test_mean if test_mean > 0 else "",
                "mean_per_video_speedup": mean(row["speedup"] for row in rows if row["speedup"]),
            }
        ]
    else:
        summary = []
    write_csv(
        timing_dir / "speedup_summary.csv",
        summary,
        [
            "matched_videos",
            "mean_baseline_duration_wall_s",
            "mean_test_duration_wall_s",
            "speedup_of_means",
            "mean_per_video_speedup",
        ],
    )


def main():
    args = parse_args()
    args.hy_worldplay_root = resolve_path(Path.cwd(), args.hy_worldplay_root).resolve()
    args.prompt_json = resolve_path(args.hy_worldplay_root, args.prompt_json).resolve()
    args.output_root = resolve_path(args.hy_worldplay_root, args.output_root).resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=args.output_root / "batch_generation.log",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    allowed = {int(x) for x in args.allowed_gpus.split(",") if x.strip()} if args.allowed_gpus else set()
    actions = parse_actions(args.actions)
    with Path(args.prompt_json).open(encoding="utf-8") as f:
        tasks = json.load(f)
    tasks = tasks[args.start_index :]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    mp.set_start_method("spawn", force=True)
    for action_name, pose in actions.items():
        run_action(action_name, pose, tasks, args, allowed)
        records = collect_timings(args.output_root)
        if args.baseline_timing_csv:
            write_speedup_csv(records, args.baseline_timing_csv, args.output_root)

    records = collect_timings(args.output_root)
    if args.baseline_timing_csv:
        write_speedup_csv(records, args.baseline_timing_csv, args.output_root)
    print(f"[timing] wrote {Path(args.output_root, 'timing', 'generation_times.csv')}")
    print(f"[timing] wrote {Path(args.output_root, 'timing', 'generation_time_summary.csv')}")


if __name__ == "__main__":
    main()
