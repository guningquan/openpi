"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>

Cameras listed in CAMERAS but missing from the data (e.g. cam_low) are left out of the dataset.

Faster conversion:
  --num-workers N   converts the episodes in N parallel processes (the default 1 converts one episode at a time on a
                    single core), then merges them into repo-id and checks the result against the raw hdf5 files.
  --base-repo-id D  reuses the already converted LeRobot dataset D as the first episodes of repo-id: its parquet files
                    are hard-linked (instant, no extra disk space, D is left unchanged) and only raw-dir is converted.
                    D must have the same features (cameras, velocity/effort). Implies the parallel path.

Example: add 30 new episodes to an already converted 50-episode dataset, converting with 6 processes:
uv run examples/aloha_real/convert_aloha_data_to_lerobot_no_bgr2rgb.py \
  --raw-dir /mnt/ssd2/guningquan/Dataset/packaging_simple_2_plus \
  --base-repo-id /mnt/ssd2/guningquan/Dataset/packaging_simple_2_lerobot \
  --repo-id /mnt/ssd2/guningquan/Dataset/packaging_simple_2_plus_lerobot \
  --task "packaging the items" --num-workers 6
"""

import dataclasses
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import tqdm
import tyro
import matplotlib.pyplot as plt

CAMERAS = [
    "cam_high",
    # "gel",
    "cam_low",
    "cam_left_wrist",
    "cam_right_wrist",
]


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    cameras: list[str] | None = None,
) -> LeRobotDataset:
    motors = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]
    if cameras is None:
        cameras = CAMERAS

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                # imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))  # commented by gnq
                imgs_array.append(cv2.imdecode(data, 1))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [cam for cam in CAMERAS if f"/observations/images/{cam}" in ep],
        )

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]
                # frame[f"observation.images.{camera}"] = img_array[i][..., ::-1]  # BGR to RGB @gnq

                # print(f"camera: {camera}, img_array[i].shape: {img_array[i].shape}")
                # plt.figure(figsize=(10, 6))
                # plt.imshow(frame[f"observation.images.{camera}"])
                # plt.axis('off')
                # plt.title(camera)
                # plt.show(block=False)
                # plt.waitforbuttonpress()  # Wait for any key or mouse click
                # plt.close()

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
            
            frame['task'] = task

            dataset.add_frame(frame)

        # dataset.save_episode(task=task)
        dataset.save_episode()  # gnq


    return dataset


# ----------------------------------------------------------------------------------------------------------------------
# Parallel conversion (--num-workers / --base-repo-id). Also used by convert_aloha_data_to_lerobot_no_bgr2rgb_multi.py,
# which passes its own create_empty_dataset / populate_dataset so each script converts exactly as its sequential path.

SHARD_IMAGE_WRITER_PROCESSES = 4  # per parallel process; several processes run at once


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def convert_shard_from_spec(spec_path: str, create_empty_dataset_fn, populate_dataset_fn) -> None:
    """Child process: convert the (hdf5 file, task) pairs listed in the spec, in order, into one LeRobot dataset."""
    spec = json.load(open(spec_path))
    dataset = create_empty_dataset_fn(
        spec["out"],
        robot_type=spec["robot_type"],
        mode=spec["mode"],
        has_velocity=spec["has_velocity"],
        has_effort=spec["has_effort"],
        dataset_config=DatasetConfig(**spec["dataset_config"]),
        cameras=spec["cameras"],
    )
    for task, group in itertools.groupby(spec["files_tasks"], key=lambda file_task: file_task[1]):
        populate_dataset_fn(dataset, [Path(f) for f, _ in group], task=task)
    dataset.stop_image_writer()
    print("SHARD DONE", flush=True)


def _column_stats(values: np.ndarray) -> dict:
    return {
        "min": [int(values.min())],
        "max": [int(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(len(values))],
    }


def _merge_shards(out: Path, base: Path | None, shard_dirs: list[Path]) -> None:
    """Write base (hard-linked) + shards (episode_index / index / task_index renumbered) into out."""
    (out / "data" / "chunk-000").mkdir(parents=True)
    (out / "meta").mkdir()
    if base is not None:
        info = json.load(open(base / "meta" / "info.json"))
        tasks = [t["task"] for t in sorted(_read_jsonl(base / "meta" / "tasks.jsonl"), key=lambda t: t["task_index"])]
        episodes = _read_jsonl(base / "meta" / "episodes.jsonl")
        ep_stats = _read_jsonl(base / "meta" / "episodes_stats.jsonl")
        assert info["total_episodes"] == len(episodes) == len(ep_stats) and info["total_chunks"] == 1
        for i in range(info["total_episodes"]):
            name = f"episode_{i:06d}.parquet"
            os.link(base / "data" / "chunk-000" / name, out / "data" / "chunk-000" / name)
    else:
        info = json.load(open(shard_dirs[0] / "meta" / "info.json"))
        info["total_episodes"] = info["total_frames"] = 0
        tasks, episodes, ep_stats = [], [], []

    next_ep, next_index = info["total_episodes"], info["total_frames"]
    for shard in shard_dirs:
        s_info = json.load(open(shard / "meta" / "info.json"))
        if s_info["features"] != info["features"]:
            raise ValueError(f"features of {shard} differ from {base or shard_dirs[0]} (cameras / velocity / effort?)")
        task_map = {}
        for t in _read_jsonl(shard / "meta" / "tasks.jsonl"):
            if t["task"] not in tasks:
                tasks.append(t["task"])
            task_map[t["task_index"]] = tasks.index(t["task"])
        s_stats = {r["episode_index"]: r for r in _read_jsonl(shard / "meta" / "episodes_stats.jsonl")}
        for e in _read_jsonl(shard / "meta" / "episodes.jsonl"):
            pf = pq.ParquetFile(shard / "data" / "chunk-000" / f"episode_{e['episode_index']:06d}.parquet")
            table = pf.read()
            n = table.num_rows
            new_cols = {
                "episode_index": np.full(n, next_ep, dtype=np.int64),
                "index": np.arange(next_index, next_index + n, dtype=np.int64),
                "task_index": np.array([task_map[t] for t in table.column("task_index").to_pylist()], dtype=np.int64),
            }
            for name, values in new_cols.items():
                i = table.schema.get_field_index(name)
                table = table.set_column(i, table.schema.field(name), pa.array(values))
            pq.write_table(
                table.replace_schema_metadata(pf.schema_arrow.metadata),
                out / "data" / "chunk-000" / f"episode_{next_ep:06d}.parquet",
                compression="snappy",
                row_group_size=pf.metadata.row_group(0).num_rows,
            )
            episodes.append({**e, "episode_index": next_ep})
            stats = {**s_stats[e["episode_index"]]["stats"], **{k: _column_stats(v) for k, v in new_cols.items()}}
            ep_stats.append({**s_stats[e["episode_index"]], "episode_index": next_ep, "stats": stats})
            next_ep += 1
            next_index += n

    if next_ep > info["chunks_size"]:
        raise ValueError(f"{next_ep} episodes do not fit in one chunk of {info['chunks_size']}")
    info.update(total_episodes=next_ep, total_frames=next_index, total_tasks=len(tasks), splits={"train": f"0:{next_ep}"})
    json.dump(info, open(out / "meta" / "info.json", "w"), indent=4)
    _write_jsonl(out / "meta" / "tasks.jsonl", [{"task_index": i, "task": t} for i, t in enumerate(tasks)])
    _write_jsonl(out / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(out / "meta" / "episodes_stats.jsonl", ep_stats)


def _verify(out: Path, base_episodes: int, files_tasks: list[tuple[Path, str]]) -> None:
    episodes = _read_jsonl(out / "meta" / "episodes.jsonl")
    task_index = {t["task"]: t["task_index"] for t in _read_jsonl(out / "meta" / "tasks.jsonl")}
    raw_lens = []
    for path, _ in files_tasks:
        with h5py.File(path, "r") as ep:
            raw_lens.append(ep["/action"].shape[0])
    if [e["length"] for e in episodes[base_episodes:]] != raw_lens:
        raise ValueError("converted episode lengths do not match the raw hdf5 files")
    if [e["episode_index"] for e in episodes] != list(range(len(episodes))):
        raise ValueError("episode_index in episodes.jsonl is not 0..N-1")
    expect = 0
    for i, e in enumerate(episodes):
        t = pq.read_table(out / "data" / "chunk-000" / f"episode_{i:06d}.parquet",
                          columns=["index", "episode_index", "frame_index", "task_index"])
        n = e["length"]
        ok = (
            (t.column("index").to_numpy() == np.arange(expect, expect + n)).all()
            and (t.column("episode_index").to_numpy() == i).all()
            and (t.column("frame_index").to_numpy() == np.arange(n)).all()
        )
        if i >= base_episodes:
            ok = ok and (t.column("task_index").to_numpy() == task_index[files_tasks[i - base_episodes][1]]).all()
        if not ok:
            raise ValueError(f"index / episode_index / frame_index / task_index wrong in episode {i}")
        expect += n


def run_parallel(
    script: str,
    repo_id: str,
    files_tasks: list[tuple[Path, str]],
    *,
    num_workers: int,
    base_repo_id: str | None,
    cameras: list[str],
    has_velocity: bool,
    has_effort: bool,
    robot_type: str,
    mode: str,
    dataset_config: DatasetConfig,
) -> None:
    """Convert files_tasks in num_workers processes (each runs `script _shard <spec>`), then merge into repo_id."""
    out = Path(LEROBOT_HOME / repo_id)
    base = Path(LEROBOT_HOME / base_repo_id) if base_repo_id else None
    base_episodes = json.load(open(base / "meta" / "info.json"))["total_episodes"] if base else 0
    if out.exists():
        shutil.rmtree(out)
    tmp = out.parent / f".{out.name}.tmp_shards"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    num_workers = max(1, min(num_workers, len(files_tasks)))
    bounds = np.linspace(0, len(files_tasks), num_workers + 1).astype(int)
    config = {**dataclasses.asdict(dataset_config), "image_writer_processes": SHARD_IMAGE_WRITER_PROCESSES}
    print(f"Converting {len(files_tasks)} episodes in {num_workers} processes"
          + (f", reusing {base_episodes} episodes of {base}" if base else ""), flush=True)
    procs, shard_dirs, logs = [], [], []
    for k in range(num_workers):
        spec = {
            "out": str(tmp / f"shard_{k}"),
            "files_tasks": [(str(f), t) for f, t in files_tasks[bounds[k] : bounds[k + 1]]],
            "cameras": cameras, "has_velocity": has_velocity, "has_effort": has_effort,
            "robot_type": robot_type, "mode": mode, "dataset_config": config,
        }
        spec_path = tmp / f"shard_{k}.json"
        json.dump(spec, open(spec_path, "w"))
        logs.append(tmp / f"shard_{k}.log")
        procs.append(subprocess.Popen([sys.executable, script, "_shard", str(spec_path)],
                                      stdout=open(logs[-1], "w"), stderr=subprocess.STDOUT))
        shard_dirs.append(tmp / f"shard_{k}")

    start, reported = time.time(), -1
    while any(p.poll() is None for p in procs):
        done = sum(len(_read_jsonl(d / "meta" / "episodes.jsonl")) for d in shard_dirs if (d / "meta" / "episodes.jsonl").exists())
        if done != reported:
            print(f"[{(time.time() - start) / 60:5.1f} min] converted {done}/{len(files_tasks)}", flush=True)
            reported = done
        time.sleep(15)
    failed = [k for k, p in enumerate(procs) if p.returncode != 0 or "SHARD DONE" not in logs[k].read_text()]
    if failed:
        raise RuntimeError(f"conversion process(es) {failed} failed; see {tmp}/shard_<k>.log (temporary files kept)")

    print(f"[{(time.time() - start) / 60:5.1f} min] all processes done, merging into {out} ...", flush=True)
    _merge_shards(out, base, shard_dirs)
    _verify(out, base_episodes, files_tasks)
    shutil.rmtree(tmp)
    info = json.load(open(out / "meta" / "info.json"))
    print(f"DONE in {(time.time() - start) / 60:.1f} min: {out} has {info['total_episodes']} episodes, "
          f"{info['total_frames']} frames; checked against the raw hdf5 files", flush=True)


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    num_workers: int = 1,
    base_repo_id: str | None = None,
):
    """num_workers > 1: convert in that many parallel processes. base_repo_id: reuse that converted dataset as the
    first episodes (see the module docstring)."""
    if base_repo_id is not None and Path(LEROBOT_HOME / base_repo_id).resolve() == Path(LEROBOT_HOME / repo_id).resolve():
        raise ValueError("base_repo_id must differ from repo_id (repo_id is deleted and rewritten)")

    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        download_raw(raw_dir, repo_id=raw_repo_id)

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))
    cameras = [cam for cam in CAMERAS if cam in get_cameras(hdf5_files)]

    if num_workers > 1 or base_repo_id is not None:
        if episodes is not None:
            hdf5_files = [hdf5_files[i] for i in episodes]
        run_parallel(
            __file__,
            repo_id,
            [(f, task) for f in hdf5_files],
            num_workers=num_workers,
            base_repo_id=base_repo_id,
            cameras=cameras,
            has_velocity=has_velocity(hdf5_files),
            has_effort=has_effort(hdf5_files),
            robot_type="mobile_aloha" if is_mobile else "aloha",
            mode=mode,
            dataset_config=dataset_config,
        )
        return

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
        cameras=cameras,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    # dataset.consolidate()

    # if push_to_hub:
    #     dataset.push_to_hub()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "_shard":  # child process started by run_parallel
        convert_shard_from_spec(sys.argv[2], create_empty_dataset, populate_dataset)
    else:
        tyro.cli(port_aloha)
