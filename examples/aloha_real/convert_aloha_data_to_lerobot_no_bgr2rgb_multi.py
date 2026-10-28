"""
Script to convert multiple subfolders of Aloha hdf5 data to a single LeRobot dataset v2.0 format.

Each subfolder should contain:
- One or more episode_*.hdf5 files
- One .txt file containing the task description

Example usage: 
uv run examples/aloha_real/convert_aloha_data_to_lerobot_multi.py \
  --raw-dir /path/to/multiobject \
  --repo-id /path/to/multiobject_lerobot

Cameras listed in CAMERAS but missing from the data are left out of the dataset.

Faster conversion (same options as convert_aloha_data_to_lerobot_no_bgr2rgb.py, which implements them):
  --num-workers N   converts the episodes in N parallel processes, then merges them into repo-id and checks the
                    result against the raw hdf5 files (the default 1 converts one episode at a time).
  --base-repo-id D  reuses the already converted LeRobot dataset D as the first episodes of repo-id (hard-linked,
                    D is left unchanged); only raw-dir is converted. Tasks of D and raw-dir are merged.

uv run examples/aloha_real/convert_aloha_data_to_lerobot_no_bgr2rgb_multi.py \
  --raw-dir /path/to/multiobject \
  --repo-id /path/to/multiobject_lerobot \
  --num-workers 6
"""

import dataclasses
from pathlib import Path
import shutil
import sys
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import matplotlib.pyplot as plt

from convert_aloha_data_to_lerobot_no_bgr2rgb import convert_shard_from_spec
from convert_aloha_data_to_lerobot_no_bgr2rgb import run_parallel

CAMERAS = [
    "cam_high",
    # "cam_low",
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
    """Check if any hdf5 file has velocity data."""
    if not hdf5_files:
        return False
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    """Check if any hdf5 file has effort data."""
    if not hdf5_files:
        return False
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def read_task_from_txt(subfolder: Path) -> str:
    """Read task content from a txt file in the subfolder.
    
    Args:
        subfolder: Path to the subfolder containing the txt file
        
    Returns:
        Task string read from the txt file
        
    Raises:
        FileNotFoundError: If no txt file is found in the subfolder
    """
    txt_files = list(subfolder.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No txt file found in {subfolder}")
    
    # Use the first txt file found
    txt_file = txt_files[0]
    task = txt_file.read_text(encoding="utf-8").strip()
    return task


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
                # imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
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


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    num_workers: int = 1,
    base_repo_id: str | None = None,
):
    """Convert multiple subfolders of Aloha hdf5 data to a single LeRobot dataset.
    
    Each subfolder should contain:
    - One or more episode_*.hdf5 files
    - One .txt file containing the task description

    num_workers > 1: convert in that many parallel processes. base_repo_id: reuse that converted dataset as the
    first episodes (see the module docstring).
    """
    if base_repo_id is not None and Path(LEROBOT_HOME / base_repo_id).resolve() == Path(LEROBOT_HOME / repo_id).resolve():
        raise ValueError("base_repo_id must differ from repo_id (repo_id is deleted and rewritten)")

    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        # download_raw(raw_dir, repo_id=raw_repo_id)

    # Collect all subfolders and their data
    subfolders = [item for item in raw_dir.iterdir() if item.is_dir()]
    if not subfolders:
        raise ValueError(f"No subfolders found in {raw_dir}")

    # Collect all hdf5 files from all subfolders to determine dataset properties
    all_hdf5_files = []
    subfolder_data = []  # List of (subfolder, task, hdf5_files) tuples

    for subfolder in sorted(subfolders):
        hdf5_files = sorted(subfolder.glob("episode_*.hdf5"))
        if not hdf5_files:
            print(f"Warning: No hdf5 files found in {subfolder}, skipping...")
            continue

        try:
            task = read_task_from_txt(subfolder)
        except FileNotFoundError as e:
            print(f"Warning: {e}, skipping {subfolder}...")
            continue

        all_hdf5_files.extend(hdf5_files)
        subfolder_data.append((subfolder, task, hdf5_files))

    if not all_hdf5_files:
        raise ValueError(f"No hdf5 files found in any subfolder of {raw_dir}")

    cameras = [cam for cam in CAMERAS if cam in get_cameras(all_hdf5_files)]

    if num_workers > 1 or base_repo_id is not None:
        files_tasks = []
        for _, task, hdf5_files in subfolder_data:
            selected = hdf5_files if episodes is None else [hdf5_files[i] for i in episodes]
            files_tasks.extend((f, task) for f in selected)
        run_parallel(
            __file__,
            repo_id,
            files_tasks,
            num_workers=num_workers,
            base_repo_id=base_repo_id,
            cameras=cameras,
            has_velocity=has_velocity(all_hdf5_files),
            has_effort=has_effort(all_hdf5_files),
            robot_type="mobile_aloha" if is_mobile else "aloha",
            mode=mode,
            dataset_config=dataset_config,
        )
        return

    # Create dataset with properties determined from all hdf5 files
    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(all_hdf5_files),
        has_velocity=has_velocity(all_hdf5_files),
        dataset_config=dataset_config,
        cameras=cameras,
    )

    # Process each subfolder and add to the unified dataset
    for subfolder, task, hdf5_files in tqdm.tqdm(subfolder_data, desc="Processing subfolders"):
        print(f"Processing {subfolder.name} with task: {task}")
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
