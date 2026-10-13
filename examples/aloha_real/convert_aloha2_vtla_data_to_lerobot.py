"""
Convert raw ALOHA2-VTLA episodes (as recorded by
act-triple-plus-aloha2/aloha_scripts/record_episodes_aloha2_vtla.py) to a LeRobot dataset v2.0.

Unlike examples/aloha_real/convert_aloha_data_to_lerobot_no_bgr2rgb_multi.py, the raw data here
is NOT hdf5. Each episode is a directory:

  episode_N/
    metadata.json
    observation.state.joint_position/data.csv
    actions.joint_position/data.csv
    observation.image.third_view/video.mp4        -> cam_high (per metadata's video_sources)
    observation.image.left_wrist_view/video.mp4    -> cam_left_wrist
    observation.image.right_wrist_view/video.mp4   -> cam_right_wrist
    observation.image.<slot>_tactile/video.mp4     (tactile views, converted by default)
    observation.tactile.force/<sensor>/data.npy    (optional, not converted)

Camera slots are resolved from each episode's metadata.json "video_sources" entry, keeping
sources that start with "cam_". The third-person camera is whatever the recorder was run with
(cam_high by default). By default the tactile image slots ("<slot>_tactile") are also converted,
stored as observation.images.<sensor> (e.g. observation.images.left_gripper_outer); pass
--no-include-tactile to skip them. Tactile force arrays are always skipped. Video frames are
decoded with OpenCV and kept as BGR (no BGR2RGB conversion), matching
convert_aloha_data_to_lerobot_no_bgr2rgb_multi.py.

--arm selects which arm to keep: "both" (default), "left" or "right". For a single arm, the other
arm's wrist camera, joints (state and action) and tactile images are dropped; shared cameras
such as cam_high are kept. Joint columns are assigned to an arm by a "left"/"right" token in
their name; if the columns have no such token, the first half is taken as left and the second
half as right (ALOHA convention).

raw_dir layout: either raw_dir itself directly contains episode_* directories (single task), or
raw_dir contains subfolders that each contain episode_* directories (one task per subfolder,
e.g. one subfolder per invocation of the recorder with a different --task_name/--task). By
default the task label for each episode is read from its own metadata.json "task" field; pass
--task to override it with the same fixed label for every episode instead.

Example usage:
uv run examples/aloha_real/convert_aloha2_vtla_data_to_lerobot.py \
  --raw-dir /path/to/raw_episodes \
  --repo-id /path/to/vtla_lerobot \
  --task "board insertion" \
  --arm right
"""

import csv
import dataclasses
import json
from pathlib import Path
import shutil
from typing import Literal

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro

FPS = 30

Arm = Literal["both", "left", "right"]


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
    motors: list[str],
    cameras: list[str],
    image_shapes: dict[str, tuple[int, int]],
    mode: Literal["video", "image"] = "image",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
    }

    for cam in cameras:
        height, width = image_shapes[cam]
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }

    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def find_episode_dirs(folder: Path) -> list[Path]:
    """Find episode_* directories directly inside `folder` that have a metadata.json."""
    episode_dirs = [
        p for p in folder.iterdir() if p.is_dir() and p.name.startswith("episode_") and (p / "metadata.json").exists()
    ]
    return sorted(episode_dirs, key=lambda p: int(p.name.split("_")[-1]))


def load_metadata(episode_dir: Path) -> dict:
    return json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))


def read_task(episode_dir: Path) -> str:
    return load_metadata(episode_dir)["task"]


def belongs_to_arm(name: str, arm: Arm) -> bool:
    """Whether a camera/sensor name is kept for `arm`. Names without a left/right token are shared."""
    if arm == "both":
        return True
    other = "right" if arm == "left" else "left"
    return other not in name.split("_")


def get_cameras_and_shapes(
    episode_dir: Path, arm: Arm = "both", *, include_tactile: bool = True
) -> tuple[list[str], dict[str, str], dict[str, tuple[int, int]]]:
    """Resolve image streams from one episode's metadata.json.

    Returns (camera_names, camera_to_image_key, camera_to_(height,width)). Sources whose name
    starts with "cam_" are always kept; tactile slots (image key ending in "_tactile", mapped to
    sensor names like "left_gripper_outer") are kept when `include_tactile` (default). Streams that
    belong to the other arm are dropped.
    """
    metadata = load_metadata(episode_dir)
    video_sources: dict[str, str] = metadata["video_sources"]
    video_sizes: dict[str, list[int]] = metadata["video_sizes"]

    camera_to_key = {
        source: key
        for key, source in video_sources.items()
        if (source.startswith("cam_") or (include_tactile and key.endswith("_tactile")))
        and belongs_to_arm(source, arm)
    }
    cameras = sorted(camera_to_key)
    shapes = {cam: (video_sizes[camera_to_key[cam]][1], video_sizes[camera_to_key[cam]][0]) for cam in cameras}
    return cameras, camera_to_key, shapes


def read_csv_array(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a recorder CSV (timestamp + data columns) into (column_names, data[T, D])."""
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    columns = header[1:]  # drop "timestamp"
    if not rows:
        return columns, np.zeros((0, len(columns)), dtype=np.float32)
    data = np.array([[float(v) for v in row[1:]] for row in rows], dtype=np.float32)
    return columns, data


def select_arm_columns(columns: list[str], arm: Arm) -> list[int]:
    """Indices of the joint columns belonging to `arm`."""
    if arm == "both":
        return list(range(len(columns)))
    indices = [i for i, c in enumerate(columns) if arm in c.split("_")]
    if indices:
        return indices
    if any(side in c.split("_") for c in columns for side in ("left", "right")) or len(columns) % 2:
        raise ValueError(f"Cannot determine {arm} arm joints from columns {columns}")
    # No left/right token in the names: ALOHA convention is [left joints..., right joints...].
    half = len(columns) // 2
    return list(range(half)) if arm == "left" else list(range(half, len(columns)))


def read_joint_csv(path: Path, arm: Arm) -> tuple[list[str], np.ndarray]:
    columns, data = read_csv_array(path)
    indices = select_arm_columns(columns, arm)
    return [columns[i] for i in indices], data[:, indices]


def get_motor_names(episode_dir: Path, arm: Arm = "both") -> list[str]:
    columns, _ = read_joint_csv(episode_dir / "observation.state.joint_position" / "data.csv", arm)
    return columns


def load_video_frames(path: Path) -> np.ndarray:
    """Decode all frames of a video as BGR uint8 (no BGR2RGB conversion, see module docstring)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return np.asarray(frames)


def load_raw_episode_data(
    episode_dir: Path,
    cameras: list[str],
    camera_to_key: dict[str, str],
    arm: Arm = "both",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    _, state = read_joint_csv(episode_dir / "observation.state.joint_position" / "data.csv", arm)
    _, action = read_joint_csv(episode_dir / "actions.joint_position" / "data.csv", arm)

    imgs_per_cam = {cam: load_video_frames(episode_dir / camera_to_key[cam] / "video.mp4") for cam in cameras}

    num_frames = state.shape[0]
    if action.shape[0] != num_frames:
        raise ValueError(f"{episode_dir}: state has {num_frames} frames but action has {action.shape[0]}")
    for cam, imgs in imgs_per_cam.items():
        if len(imgs) != num_frames:
            raise ValueError(f"{episode_dir}: {cam} video has {len(imgs)} frames but state has {num_frames}")

    return state, action, imgs_per_cam


def populate_dataset(
    dataset: LeRobotDataset,
    episode_dirs: list[Path],
    cameras: list[str],
    camera_to_key: dict[str, str],
    task_lookup: dict[Path, str],
    episodes: list[int] | None = None,
    arm: Arm = "both",
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(episode_dirs))

    for ep_idx in tqdm.tqdm(episodes, desc="Episodes", leave=False):
        episode_dir = episode_dirs[ep_idx]
        state, action, imgs_per_cam = load_raw_episode_data(episode_dir, cameras, camera_to_key, arm)
        task = task_lookup[episode_dir]

        for i in range(state.shape[0]):
            frame = {
                "observation.state": torch.from_numpy(state[i]),
                "action": torch.from_numpy(action[i]),
                "task": task,
            }
            for cam in cameras:
                frame[f"observation.images.{cam}"] = imgs_per_cam[cam][i]

            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def port_aloha_vtla(
    raw_dir: Path,
    repo_id: str,
    *,
    task: str | None = None,
    episodes: list[int] | None = None,
    arm: Arm = "both",
    include_tactile: bool = True,
    robot_type: str = "aloha2",
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """Convert raw ALOHA2-VTLA episode directories to a single LeRobot dataset.

    `task`, if given, overrides the task label for every episode. Otherwise each episode's task
    label is read from its own metadata.json["task"].

    `arm` keeps "both" arms (default) or only the "left"/"right" arm's cameras, joints and tactile
    images. `include_tactile` (default True) also converts the tactile image streams.
    """
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)

    # A "group" is a folder of episode_* dirs: either raw_dir itself (single task) or one of its
    # immediate subfolders (one task per subfolder).
    direct_episodes = find_episode_dirs(raw_dir)
    groups = [raw_dir] if direct_episodes else sorted(p for p in raw_dir.iterdir() if p.is_dir())

    group_episode_dirs: list[tuple[Path, list[Path]]] = []
    for group in groups:
        episode_dirs = find_episode_dirs(group)
        if not episode_dirs:
            print(f"Warning: no episode_* dirs found in {group}, skipping...")
            continue
        group_episode_dirs.append((group, episode_dirs))

    if not group_episode_dirs:
        raise ValueError(f"No raw VTLA episodes found under {raw_dir}")

    first_episode_dir = group_episode_dirs[0][1][0]
    motors = get_motor_names(first_episode_dir, arm)
    cameras, camera_to_key, image_shapes = get_cameras_and_shapes(
        first_episode_dir, arm, include_tactile=include_tactile
    )
    print(f"Arm: {arm}")
    print(f"Motors: {motors}")
    print(f"Cameras: {camera_to_key}")

    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        motors=motors,
        cameras=cameras,
        image_shapes=image_shapes,
        mode=mode,
        dataset_config=dataset_config,
    )

    task_lookup = {
        episode_dir: task if task is not None else read_task(episode_dir)
        for _, episode_dirs in group_episode_dirs
        for episode_dir in episode_dirs
    }

    for group, episode_dirs in tqdm.tqdm(group_episode_dirs, desc="Processing groups"):
        print(f"Processing {group.name} ({len(episode_dirs)} episodes)")
        dataset = populate_dataset(
            dataset,
            episode_dirs,
            cameras,
            camera_to_key,
            task_lookup,
            episodes=episodes,
            arm=arm,
        )

    return dataset


if __name__ == "__main__":
    tyro.cli(port_aloha_vtla)
