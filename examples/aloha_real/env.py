import time
from typing import List, Literal, Optional  # noqa: UP035

import einops
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.aloha_real import real_env as _real_env
from examples.aloha_real.keyboard_control import KeyboardControl, ResetRequested, UserAborted


class AlohaRealEnvironment(_environment.Environment):
    """An environment for an Aloha robot on real hardware."""

    def __init__(
        self,
        reset_position: Optional[List[float]] = None,  # noqa: UP006,UP007
        render_height: int = 224,
        render_width: int = 224,
        arm: Literal["both", "left", "right"] = "both",
    ) -> None:
        self._env = _real_env.make_real_env(init_node=True, reset_position=reset_position)
        self._render_height = render_height
        self._render_width = render_width
        self._keyboard = KeyboardControl()

        # For a single-arm policy ("left"/"right"), only that arm's state [6 joints, gripper] and
        # wrist camera are sent to the policy, which returns 7-dim actions. The other arm is held
        # at the joint pose it has right after reset(), with its gripper closed.
        self._arm = arm
        self._hold_action = None

        self._ts = None

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()
        if self._arm != "both":
            other = slice(7, 14) if self._arm == "left" else slice(0, 7)
            hold = np.array(self._ts.observation["qpos"][other], dtype=np.float64)
            # qpos holds the gripper *position*, but step() takes a normalized gripper *joint*
            # command; 1 is the open joint position that reset() leaves the gripper in.
            hold[6] = 1.0
            self._hold_action = hold

    def _abort(self) -> None:
        print("ESC/q detected: stopping and moving the arms to their sleep position...")
        raise UserAborted

    def wait_for_start(self, episode_idx: int, num_episodes: int) -> None:
        """Block until the operator presses the right arrow key, or raise UserAborted on ESC/q.

        Called by Runtime right after reset() has already homed the arms. If stdin isn't a
        tty (keyboard control unavailable), skip waiting and start immediately.
        """
        if not self._keyboard.enabled:
            print(f"Episode {episode_idx + 1}/{num_episodes}: no tty on stdin, starting immediately.")
            return
        print(f"Episode {episode_idx + 1}/{num_episodes}: press the RIGHT arrow key to start (ESC/q to cancel).")
        self._keyboard.drain()
        while True:
            for event in self._keyboard.poll():
                if event == "right":
                    print(
                        f"Episode {episode_idx + 1}/{num_episodes}: running - press LEFT to reset to home, "
                        "q or ESC to cancel."
                    )
                    return
                if event == "esc":
                    self._abort()
            time.sleep(0.02)

    @override
    def is_episode_complete(self) -> bool:
        for event in self._keyboard.poll():
            if event == "esc":
                self._abort()
            if event == "left":
                # main.py catches this and reruns the runtime, which homes the arms and waits for RIGHT.
                print("LEFT detected: stopping and resetting to home...")
                raise ResetRequested
        return False

    def close(self) -> None:
        self._keyboard.close()

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation
        for k in list(obs["images"].keys()):
            if "_depth" in k:
                del obs["images"][k]

        for cam_name in obs["images"]:
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
            )
            obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")

        state = obs["qpos"]
        images = obs["images"]
        if self._arm != "both":
            state = state[7:14] if self._arm == "right" else state[0:7]
            other_wrist = "cam_left_wrist" if self._arm == "right" else "cam_right_wrist"
            images = {name: img for name, img in images.items() if name != other_wrist}

        return {
            "state": state,
            "images": images,
        }

    @override
    def apply_action(self, action: dict) -> None:
        actions = np.asarray(action["actions"])
        if self._arm != "both":
            if actions.shape[-1] != 7:
                raise ValueError(f"Expected a 7-dim action for arm={self._arm!r}, got shape {actions.shape}")
            if self._arm == "right":
                actions = np.concatenate([self._hold_action, actions])
            else:
                actions = np.concatenate([actions, self._hold_action])
        self._ts = self._env.step(actions)
