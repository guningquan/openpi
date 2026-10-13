import dataclasses
import logging
from typing import Literal

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.aloha_real import env as _env
from examples.aloha_real.keyboard_control import UserAborted

from examples.aloha_real.sleep_modify import shut_down_puppet_robots


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 50  # 25 is the default value

    max_hz: float = 50  # control loop rate

    num_episodes: int = 1
    max_episode_steps: int = 1500 

    # Must match the policy's training data: "right"/"left" for a single-arm policy (e.g.
    # pi05_board_insertion_right_full), in which case the other arm is held still.
    arm: Literal["both", "left", "right"] = "both"


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    metadata = ws_client_policy.get_server_metadata()
    environment = _env.AlohaRealEnvironment(reset_position=metadata.get("reset_pose"), arm=args.arm)
    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=args.max_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    try:
        runtime.run()
    except UserAborted:
        logging.info("Cancelled by user (ESC/q).")
    finally:
        # Always park the arms to their sleep pose and torque off, whether the run finished
        # normally, was cancelled (ESC/q), or hit an error/Ctrl+C.
        shut_down_puppet_robots()
        environment.close()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
