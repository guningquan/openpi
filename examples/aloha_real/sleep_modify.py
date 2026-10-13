import sys
from pathlib import Path

# Add project root to Python path to enable absolute imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from interbotix_xs_modules.arm import InterbotixManipulatorXS
try:
    from examples.aloha_real.robot_utils import move_arms, torque_on, move_grippers, torque_off
    from examples.aloha_real.constants import MASTER_LEFT_GRIPPER_JOINT_OPEN, MASTER_RIGHT_GRIPPER_JOINT_OPEN
    from examples.aloha_real.constants import PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN
except ImportError:
    from robot_utils import move_arms, torque_on, move_grippers, torque_off
    from constants import MASTER_LEFT_GRIPPER_JOINT_OPEN, MASTER_RIGHT_GRIPPER_JOINT_OPEN
    from constants import PUPPET_LEFT_GRIPPER_JOINT_OPEN, PUPPET_RIGHT_GRIPPER_JOINT_OPEN
import rospy
import threading,time
import argparse

# Sleep poses, kept in sync with act_tactile/aloha_scripts/sleep.py
PUPPET_SLEEP_ALL_POSITION = (0, -1.85, 1.6, 0.06, 0.65, 0)
PUPPET_SLEEP_POSITION = (0, -1.85, 1.6, 0.12, 0.65, 0)
MASTER_SLEEP_POSITION = (0, -1.85, 1.6, 0, 0, 0)

def move_robot(robot, gripper_position, arm_position, move_time):
    torque_on(robot)
    move_grippers([robot], [gripper_position], move_time=move_time)
    move_arms([robot], [arm_position], move_time=move_time)
    # torque_off(robot)

def move_shut_robot(robot, gripper_position, arm_position, move_time):
    torque_on(robot)
    move_grippers([robot], [gripper_position], move_time=move_time)
    move_arms([robot], [arm_position], move_time=move_time)
    time.sleep(1)
    torque_off(robot)

# Two-stage shutdown waypoints for shut_down_puppet_robots(), ported from
# act-triple-plus-aloha2/aloha_scripts/sleep_plus.py: going straight from a rollout's working
# pose to the final sleep pose in one move has been observed to drop the arm when torque cuts,
# so visit the old, well-traveled pose first, then make a small final correction onto the
# measured rest pose right before torque cuts.
PUPPET_SHUTDOWN_WAYPOINTS = ((0, -1.7, 1.55, 0.12, 0.65, 0), (0, -1.85, 1.594, 0, 0.782, 0))
PUPPET_SHUTDOWN_MOVE_TIMES = (2.0, 2.0)


def move_shut_robot_2stage(robot, gripper_position, arm_waypoints, move_times):
    """Visit arm_waypoints in order, then cut torque - see PUPPET_SHUTDOWN_WAYPOINTS above."""
    torque_on(robot)
    move_grippers([robot], [gripper_position], move_time=move_times[0])
    for arm_position, move_time in zip(arm_waypoints, move_times):
        move_arms([robot], [arm_position], move_time=move_time)
    time.sleep(0.3)
    torque_off(robot)

def shut_down_all_robots():
    # 创建机械臂实例
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                              robot_name='puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                               robot_name='puppet_right', init_node=False)
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name='master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name='master_right', init_node=False)

    puppet_sleep_position = (0, -1.7, 1.55, 0.12, 0.65, 0)
    master_sleep_position = (0, -1.76, 1.55, 0, 0.0, 0)

    # 创建线程
    threads = []
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(puppet_bot_left, PUPPET_LEFT_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(puppet_bot_right, PUPPET_RIGHT_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(master_bot_left, MASTER_LEFT_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))
    threads.append(threading.Thread(target=move_shut_robot,
                                    args=(master_bot_right, MASTER_RIGHT_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

def shut_down_puppet_robots():
    # 创建机械臂实例
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                              robot_name='puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                               robot_name='puppet_right', init_node=False)

    # 创建线程 - 两步 sleep：先走旧的（已验证过很多次的）中间姿态，再做最后的小幅修正，
    # 避免直接从工作姿态大幅度跳到最终 sleep 姿态、断电时把机械臂摔一下。
    threads = []
    threads.append(threading.Thread(target=move_shut_robot_2stage,
                                    args=(puppet_bot_left, PUPPET_LEFT_GRIPPER_JOINT_OPEN,
                                          PUPPET_SHUTDOWN_WAYPOINTS, PUPPET_SHUTDOWN_MOVE_TIMES)))
    threads.append(threading.Thread(target=move_shut_robot_2stage,
                                    args=(puppet_bot_right, PUPPET_RIGHT_GRIPPER_JOINT_OPEN,
                                          PUPPET_SHUTDOWN_WAYPOINTS, PUPPET_SHUTDOWN_MOVE_TIMES)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()


def sleep_all_robots():
    # 创建机械臂实例
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                              robot_name='puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper",
                                               robot_name='puppet_right', init_node=False)
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name='master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name='master_right', init_node=False)

    puppet_sleep_position = PUPPET_SLEEP_ALL_POSITION
    master_sleep_position = MASTER_SLEEP_POSITION

    # 创建线程
    threads = []
    threads.append(threading.Thread(target=move_robot,
                                    args=(puppet_bot_left, PUPPET_LEFT_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(puppet_bot_right, PUPPET_RIGHT_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(master_bot_left, MASTER_LEFT_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(master_bot_right, MASTER_RIGHT_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()


def sleep_puppet_robots():
    # rospy.init_node('sleep_modify')

    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_left', init_node=False)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_right', init_node=False)

    puppet_sleep_position = PUPPET_SLEEP_POSITION

    threads = []
    threads.append(threading.Thread(target=move_robot,
                                    args=(puppet_bot_left, PUPPET_LEFT_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(puppet_bot_right, PUPPET_RIGHT_GRIPPER_JOINT_OPEN, puppet_sleep_position, 1)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

def sleep_master_robots():
    # 创建机械臂实例
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name='master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name='master_right', init_node=False)

    master_sleep_position = MASTER_SLEEP_POSITION

    # 创建线程
    threads = []
    threads.append(threading.Thread(target=move_robot,
                                    args=(master_bot_left, MASTER_LEFT_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))
    threads.append(threading.Thread(target=move_robot,
                                    args=(master_bot_right, MASTER_RIGHT_GRIPPER_JOINT_OPEN, master_sleep_position, 1)))

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()


def main():
    parser = argparse.ArgumentParser(description="Control script for robots")
    parser.add_argument("--sleep", action="store_true",
                        help="Put all robots into sleep mode")
    parser.add_argument("--shut_down", action="store_true",
                        help="Shut down all robots")
    parser.add_argument("--shut_down_puppet", action="store_true",
                        help="Shut down puppet robots")

    args = parser.parse_args()

    if args.sleep:
        sleep_all_robots()
    elif args.shut_down:
        shut_down_all_robots()
    elif args.shut_down_puppet:
        shut_down_puppet_robots()
    else:
        print("Please provide a valid argument (--sleep or --shut_down).")

if __name__ == "__main__":
    rospy.init_node('sleep_modify')
    main()
