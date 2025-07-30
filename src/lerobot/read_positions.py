import time
from typing import Generator

import draccus
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.motors import Motor, MotorNormMode, MotorCalibration
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.utils.robot_utils import busy_wait


def loop(fps: float = 60.0) -> Generator[int, None, None]:
    """
    Generator that yields an incrementing frame counter at a fixed rate.

    Parameters
    ----------
    fps : float, default 60.0
        Target frames per second.

    Yields
    ------
    int
        Frame index (starts at 0).
    """
    period = 1.0 / fps
    while True:
        start = time.perf_counter()

        yield None

        elapsed = time.perf_counter() - start
        remaining = period - elapsed
        if remaining > 0:
            busy_wait(remaining)



def main():
    with open(
        "/Users/martin/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/leader_right.json"
    ) as f, draccus.config_type("json"):
        calibration = draccus.load(dict[str, MotorCalibration], f)

    kinematics = RobotKinematics(
        urdf_path="/Users/martin/projects/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
    )

    bus = FeetechMotorsBus(
        port="/dev/tty.usbmodem5A680095901",
        motors={
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        },
        calibration=calibration,
    )

    bus.connect()
    bus.write_calibration(calibration)

    for _ in loop():
        current_joint_pos = bus.sync_read("Present_Position")
        current_joint_pos = np.array(
            [current_joint_pos[name] for name in bus.motors]
        )

        current_ee_position = kinematics.forward_kinematics(current_joint_pos)
        print(current_ee_position[:3, 3])


if __name__ == "__main__":
    main()
