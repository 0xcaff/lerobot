import time
from typing import Generator

import draccus

from lerobot.model.kinematics import RobotKinematics
from lerobot.motors import Motor, MotorNormMode, MotorCalibration
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.utils.robot_utils import busy_wait


def loop(fps: float = 60.0) -> Generator[int, None, None]:
    period = 1.0 / fps
    while True:
        start = time.perf_counter()

        yield None

        elapsed = time.perf_counter() - start
        remaining = period - elapsed
        if remaining > 0:
            busy_wait(remaining)


class Arm:
    bus: FeetechMotorsBus

    def __init__(self, calibration_config_path: str, urdf_path: str, bus_port: str):
        with open(calibration_config_path) as f, draccus.config_type("json"):
            calibration = draccus.load(dict[str, MotorCalibration], f)

        self.kinematics = RobotKinematics(
            urdf_path=urdf_path,
        )

        bus = FeetechMotorsBus(
            port=bus_port,
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

        with bus.torque_disabled():
            bus.configure_motors()
            for motor in bus.motors:
                bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                bus.write("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                bus.write("I_Coefficient", motor, 0)
                bus.write("D_Coefficient", motor, 32)

        self.bus = bus


def main():
    urdf_path = "/Users/martin/projects/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
    calibration_config_path_base = "/Users/martin/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader"

    leader_l = Arm(
        calibration_config_path=calibration_config_path_base + "/actual_leader_left.json",
        urdf_path=urdf_path,
        bus_port="/dev/tty.usbmodem5A680133731",
    )
    leader_r = Arm(
        calibration_config_path=calibration_config_path_base + "/actual_leader_right.json",
        urdf_path=urdf_path,
        bus_port="/dev/tty.usbmodem5A680089711",
    )

    follower_l = Arm(
        calibration_config_path=calibration_config_path_base + "/leader_right.json",
        urdf_path=urdf_path,
        bus_port="/dev/tty.usbmodem5A680095901",
    )
    follower_r = Arm(
        calibration_config_path=calibration_config_path_base + "/leader_left.json",
        urdf_path=urdf_path,
        bus_port="/dev/tty.usbmodem5A680120861",
    )

    with leader_l.bus.torque_disabled(), leader_r.bus.torque_disabled():
        for _ in loop():
            l_current_joint_pos = leader_l.bus.sync_read("Present_Position")
            r_current_joint_pos = leader_r.bus.sync_read("Present_Position")

            follower_l.bus.sync_write("Goal_Position", l_current_joint_pos)
            follower_r.bus.sync_write("Goal_Position", r_current_joint_pos)

if __name__ == "__main__":
    main()
