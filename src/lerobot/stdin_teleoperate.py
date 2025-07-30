import json
import logging
import sys
import threading
import time
import typing
from dataclasses import dataclass
from typing import Generator, TypedDict, List

import draccus
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.motors import Motor, MotorNormMode, MotorCalibration
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.utils.robot_utils import busy_wait


class StdinLatest:
    def __init__(self):
        self._latest = None
        self._lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in sys.stdin:
            with self._lock:
                self._latest = line.rstrip("\n")

    def get(self) -> str | None:
        with self._lock:
            v, self._latest = self._latest, None
        return v


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


class Vector3(TypedDict):
    x: float
    y: float
    z: float


class Quaternion(TypedDict):
    x: float
    y: float
    z: float
    w: float


class ControllerState(TypedDict):
    position: Vector3
    rotation: Quaternion
    grip: float
    trigger: float


class ControllerPositionMessage(TypedDict):
    sessionId: str
    controllers: List[ControllerState]


@dataclass
class SessionState:
    session_id: str
    starting_joint_values: np.ndarray
    starting_end_effector_position: np.ndarray
    starting_controller: ControllerState


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

    controller_idx = 0

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

    stdin_reader = StdinLatest()

    session_state: None | SessionState = None
    last_solution = None

    for _ in loop():
        last_line = stdin_reader.get()
        if last_line is None:
            continue

        try:
            message = typing.cast(ControllerPositionMessage, json.loads(last_line))
        except Exception as e:
            logging.warning(f"failed to parse json {e}")
            continue

        controllers = message["controllers"]
        if len(controllers) < controller_idx + 1:
            logging.warning("not enough controllers in session")
            continue

        controller = controllers[controller_idx]

        if session_state is None or session_state.session_id != message["sessionId"]:
            current_joint_pos = bus.sync_read("Present_Position")
            current_joint_pos = np.array(
                [current_joint_pos[name] for name in bus.motors]
            )

            current_ee_position = kinematics.forward_kinematics(current_joint_pos)

            session_state = SessionState(
                session_id=message["sessionId"],
                starting_joint_values=current_joint_pos,
                starting_end_effector_position=current_ee_position,
                starting_controller=controller,
            )
            last_solution = current_joint_pos.copy()
            continue

        action = np.array(
            [
                -(controller["position"]["z"]
                 - session_state.starting_controller["position"]["z"]),
                -(controller["position"]["x"]
                - session_state.starting_controller["position"]["x"]),
                (controller["position"]["y"]
                 - session_state.starting_controller["position"]["y"]),
            ]
        )

        desired_ee_pos = np.eye(4)
        desired_ee_pos[:3, 3] = session_state.starting_end_effector_position[:3, 3] + action[:3]

        target_joint_values_in_degrees = kinematics.inverse_kinematics(
            last_solution, desired_ee_pos, orientation_weight=0.00
        )

        last_solution = target_joint_values_in_degrees
        joint_action = {"gripper": (1 - controller["grip"]) * 50} | {
            key: target_joint_values_in_degrees[i]
            for i, key in enumerate(bus.motors.keys())
        }

        bus.sync_write("Goal_Position", joint_action)


if __name__ == "__main__":
    main()
