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

from scipy.spatial.transform import Rotation as R


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


def v3_to_np(v: Vector3) -> np.ndarray:
    return np.array([v["x"], v["y"], v["z"]], dtype=float)


class Quaternion(TypedDict):
    x: float
    y: float
    z: float
    w: float


def quaternion_to_np(q: Quaternion) -> np.ndarray:
    return np.array(
        [
            q["x"],
            q["y"],
            q["z"],
            q["w"],
        ]
    )


# fixed controller‑to‑robot mapping rotation
R_MAP = R.from_matrix(
    np.array(
        [
            [0.0, 0.0, -1.0],  # +Xc → −Zr
            [-1.0, 0.0, 0.0],  # +Yc → −Xr
            [0.0, 1.0, 0.0],  # +Zc → +Yr
        ]
    )
)


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


class Arm:
    session_state: None | SessionState = None
    bus: FeetechMotorsBus
    last_solution: np.ndarray

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

    def update(self, controller: ControllerState, session_id: str):
        if self.session_state is None or self.session_state.session_id != session_id:
            current_joint_pos = self.bus.sync_read("Present_Position")
            current_joint_pos = np.array(
                [current_joint_pos[name] for name in self.bus.motors]
            )

            current_ee_position = self.kinematics.forward_kinematics(current_joint_pos)

            self.session_state = SessionState(
                session_id=session_id,
                starting_joint_values=current_joint_pos,
                starting_end_effector_position=current_ee_position,
                starting_controller=controller,
            )
            self.last_solution = current_joint_pos.copy()
            return

        assert self.session_state is not None and self.last_solution is not None

        action = R_MAP.apply(
            v3_to_np(controller["position"])
            - v3_to_np(self.session_state.starting_controller["position"])
        )

        R_start = R.from_quat(
            quaternion_to_np(self.session_state.starting_controller["rotation"])
        )
        R_now = R.from_quat(quaternion_to_np(controller["rotation"]))
        delta_R = R_MAP * (R_now * R_start.inv()) * R_MAP.inv()

        T_start_rot = R.from_matrix(
            self.session_state.starting_end_effector_position[:3, :3]
        )

        desired_ee_pos = np.block(
            [
                [
                    (delta_R * T_start_rot).as_matrix(),
                    (
                        self.session_state.starting_end_effector_position[:3, 3]
                        + action[:3]
                    ).reshape(3, 1),
                ],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]
        )

        target_joint_values_in_degrees = self.kinematics.inverse_kinematics(
            self.last_solution, desired_ee_pos, orientation_weight=0.20
        )

        self.last_solution = target_joint_values_in_degrees
        joint_action = {
            key: target_joint_values_in_degrees[i]
            for i, key in enumerate(self.bus.motors.keys())
        } | {"gripper": (1 - controller["grip"]) * 50}

        self.bus.sync_write("Goal_Position", joint_action)


def main():
    urdf_path = "/Users/martin/projects/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
    calibration_config_path_base = "/Users/martin/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader"

    arms = [
        Arm(
            calibration_config_path=calibration_config_path_base + "/leader_right.json",
            urdf_path=urdf_path,
            bus_port="/dev/tty.usbmodem5A680095901",
        ),
        Arm(
            calibration_config_path=calibration_config_path_base + "/leader_left.json",
            urdf_path=urdf_path,
            bus_port="/dev/tty.usbmodem5A680120861",
        ),
    ]

    stdin_reader = StdinLatest()

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
        for controller_idx, arm in enumerate(arms):
            if len(controllers) < controller_idx + 1:
                logging.warning("not enough controllers in session")
                continue

            controller = controllers[controller_idx]
            arm.update(controller, session_id=message["sessionId"])


if __name__ == "__main__":
    main()
