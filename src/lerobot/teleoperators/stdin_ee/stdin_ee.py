#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import sys
import threading
import time
from typing import Any

import torch

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .configuration_stdin_ee import StdinEETeleopConfig


class StdinEETeleop(Teleoperator):
    """
    Teleoperator class that reads VR controller data from stdin.

    Expects JSON messages with the format:
    {
        "sessionId": "uuid",
        "controllers": [
            {
                "position": {"x": float, "y": float, "z": float},
                "rotation": {"x": float, "y": float, "z": float, "w": float},
                "grip": float,
                "trigger": float
            },
            ...
        ]
    }

    Sessions are tracked by sessionId. The first message in each session
    establishes the reference position/rotation. Subsequent messages in the
    same session are converted to relative deltas from that reference.

    Uses controller index 0 (first controller) for single-arm control.
    """

    name = "stdin_ee"
    config_class = StdinEETeleopConfig

    def __init__(self, config: StdinEETeleopConfig):
        super().__init__(config)
        self.config = config

        # Threading for non-blocking stdin reads
        self.latest_message = None
        self.lock = threading.Lock()
        self.read_thread = None
        self.running = False

        # Session tracking
        self.current_session_id = None
        self.first_position = None
        self.first_rotation = None
        self.first_grip = 0.0

        # Controller index to use (from config)
        self.controller_index = config.controller_index if config else 0

        logging.info("StdinEETeleop initialized - will read VR controller data from stdin")

    @property
    def action_features(self) -> dict:
        """Define the action space for end-effector control with rotation and grip."""
        return {
            "delta_x": "float32",
            "delta_y": "float32",
            "delta_z": "float32",
            "rotation_x": "float32",
            "rotation_y": "float32",
            "rotation_z": "float32",
            "rotation_w": "float32",
            "grip": "float32"
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.running and self.read_thread is not None and self.read_thread.is_alive()

    @property
    def is_calibrated(self) -> bool:
        return True  # No calibration needed

    def connect(self, calibrate: bool = True) -> None:
        """Start the background thread to read from stdin."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "StdinEETeleop is already connected. Do not run `connect()` twice."
            )

        logging.info("Starting stdin reader thread...")
        self.running = True
        self.read_thread = threading.Thread(target=self._read_stdin_loop, daemon=True)
        self.read_thread.start()

        # Give the thread a moment to start
        time.sleep(0.1)

        logging.info("StdinEETeleop connected - ready to read JSON messages from stdin")

        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        """No calibration needed for stdin input."""
        pass

    def configure(self) -> None:
        """No configuration needed."""
        pass

    def _read_stdin_loop(self) -> None:
        """Background thread function to continuously read and parse stdin."""
        logging.info("Stdin reader thread started")

        while self.running:
            try:
                # Read line from stdin (blocking)
                line = sys.stdin.readline()

                if not line:  # EOF
                    logging.info("Reached end of stdin stream")
                    break

                line = line.strip()
                if not line:
                    continue

                # Parse JSON message
                try:
                    message = json.loads(line)
                    self._process_message(message)
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse JSON: {e}")
                    continue

            except Exception as e:
                logging.error(f"Error in stdin reader thread: {e}")
                break

        logging.info("Stdin reader thread stopped")

    def _process_message(self, message: dict) -> None:
        """Process a parsed JSON message and update internal state."""
        try:
            session_id = message.get("sessionId")
            controllers = message.get("controllers", [])

            if not session_id or not controllers:
                logging.warning("Message missing sessionId or controllers")
                return

            if len(controllers) <= self.controller_index:
                logging.warning(f"Message has only {len(controllers)} controllers, need at least {self.controller_index + 1}")
                return

            with self.lock:
                # Update latest message
                self.latest_message = message

                # Check if this is a new session
                if session_id != self.current_session_id:
                    self._start_new_session(session_id, controllers[self.controller_index])

        except Exception as e:
            logging.error(f"Error processing message: {e}")

    def _start_new_session(self, session_id: str, controller: dict) -> None:
        """Start a new session and set reference position/rotation."""
        logging.info(f"Starting new session: {session_id}")

        self.current_session_id = session_id

        # Extract reference position and rotation
        pos = controller.get("position", {})
        rot = controller.get("rotation", {})

        self.first_position = torch.tensor([
            pos.get("x", 0.0),
            pos.get("y", 0.0),
            pos.get("z", 0.0)
        ], dtype=torch.float32)

        self.first_rotation = torch.tensor([
            rot.get("x", 0.0),
            rot.get("y", 0.0),
            rot.get("z", 0.0),
            rot.get("w", 1.0)
        ], dtype=torch.float32)

        self.first_grip = controller.get("grip", 0.0)

        logging.info(f"Reference position: {self.first_position}")
        logging.info(f"Reference rotation: {self.first_rotation}")
        logging.info(f"Reference grip: {self.first_grip}")

    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions (q1 * q2)."""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.tensor([x, y, z, w], dtype=torch.float32)

    def _quaternion_inverse(self, q: torch.Tensor) -> torch.Tensor:
        """Compute quaternion inverse (conjugate for unit quaternions)."""
        x, y, z, w = q
        return torch.tensor([-x, -y, -z, w], dtype=torch.float32)

    def get_action(self) -> dict[str, Any]:
        """Get the current action based on the latest VR controller data."""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "StdinEETeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        with self.lock:
            # If no data available, return zero action
            if (self.latest_message is None or
                self.current_session_id is None or
                self.first_position is None or
                self.first_rotation is None):
                return {
                    "delta_x": 0.0,
                    "delta_y": 0.0,
                    "delta_z": 0.0,
                    "rotation_x": 0.0,
                    "rotation_y": 0.0,
                    "rotation_z": 0.0,
                    "rotation_w": 1.0,
                    "grip": 0.0
                }

            # Extract current controller data
            controllers = self.latest_message.get("controllers", [])
            if len(controllers) <= self.controller_index:
                # Return zero action if controller not available
                return {
                    "delta_x": 0.0,
                    "delta_y": 0.0,
                    "delta_z": 0.0,
                    "rotation_x": 0.0,
                    "rotation_y": 0.0,
                    "rotation_z": 0.0,
                    "rotation_w": 1.0,
                    "grip": 0.0
                }

            controller = controllers[self.controller_index]

            # Extract current position and rotation
            pos = controller.get("position", {})
            rot = controller.get("rotation", {})
            grip = controller.get("grip", 0.0)

            current_position = torch.tensor([
                pos.get("x", 0.0),
                pos.get("y", 0.0),
                pos.get("z", 0.0)
            ], dtype=torch.float32)

            current_rotation = torch.tensor([
                rot.get("x", 0.0),
                rot.get("y", 0.0),
                rot.get("z", 0.0),
                rot.get("w", 1.0)
            ], dtype=torch.float32)

            # Compute relative position (delta from first)
            delta_position = current_position - self.first_position

            # Compute relative rotation: first_inverse * current
            first_inv = self._quaternion_inverse(self.first_rotation)
            relative_rotation = self._quaternion_multiply(first_inv, current_rotation)

            # Create action dictionary
            action = {
                "delta_x": delta_position[0].item(),
                "delta_y": delta_position[1].item(),
                "delta_z": delta_position[2].item(),
                "rotation_x": relative_rotation[0].item(),
                "rotation_y": relative_rotation[1].item(),
                "rotation_z": relative_rotation[2].item(),
                "rotation_w": relative_rotation[3].item(),
                "grip": grip
            }

            return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """No feedback mechanism for stdin input."""
        pass

    def disconnect(self) -> None:
        """Stop the background thread and disconnect."""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "StdinEETeleop is not connected. You need to run `connect()` before `disconnect()`."
            )

        logging.info("Disconnecting StdinEETeleop...")
        self.running = False

        if self.read_thread and self.read_thread.is_alive():
            # Give the thread some time to finish
            self.read_thread.join(timeout=1.0)

        logging.info("StdinEETeleop disconnected")
