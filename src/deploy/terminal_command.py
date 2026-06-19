#!/usr/bin/env python3

"""
The initial author of this file is [YixuanQiu](https://github.com/YixuanQiu).
The contents have since been modified.
"""

import argparse
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
from typing import Dict

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

# Constants
MAX_LINEAR_SPEED = 1.5  # m/s
MAX_LATERAL_SPEED = 0.8  # m/s
MAX_ANGULAR_SPEED = 1.5  # rad/s
SPEED_INCREMENT = 0.1

# Key mappings
KEY_UP = "\x1b[A"
KEY_DOWN = "\x1b[B"
KEY_RIGHT = "\x1b[C"
KEY_LEFT = "\x1b[D"
KEY_W = "w"  # Forward
KEY_S = "s"  # Backward
KEY_A = "a"  # Strafe left
KEY_D = "d"  # Strafe right
KEY_Q = "q"
KEY_E = "e"  # Turn left (positive wz)
KEY_R = "r"  # Turn right (negative wz)
KEY_SPACE = " "
KEY_CTRL_C = "\x03"


# For non-blocking key detection
def getch():
    """Gets a single character from standard input, does not echo to the screen."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        # Handle arrow keys (they send multiple chars)
        if ch == "\x1b":
            ch = ch + sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


class KeyboardController(Node):
    def __init__(self, ctrl_mode: str):
        super().__init__("keyboard_controller")

        # Control state - velocity commands
        self.forward_speed = 0.0  # vx (m/s)
        self.lateral_speed = 0.0  # vy (m/s)
        self.angular_speed = 0.0  # wz (rad/s)

        # Create ROS2 publisher for velocity commands (vx, vy, wz)
        if ctrl_mode == "diffloco":
            self.command_publisher = self.create_publisher(
                PointStamped, "/velocity_command", 10
            )
        else:
            raise ValueError("Invalid control mode. Use 'diffloco'.")

        # Create command messages
        self.command_msg = PointStamped()  # Velocity command (point.x=vx, point.y=vy, point.z=wz)

        # Initialize velocity command (vx, vy, wz)
        self.command_msg.point.x = 0.0  # forward speed (vx)
        self.command_msg.point.y = 0.0  # lateral speed (vy)
        self.command_msg.point.z = 0.0  # angular speed (wz)

        # Thread control
        self.running = False
        self.publish_thread = None
        self._cleaned_up = False

        # Initialize to handle terminal resize and exit cleanly
        signal.signal(signal.SIGWINCH, self.handle_resize)
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

    def handle_resize(self, *args):
        """Handle terminal resize events"""
        # Re-draw the UI
        self.clear_screen()
        self.draw_control_state()

    def handle_interrupt(self, *args):
        """Handle interrupt signals"""
        self.running = False

    def clear_screen(self):
        """Clear the terminal screen"""
        os.system("clear")

    def draw_control_state(self):
        """Draw the current state to the terminal"""
        self.clear_screen()

        # Draw text instructions and status
        instructions = [
            "Go2 Locomotion Control - Terminal Version",
            "",
            "=== VELOCITY CONTROLS ===",
            "↑/↓: Increase/Decrease Forward Speed (vx)",
            "←/→: Increase/Decrease Strafe Speed (vy, Left/Right)",
            "a/d: Increase/Decrease Turn Speed (wz, Left/Right)",
            "",
            "",
            "SPACE: Clear all commands (STOP)",
            "q: Quit",
            "",
            "=== CURRENT MOVEMENT ===",
        ]

        # Show current movement direction
        movement_status = []
        if self.forward_speed > 0:
            movement_status.append(f"FORWARD ({self.forward_speed:.2f} m/s)")
        elif self.forward_speed < 0:
            movement_status.append(f"BACKWARD ({abs(self.forward_speed):.2f} m/s)")

        if self.lateral_speed > 0:
            movement_status.append(f"STRAFING LEFT ({self.lateral_speed:.2f} m/s)")
        elif self.lateral_speed < 0:
            movement_status.append(
                f"STRAFING RIGHT ({abs(self.lateral_speed):.2f} m/s)"
            )

        if self.angular_speed > 0:
            movement_status.append(f"TURNING LEFT ({self.angular_speed:.2f} rad/s)")
        elif self.angular_speed < 0:
            movement_status.append(
                f"TURNING RIGHT ({abs(self.angular_speed):.2f} rad/s)"
            )

        if not movement_status:
            movement_status.append("STOPPED")

        instructions.extend(movement_status)

        instructions.extend(
            [
                "",
                "=== CURRENT STATE ===",
                f"Forward Speed (vx): {self.forward_speed:.2f} m/s",
                f"Lateral Speed (vy): {self.lateral_speed:.2f} m/s",
                f"Angular Speed (wz): {self.angular_speed:.2f} rad/s",
            ]
        )

        # Print all instructions
        print("\n".join(instructions))

    def clear_all_commands(self):
        """Reset all speeds to zero"""
        self.forward_speed = 0.0
        self.lateral_speed = 0.0
        self.angular_speed = 0.0

    def update_speed_from_key(self, key):
        """Update speeds based on key press with conflict resolution"""
        # Velocity controls
        if key == KEY_UP:  # Forward
            self.forward_speed = min(
                self.forward_speed + SPEED_INCREMENT, MAX_LINEAR_SPEED
            )
        elif key == KEY_DOWN:  # Backward
            self.forward_speed = max(
                self.forward_speed - SPEED_INCREMENT, -MAX_LINEAR_SPEED
            )
        elif key == KEY_LEFT:  # Strafe left
            self.lateral_speed = min(
                self.lateral_speed + SPEED_INCREMENT, MAX_LATERAL_SPEED
            )
        elif key == KEY_RIGHT:  # Strafe right
            self.lateral_speed = max(
                self.lateral_speed - SPEED_INCREMENT, -MAX_LATERAL_SPEED
            )
        # Angular velocity controls (wz)
        elif key == KEY_A:  # Turn left (positive wz)
            self.angular_speed = min(
                self.angular_speed + SPEED_INCREMENT, MAX_ANGULAR_SPEED
            )
        elif key == KEY_D:  # Turn right (negative wz)
            self.angular_speed = max(
                self.angular_speed - SPEED_INCREMENT, -MAX_ANGULAR_SPEED
            )
        elif key == KEY_SPACE:  # Clear all commands
            self.clear_all_commands()

    def publish_command(self):
        """Publish velocity commands (vx, vy, wz) to the locomotion node"""
        # Set velocities in the velocity command message
        self.command_msg.header.stamp = self.get_clock().now().to_msg()
        self.command_msg.point.x = float(self.forward_speed)  # vx
        self.command_msg.point.y = float(self.lateral_speed)  # vy
        self.command_msg.point.z = float(self.angular_speed)  # wz

        # Publish the velocity command
        self.command_publisher.publish(self.command_msg)

    def publisher_thread_function(self):
        """Function that runs in the publisher thread"""
        while self.running and rclpy.ok():
            self.publish_command()
            time.sleep(0.02)  # 50 Hz

    def cleanup(self):
        """Clean up before exit"""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        # Send zero commands
        self.forward_speed = 0.0
        self.lateral_speed = 0.0
        self.angular_speed = 0.0
        if rclpy.ok():
            self.publish_command()

        # Restore terminal
        os.system("stty sane")
        print("\033[?25h")  # Show cursor

    def run(self):
        """Main control loop"""
        self.running = True

        try:
            # Hide cursor
            print("\033[?25l")

            # Start publisher thread
            self.publish_thread = threading.Thread(
                target=self.publisher_thread_function
            )
            self.publish_thread.daemon = True
            self.publish_thread.start()

            # Draw the initial UI
            self.draw_control_state()

            # Main input loop - process one key at a time
            while self.running and rclpy.ok():
                key = getch()

                # Process key
                if key == KEY_Q or key == KEY_CTRL_C:  # q or Ctrl+C
                    self.running = False
                    break

                # Update speed based on key press
                if key in [
                    KEY_UP,
                    KEY_DOWN,
                    KEY_LEFT,
                    KEY_RIGHT,
                    KEY_A,
                    KEY_D,
                    KEY_W,
                    KEY_S,
                    KEY_SPACE,
                ]:
                    self.update_speed_from_key(key)
                    self.draw_control_state()

        except KeyboardInterrupt:
            self.running = False
            print("Keyboard interrupt received, shutting down")
        finally:
            self.cleanup()

            if self.publish_thread:
                self.publish_thread.join(timeout=1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go2 Locomotion Control")
    parser.add_argument(
        "--control",
        type=str,
        choices=["diffloco"],
        default="diffloco",
        help="Control mode: 'diffloco' for differential locomotion commands",
    )
    args = parser.parse_args()

    # Initialize ROS2
    rclpy.init()

    controller = KeyboardController(ctrl_mode=args.control)

    def spin_controller():
        try:
            rclpy.spin(controller)
        except ExternalShutdownException:
            pass
        except Exception:
            if rclpy.ok():
                raise

    ros_thread = threading.Thread(target=spin_controller)
    ros_thread.start()

    try:
        controller.run()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down")
    finally:
        controller.running = False
        controller.cleanup()
        if rclpy.ok():
            rclpy.shutdown()
        ros_thread.join(timeout=1.0)
        controller.destroy_node()
        print("Exiting controller")
