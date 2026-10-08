#  Copyright (c) 2019 Diego Damasceno
#
#  This file is part of pygame-logitechG29_wheel.
#  Documentation, related files, and licensing can be found at
#
#      <https://github.com/damascenodiego/pygame-logitechG29_wheel>.


import pygame
import sys
import os
import argparse
import time

if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:
    import ConfigParser.RawConfigParser as ConfigParser

class Controller:

    def __init__(self, id, dead_zone = 0.15):
        """
        Initializes a controller.

        Args:
            id: The ID of the controller which must be a value from `0` to
                `pygame.joystick.get_count() - 1`
            dead_zone: The size of dead zone for the    analog sticks (default 0.15)
        """

        self._joystick = pygame.joystick.Joystick(id)
        self._joystick.init()
        self.dead_zone = dead_zone

        self._parser = ConfigParser()
        self._parser.read(os.path.join(os.path.dirname(__file__), 'wheel_config.ini'))
        self._steer_idx     = int(self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._clutch_idx  = int(self._parser.get('G29 Racing Wheel', 'gear'))
        self._throttle_idx = int(self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx     = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx   = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(self._parser.get('G29 Racing Wheel', 'handbrake'))


    def get_id(self):
        """
        Returns:
            The ID of the controller. This is the same as the ID passed into
            the initializer.
        """

        return self._joystick.get_id()

    def get_buttons(self):
        """
        Gets the state of each button on the controller.

        Returns:
            A tuple with the state of each button. 1 is pressed, 0 is unpressed.
        """

        numButtons = self._joystick.get_numbuttons()
        jsButtons = [float(self._joystick.get_button(i)) for i in range(numButtons)]

        return (jsButtons)


    def get_axis(self):
        """
        Gets the state of each axis on the controller.

        Returns:
            The axes values x as a tuple such that

            -1 <= x <= 1

        """

        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]


        return (jsInputs)


    def get_steer(self):
        """
        Gets the state of the steering wheel.

        Returns:
            A value x such that

            -1 <= x <= 1 && -1 <= y <= 1

            Negative values are left.
            Positive values are right.
        """


        return (self.get_axis()[self._steer_idx])


    def get_clutch(self):
        """
        Gets the state of the gear pedal.

        Returns:
            A value x such that

            -1 <= x <= 1

        """


        return (self.get_axis()[self._clutch_idx])


    def get_break(self):
        """
        Gets the state of the break pedal.

        Returns:
            A value x such that

            -1 <= x <= 1

        """


        return (self.get_axis()[self._brake_idx])



    def get_throttle(self):
        """
        Gets the state of the throttle pedal.

        Returns:
            A value x such that

            -1 <= x <= 1

        """


        return (self.get_axis()[self._throttle_idx])


    def get_reverse(self):
        """
        Gets the state of the reverse button.

        Returns:
            A value x such that 1 is pressed, 0 is unpressed.

        """


        return (self.get_buttons()[self._reverse_idx])


    def get_handbrake(self):
        """
        Gets the state of the handbrake.

        Returns:
            A value x such that 1 is pressed, 0 is unpressed.
        """


        return (self.get_buttons()[self._handbrake_idx])


def _format_values(values):
    return "[" + ", ".join(f"{value:+.3f}" for value in values) + "]"


def main():
    parser = argparse.ArgumentParser(description="Print Logitech G29 wheel inputs for testing.")
    parser.add_argument("--id", type=int, default=0, help="pygame joystick id to open.")
    parser.add_argument("--hz", type=float, default=20.0, help="Polling/printing rate.")
    parser.add_argument("--dead-zone", type=float, default=0.15, help="Controller dead zone.")
    parser.add_argument("--raw", action="store_true", help="Also print all raw axes and buttons.")
    args = parser.parse_args()

    pygame.init()
    pygame.joystick.init()
    try:
        joystick_count = pygame.joystick.get_count()
        print(f"Detected {joystick_count} joystick(s).")
        for joystick_id in range(joystick_count):
            joystick = pygame.joystick.Joystick(joystick_id)
            joystick.init()
            print(
                f"  id={joystick_id} name={joystick.get_name()!r} "
                f"axes={joystick.get_numaxes()} buttons={joystick.get_numbuttons()}"
            )

        if joystick_count == 0:
            raise SystemExit("No joystick detected.")
        if args.id < 0 or args.id >= joystick_count:
            raise SystemExit(f"Invalid joystick id {args.id}; expected 0..{joystick_count - 1}.")

        controller = Controller(args.id, dead_zone=args.dead_zone)
        sleep_s = max(1.0 / args.hz, 0.001)
        print("Polling G29 inputs. Press Ctrl+C or close the pygame window to quit.")

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            fields = [
                f"steer={controller.get_steer():+.3f}",
                f"clutch={controller.get_clutch():+.3f}",
                f"throttle={controller.get_throttle():+.3f}",
                f"brake={controller.get_break():+.3f}",
                f"reverse={controller.get_reverse():.0f}",
                f"handbrake={controller.get_handbrake():.0f}",
            ]
            if args.raw:
                fields.append(f"axes={_format_values(controller.get_axis())}")
                fields.append(f"buttons={_format_values(controller.get_buttons())}")
            print(" ".join(fields), flush=True)
            time.sleep(sleep_s)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.joystick.quit()
        pygame.quit()


if __name__ == "__main__":
    main()
