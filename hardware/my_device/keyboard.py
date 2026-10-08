import numpy as np
from pynput.keyboard import Key, Listener

class Keyboard:

    def __init__(self, is_multi_robot_env=False, allowed_keys=None):

        self.allowed_keys = None if allowed_keys is None else set(allowed_keys)
        self._display_controls(self.allowed_keys)
        self.start = False
        self.finish = False
        self.discard = False
        self.success = False
        self.fail = False
        self.quit = False
        self.ctn = False
        self.switch = False
        self.good = False
        self.bad = False
        self.help = False
        self.infer = False
        self.manual_reset = False
        self.listener = None
        # make a thread to listen to keyboard and register our callback functions
        if not is_multi_robot_env:
            self.create_listener()
       
    def create_listener(self):
        self.listener = Listener(
            on_press=self.on_press, on_release=self.on_release)
        # start listening
        self.listener.start()

    def kill_listener(self):
        if self.listener:
            self.listener.stop()  # Stop the listener first
            self.listener = None

    @staticmethod
    def _display_controls(allowed_keys=None):
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys", "Command")
        def maybe_print(char, info):
            if allowed_keys is None or char in allowed_keys:
                print_command(char, info)

        maybe_print("s", "start a demo")
        maybe_print("f", "finish a demo")
        maybe_print("d", "discard a demo")
        ##########
        maybe_print("j", "success")
        maybe_print("k", "fail")
        maybe_print("q", "quit")
        ##########
        maybe_print("c", "continue")
        maybe_print("x", "switch")
        ##########
        maybe_print("g", "good")
        maybe_print("b", "bad")
        ###########
        maybe_print("h", "human")
        maybe_print("r", "robot")
        maybe_print("z", "manual reset")
        print("")

    def _key_allowed(self, char):
        return self.allowed_keys is None or char in self.allowed_keys

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """

        try:
            if not self._key_allowed(key.char):
                return
            if key.char == "c":
                print('continue!')
                self.ctn = True
            elif key.char == "x":
                print('switch!')
                self.switch = True
            elif key.char == "g":
                print('good!')
                self.good = True
            elif key.char == "b":
                print('bad!')
                self.bad = True

        except AttributeError as e:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """

        try:
            if not self._key_allowed(key.char):
                return
            if key.char == "f":
                self.finish = True
            elif key.char == "d":
                self.discard = True
            elif key.char == "s":
                self.start = True
                print('start!')
            elif key.char == "j":
                self.success = True
            elif key.char == "k":
                self.fail = True
            elif key.char == "q":
                self.quit = True
            elif key.char == "h":
                self.help = True
            elif key.char == "r":
                self.infer = True
            elif key.char == "z":
                self.manual_reset = True
                print("manual reset!")
        except AttributeError as e:
            pass


if __name__ == '__main__':
    import time
    device = Keyboard()
    while True:
        print("finish", device.finish)
        print("start", device.start)
        print("discard", device.discard)
        print("quit", device.quit)
        print("continue", device.ctn)
        print("human", device.help)
        print("robot", device.infer)
        print("-----------------")
        time.sleep(1)
