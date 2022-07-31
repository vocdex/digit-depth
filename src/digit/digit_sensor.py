from digit_interface import Digit
import time


class DigitSensor():
    def __init__(self,fps: int,resolution: str,serial_num: str):
        self.fps=fps
        self.resolution=resolution
        self.serial_num=serial_num

    def __call__(self, *args, **kwargs):
        """Calls the digit sensor."""
        digit = self.setup_digit(self.serial_num, self.fps)
        return digit

    def setup_digit(self,serial_num: str, fps: int):
        """Sets up the digit sensor and returns it."""
        digit = Digit(self.serial_num)
        digit.connect()

        rgb_list = [(15, 0, 0), (0, 15, 0), (0, 0, 15)]

        for rgb in rgb_list:
            digit.set_intensity_rgb(*rgb)
            time.sleep(1)
        digit.set_intensity(15)
        fps=Digit.STREAMS["QVGA"]["fps"][str(self.fps)+"fps"]
        digit.set_fps(fps)
        return digit

