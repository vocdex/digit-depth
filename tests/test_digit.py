import unittest

from digit_interface import Digit

from digit_depth import DigitSensor


class TestDigit(unittest.TestCase):
    def test_digit_sensor(self):
        fps = 30
        resolution = "QVGA"
        serial_num = "12345"
        digit_sensor = DigitSensor(fps, resolution, serial_num)
        digit = digit_sensor()
        self.assertIsInstance(digit, Digit)
