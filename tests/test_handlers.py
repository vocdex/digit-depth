import unittest
import torch
from PIL import Image
from digit_depth.handlers import image

class Handler(unittest.TestCase):
    """Test for various data handlers"""
    def test_tensor_to_PIL(self):
        instance = image.ImageHandler(Image.open("/home/shuk/digit-depth/images/0001.png"), "RGB")
        tensor = torch.randn(1, 3, 224, 224)
        pil_image = instance.tensor_to_PIL()
        self.assertEqual(pil_image.size, (224, 224))


if __name__ == '__main__':
    unittest.main()
