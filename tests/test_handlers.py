import unittest
import torch
from PIL import Image
from torchvision import transforms
from src.handlers.image import ImageHandler

class Handler(unittest.TestCase):
    def test_tensor_to_PIL(self):
        instance=ImageHandler(Image.open("tests/test_handlers.py"), "RGB")
        tensor = torch.randn(1, 3, 224, 224)
        pil_image = instance.tensor_to_PIL()
        self.assertEqual(pil_image.size, (224, 224))
if __name__ == '__main__':
    unittest.main()
