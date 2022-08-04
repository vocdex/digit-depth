"""
  Data loader for the color-normal datasets
"""
from src.dataio.digit_dataset import DigitRealImageAnnotDataset
import torchvision.transforms as transforms
from torch.utils.data import  DataLoader


def data_loader(dir_dataset,params):
    """A data loader for the color-normal datasets
       Args:
           params: a dict of parameters
    """
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = DigitRealImageAnnotDataset(dir_dataset=dir_dataset,
                                         transform=transform, annot_flag=params.annot_flag)
    dataloader = DataLoader(dataset, batch_size=params.batch_size,
                            shuffle=params.shuffle, num_workers=params.num_workers)

    return dataloader, dataset

