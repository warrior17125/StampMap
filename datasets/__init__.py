import torch.utils.data
import torchvision

from datasets.dataset import build_lanedata

def build_dataset(image_set, args):
    return build_lanedata(image_set, args)