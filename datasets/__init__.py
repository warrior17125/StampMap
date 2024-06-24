import torch.utils.data
import torchvision

from datasets.dataset import build_lane_data

def build_dataset(image_set, args):
    return build_lane_data(image_set, args)