from os.path import join
from torchvision.transforms import Compose, ToTensor
from dataset import DatasetFromFolderEval, DatasetFromFolder
# from dataset_DBPN import DatasetFromFolderEval, DatasetFromFolder

def transform():
    return Compose([
        ToTensor(),
    ])

def get_training_set(data_dir, hr, bit_gap, patch_size, data_augmentation):
    hr_dir = join(data_dir, hr)
    return DatasetFromFolder(hr_dir,patch_size, bit_gap, data_augmentation,
                             transform=transform())

def get_eval_set(lr_dir, bit_gap):
    return DatasetFromFolderEval(lr_dir, bit_gap,
                             transform=transform())

