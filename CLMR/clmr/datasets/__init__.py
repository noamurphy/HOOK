import os
from clmr.data import ContrastiveDataset
from torchaudio_augmentations import(ComposeMany, RandomResizedCrop)
from .dataset import Dataset
from .audio import AUDIO
from .gtzan import GTZAN

def get_dataset(dataset):
    # # Location for datasets
    # if not os.path.exists(dataset_dir):
    # os.makedirs(dataset_dir)

    train_transform = [RandomResizedCrop(n_samples=59049)]
    num_augmented_samples = 1
    if dataset == "gtzan":
        d = GTZAN()
        contrastive_dataset = ContrastiveDataset(
        d,
        input_shape=(1, 59049),
        transform=ComposeMany(train_transform, num_augmented_samples)
    )
    # elif dataset == "audio":
    #     d = AUDIO(root=dataset_dir)        
    else:
        raise NotImplementedError("Dataset not implemented")
    return d