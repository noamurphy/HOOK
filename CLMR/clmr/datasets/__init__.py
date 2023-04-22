import os
from .dataset import Dataset
from .audio import AUDIO
from .gtzan import GTZAN

def get_dataset(dataset):
    # # Location for datasets
    # if not os.path.exists(dataset_dir):
    # os.makedirs(dataset_dir)

    if dataset == "gtzan":
        d = GTZAN()
    # elif dataset == "audio":
    #     d = AUDIO(root=dataset_dir)        
    else:
        raise NotImplementedError("Dataset not implemented")
    return d