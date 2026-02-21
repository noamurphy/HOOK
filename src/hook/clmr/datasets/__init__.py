# Derived from CLMR: https://github.com/Spijkervet/CLMR (Apache-2.0).
# Modifications for HOOK are licensed under Apache-2.0.
from hook.clmr.data import ContrastiveDataset
from .gtzan import GTZAN

def get_dataset(dataset, contrastive=False):
    if dataset == "gtzan":
        base_dataset = GTZAN()
        if contrastive:
            # Lazy import: only required when building the contrastive dataset pipeline.
            from torchaudio_augmentations import ComposeMany, RandomResizedCrop

            train_transform = [RandomResizedCrop(n_samples=59049)]
            num_augmented_samples = 1
            return ContrastiveDataset(
                base_dataset,
                input_shape=(1, 59049),
                transform=ComposeMany(train_transform, num_augmented_samples),
            )
        return base_dataset
    else:
        raise NotImplementedError("Dataset not implemented")
