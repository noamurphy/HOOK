import torchaudio
from datasets import load_dataset
from torch.utils.data import Dataset


class GTZAN(Dataset):

    def __init__(self, split="train"):
        self.dataset = load_dataset("marsyas/gtzan", split=split)
        self.labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
                       'jazz', 'metal', 'pop', 'reggae', 'rock']

        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}
        self.n_classes = len(self.labels)

    def __getitem__(self, idx):
        file_path = self.dataset[idx]['file']
        audio, sr = torchaudio.load(file_path)
        label = self.label2idx[self.labels[self.dataset[idx]['genre']]]

        return audio, label

    def __len__(self):
        return len(self.dataset)