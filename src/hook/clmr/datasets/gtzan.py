# Derived from CLMR: https://github.com/Spijkervet/CLMR (Apache-2.0).
# Modifications for HOOK are licensed under Apache-2.0.
import os
from io import BytesIO
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from torch.utils.data import Dataset


class GTZAN(Dataset):

    def __init__(self, split="train", dataset_id=None, target_num_samples=59049):
        dataset_ids = self._candidate_dataset_ids(dataset_id)
        self.dataset = None
        self.dataset_id = None
        self.target_num_samples = target_num_samples
        errors = []
        for ds_id in dataset_ids:
            try:
                self.dataset = load_dataset(ds_id, split=split)
                self.dataset_id = ds_id
                break
            except Exception as exc:
                errors.append((ds_id, str(exc)))
        if self.dataset is None:
            details = "\n".join([f"- {ds_id}: {msg}" for ds_id, msg in errors])
            raise RuntimeError(
                "Failed to load any GTZAN dataset candidate.\n"
                "Set HOOK_GTZAN_DATASET_ID to a working dataset id.\n"
                f"Tried:\n{details}"
            )
        # Avoid automatic audio decoding through torchcodec in HF datasets.
        if "audio" in self.dataset.column_names:
            self.dataset = self.dataset.cast_column("audio", Audio(decode=False))
        self.labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
                       'jazz', 'metal', 'pop', 'reggae', 'rock']

        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}
        self.n_classes = len(self.labels)

    def _fix_length(self, audio: torch.Tensor) -> torch.Tensor:
        """Deterministically center-crop or right-pad to the CLMR input length."""
        n = audio.size(-1)
        target = self.target_num_samples
        if n == target:
            return audio
        if n > target:
            start = (n - target) // 2
            return audio[:, start : start + target]

        padded = torch.zeros((audio.size(0), target), dtype=audio.dtype)
        padded[:, :n] = audio
        return padded

    @staticmethod
    def _candidate_dataset_ids(dataset_id=None):
        if dataset_id:
            return [dataset_id]
        env_id = os.environ.get("HOOK_GTZAN_DATASET_ID", "").strip()
        if env_id:
            return [env_id]
        return [
            "sanchit-gandhi/gtzan",
            "marsyas/gtzan",
            "ParitKansal/marsyas-gtzan",
        ]

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        audio_dict = sample.get("audio") if isinstance(sample, dict) else None

        file_path = None
        if "file" in sample and sample["file"]:
            file_path = sample["file"]
        elif isinstance(audio_dict, dict):
            file_path = audio_dict.get("path")

        if file_path and os.path.exists(file_path):
            audio, sr = sf.read(file_path, always_2d=True, dtype="float32")
        elif isinstance(audio_dict, dict) and audio_dict.get("bytes") is not None:
            audio, sr = sf.read(BytesIO(audio_dict["bytes"]), always_2d=True, dtype="float32")
        else:
            raise ValueError(
                "GTZAN sample has neither a readable local path nor in-memory audio bytes."
            )

        audio = torch.from_numpy(audio.T)
        if audio.dim() == 2 and audio.size(0) != 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = self._fix_length(audio)
        genre = sample["genre"]
        if isinstance(genre, int):
            label = genre
        else:
            label = self.label2idx[genre]

        return audio, label

    def __len__(self):
        return len(self.dataset)
