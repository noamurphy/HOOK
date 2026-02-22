# Derived from CLMR: https://github.com/Spijkervet/CLMR (Apache-2.0).
# Modifications for HOOK are licensed under Apache-2.0.
import os
import time
import torch
import pickle
from torch.utils.data import DataLoader
from hook.clmr.datasets import get_dataset
from hook.clmr.models import SampleCNN
from hook.clmr.utils.checkpoint import load_encoder_checkpoint

def main():
    # Load dataset
    print("Loading dataset...")
    dataset = get_dataset("gtzan")
    print("Dataset loaded successfully")

    # Initialize dataloader
    dataloader = DataLoader(
    dataset,
    batch_size=48,
    num_workers=0,
    drop_last=False,
    shuffle=False,
    )
    print("Dataloader instantialized successfully")

    # Instantiate SampleCNN model
    print("Loading model...")
    strides = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    model = SampleCNN(strides)

    # Load pre-trained checkpoint
    checkpoint_path = "artifacts/checkpoints/clmr_checkpoint_10000/clmr_checkpoint_10000.pt"
    state_dict = load_encoder_checkpoint(checkpoint_path)    
    
    # Load checkpoint into model
    model.load_state_dict(state_dict)
    print("Model loaded successfully")
    
    # Run model to get representations
    print("Evaluating representations...")
    model.eval()
    all_representations = []
    total_batches = len(dataloader)
    start_time = time.time()
    with torch.no_grad():
      for batch_idx, (data, target) in enumerate(dataloader, start=1):
          # pass the data through the model to get representations
          representations = model(data)
          all_representations.append(representations)
          if batch_idx % 10 == 0 or batch_idx == total_batches:
              elapsed = time.time() - start_time
              print(
                  f"Processed batch {batch_idx}/{total_batches} "
                  f"({elapsed:.1f}s elapsed)"
              )

    all_representations = torch.cat(all_representations)
    print(f"Total embeddings produced: {all_representations.shape[0]}")
    print("Representations evaluated: saving to file...")
    # Save representation object
    os.makedirs("artifacts/embeddings", exist_ok=True)
    with open('artifacts/embeddings/representations.pkl', 'wb') as f:
      pickle.dump(all_representations, f)
    print("Representations saved to 'artifacts/embeddings/representations.pkl'")
      
if __name__ == '__main__':
    main()
