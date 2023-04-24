import torch
import numpy as np
import pickle
from collections import OrderedDict
from torch.utils.data import DataLoader
from clmr.datasets import get_dataset
from clmr.models import SampleCNN
from clmr.utils.checkpoint import load_encoder_checkpoint

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
    drop_last=True,
    shuffle=False,
    )
    print("Dataloader instantialized successfully")

    # Instantiate SampleCNN model
    print("Loading model...")
    strides = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    model = SampleCNN(strides)

    # Load pre-trained checkpoint
    checkpoint_path = "clmr_checkpoints/clmr_checkpoint_10000/clmr_checkpoint_10000.pt"
    state_dict = load_encoder_checkpoint(checkpoint_path)    
    
    # Load checkpoint into model
    model.load_state_dict(state_dict)
    print("Model loaded successfully")
    
    # Run model to get representations
    print("Evaluating representations...")
    model.eval()
    all_representations = []
    with torch.no_grad():
      for data, target in dataloader:
          # pass the data through the model to get representations
          representations = model(data)
          all_representations.append(representations)

    all_representations = torch.cat(all_representations)
    print("Representations evaluated: saving to file...")
    # Save representation object
    with open('representations.pkl', 'wb') as f:
      pickle.dump(all_representations, f)
    print("Representations saved to 'representations.pkl'")
      
if __name__ == '__main__':
    main()