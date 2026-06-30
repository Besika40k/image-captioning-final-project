import os
import torch
from src.vocabulary import Vocabulary

try:
    from src.utils import load_checkpoint
except ImportError:
    print("utils.py or load_checkpoint not implemented yet. This is expected!")
    load_checkpoint = None

def main():
    vocab_path = "checkpoints/vocab.pkl"
    checkpoint_path = "checkpoints/model.pth"
    
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
        print("Created checkpoints/ directory.")
        
    if not os.path.exists(vocab_path):
        print(f"Vocab not found at {vocab_path}. Please build and save it first.")
        return
        
    vocab = Vocabulary.load(vocab_path)
    print(f"Loaded vocab of size: {len(vocab)}")
    
    if load_checkpoint is None:
        print("Smoke test paused: waiting for Person B to implement load_checkpoint.")
        return
        
    try:
        model = load_checkpoint(checkpoint_path, vocab_size=len(vocab), device="cpu")
        model.eval()
        print("Model loaded successfully!")
        
        # Test generate interface (dummy tensor)
        dummy_image = torch.randn(1, 3, 224, 224)
        output = model.generate(dummy_image, vocab, max_len=20)
        print(f"Generated caption type: {type(output)}")
        print(f"Output: {output}")
        
    except Exception as e:
        print(f"Smoke test failed during model load/inference: {e}")

if __name__ == '__main__':
    main()
