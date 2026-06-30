import torch
from PIL import Image
import numpy as np
import os
from src.vocabulary import Vocabulary
from src.dataset import get_transform

def test_edge_cases():
    print("--- Testing Edge Cases ---")
    
    # 1. Grayscale Image Test
    print("\\n1. Grayscale Image Test:")
    # Create a dummy grayscale image
    gray_img = Image.fromarray(np.random.randint(0, 255, (224, 224), dtype=np.uint8), mode='L')
    gray_img.save('dummy_gray.jpg')
    
    try:
        transform = get_transform(train=False)
        img = Image.open('dummy_gray.jpg').convert('RGB')
        tensor = transform(img)
        print(f"Success! Grayscale image converted and transformed. Output shape: {tensor.shape}")
    except Exception as e:
        print(f"Failed on grayscale image: {e}")
    finally:
        if os.path.exists('dummy_gray.jpg'):
            os.remove('dummy_gray.jpg')

    # 2. Empty Caption Denumericalization Test
    print("\\n2. Empty Caption Test:")
    vocab = Vocabulary()
    vocab.word2idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3, 'hello': 4}
    vocab.idx2word = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>', 4: 'hello'}
    
    # Empty tensor
    empty_tensor = torch.tensor([])
    try:
        result = vocab.denumericalize(empty_tensor)
        print(f"Success! Empty tensor denumericalized to: '{result}' (Type: {type(result)})")
    except Exception as e:
        print(f"Failed on empty tensor: {e}")
        
    # Tensor with only start/end
    start_end_tensor = torch.tensor([1, 2])
    try:
        result = vocab.denumericalize(start_end_tensor)
        print(f"Success! Start/end tensor denumericalized to: '{result}'")
    except Exception as e:
        print(f"Failed on start/end tensor: {e}")

if __name__ == '__main__':
    test_edge_cases()
