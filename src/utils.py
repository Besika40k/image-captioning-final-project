import torch

from src.model import ImageCaptioningModel


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)


def load_checkpoint(checkpoint_path, vocab_size=None, device='cpu'):
    model = ImageCaptioningModel(vocab_size=vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model
