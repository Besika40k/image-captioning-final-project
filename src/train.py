import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import load_captions_df, get_dataloaders
from src.vocabulary import Vocabulary
from src.model import ImageCaptioningModel
from src.utils import save_checkpoint


def train_model(
    captions_path="data/caption_data/captions.txt",
    image_dir="data/caption_data/Images",
    checkpoint_dir="checkpoints",
    batch_size=64,
    num_epochs=10,
    embed_size=256,
    attention_dim=256,
    decoder_dim=512,
    dropout=0.5,
    learning_rate=3e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    freq_threshold=5,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    df = load_captions_df(captions_path)

    vocab = Vocabulary()
    vocab.build_vocabulary(df['caption'].tolist(), freq_threshold=freq_threshold)
    vocab.save(os.path.join(checkpoint_dir, 'vocab.pkl'))
    print(f"Vocabulary size: {len(vocab)}")

    train_loader, val_loader, _ = get_dataloaders(
        df, image_dir, vocab, batch_size=batch_size
    )

    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embed_size=embed_size,
        attention_dim=attention_dim,
        decoder_dim=decoder_dim,
        dropout=dropout,
        train_cnn=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, captions, lengths) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions[:, :-1])
            targets = captions[:, 1:]

            outputs = outputs.reshape(-1, outputs.size(2))
            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, captions, lengths in val_loader:
                images = images.to(device)
                captions = captions.to(device)

                outputs = model(images, captions[:, :-1])
                targets = captions[:, 1:]

                outputs = outputs.reshape(-1, outputs.size(2))
                targets = targets.reshape(-1)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_checkpoint(
                model, optimizer, epoch, avg_val_loss,
                os.path.join(checkpoint_dir, 'model.pth')
            )
            print(f"Best model saved (val_loss: {avg_val_loss:.4f})")


if __name__ == '__main__':
    train_model()
