import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms

def load_captions_df(captions_csv_path: str) -> pd.DataFrame:
    """Thin wrapper around pd.read_csv to load the raw image,caption dataframe."""
    return pd.read_csv(captions_csv_path)

def split_by_image(df: pd.DataFrame, val_frac=0.1, test_frac=0.1, seed=42):
    """
    Splits dataframe by unique images to prevent data leakage.
    All 5 captions for a given image will remain in the exact same split.
    """
    unique_images = df['image'].unique().tolist()
    
    np.random.seed(seed)
    np.random.shuffle(unique_images)
    
    val_size = int(len(unique_images) * val_frac)
    test_size = int(len(unique_images) * test_frac)
    train_size = len(unique_images) - val_size - test_size
    
    train_images = set(unique_images[:train_size])
    val_images = set(unique_images[train_size:train_size + val_size])
    test_images = set(unique_images[train_size + val_size:])
    
    # Crucial assertion to ensure no leakage between splits
    assert len(train_images.intersection(val_images)) == 0, "Leakage between train and val splits!"
    assert len(train_images.intersection(test_images)) == 0, "Leakage between train and test splits!"
    
    train_df = df[df['image'].isin(train_images)].copy().reset_index(drop=True)
    val_df = df[df['image'].isin(val_images)].copy().reset_index(drop=True)
    test_df = df[df['image'].isin(test_images)].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df

class CaptionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, vocab, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image']
        caption = row['caption']
        
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            
        numericalized_caption = [self.vocab.start_idx] + \
                                self.vocab.numericalize(caption) + \
                                [self.vocab.end_idx]
                                
        return image, torch.tensor(numericalized_caption)

def get_transform(train: bool) -> transforms.Compose:
    """Returns ImageNet normalization transforms, with optional training augmentation."""
    transform_list = [
        transforms.Resize((224, 224)),
    ]
    
    if train:
        # Mild augmentation as requested
        transform_list.append(transforms.RandomHorizontalFlip())
        
    transform_list.extend([
        transforms.ToTensor(),
        # Mandatory ImageNet stats normalization for pretrained ResNet50
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)

class CollateWrapper:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        images, captions = zip(*batch)
        
        images = torch.stack(images, dim=0)
        lengths = torch.tensor([len(cap) for cap in captions])
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        
        return images, captions, lengths


def get_dataloaders(df, image_dir, vocab, batch_size=64, num_workers=2, val_frac=0.1, test_frac=0.1, seed=42):
    """Convenience function wiring together the split, datasets, and DataLoaders."""
    train_df, val_df, test_df = split_by_image(df, val_frac=val_frac, test_frac=test_frac, seed=seed)
    
    train_dataset = CaptionDataset(train_df, image_dir, vocab, transform=get_transform(train=True))
    val_dataset = CaptionDataset(val_df, image_dir, vocab, transform=get_transform(train=False))
    test_dataset = CaptionDataset(test_df, image_dir, vocab, transform=get_transform(train=False))
    
    collate_wrapper = CollateWrapper(pad_idx=vocab.pad_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, collate_fn=collate_wrapper)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_wrapper)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, collate_fn=collate_wrapper)
                             
    return train_loader, val_loader, test_loader