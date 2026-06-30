import re
import pickle
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.pad_idx = 0
        self.start_idx = 1
        self.end_idx = 2
        self.unk_idx = 3
        
        self.idx2word[self.pad_idx] = "<pad>"
        self.idx2word[self.start_idx] = "<start>"
        self.idx2word[self.end_idx] = "<end>"
        self.idx2word[self.unk_idx] = "<unk>"
        
        self.word2idx = {v: k for k, v in self.idx2word.items()}
        
    def __len__(self):
        return len(self.word2idx)
        
    @staticmethod
    def tokenize(text):
        """Simple regex tokenizer, explicitly avoiding nltk."""
        return re.findall(r"\w+", text.lower())
        
    def build_vocabulary(self, caption_list, freq_threshold=5):
        """Counts token frequencies and assigns indices to tokens meeting the threshold."""
        frequencies = Counter()
        for caption in caption_list:
            tokens = self.tokenize(caption)
            frequencies.update(tokens)
            
        idx = 4
        for word, count in frequencies.items():
            if count >= freq_threshold:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                
    def numericalize(self, text):
        """Tokenizes text and maps known tokens to indices, unknown tokens to <unk>."""
        tokenized_text = self.tokenize(text)
        return [
            self.word2idx.get(token, self.unk_idx)
            for token in tokenized_text
        ]
        
    def denumericalize(self, indices):
        """Reverses numericalization, stopping at <end> and skipping special tokens."""
        words = []
        for idx in indices:
            if hasattr(idx, 'item'):
                idx = idx.item()
                
            if idx == self.end_idx:
                break
            if idx not in [self.pad_idx, self.start_idx, self.unk_idx]:
                words.append(self.idx2word.get(idx, "<unk>"))
        return " ".join(words)
        
    def save(self, path):
        """Saves vocabulary via pickle."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path):
        """Loads vocabulary from pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)
