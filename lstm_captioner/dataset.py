import logging
import string
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

def load_captions(filepath):
    logger.info("Loading captions from %s", filepath)
    dict_captions = {}
    with open(filepath, 'r') as f:
        next(f)  # skip header row
        for line in f:
            image_filename, caption = line.strip().split(',', 1)
            if image_filename not in dict_captions:
                dict_captions[image_filename] = []
            dict_captions[image_filename].append(caption)
    _remove_punct = str.maketrans('', '', string.punctuation)
    for image_filename, captions in dict_captions.items():
        dict_captions[image_filename] = [caption.lower().translate(_remove_punct) for caption in captions]
    logger.info("Loaded captions for %d images", len(dict_captions))
    return dict_captions

def build_vocabulary(dict_captions):
    logger.info("Building vocabulary")
    words = set()
    for captions in dict_captions.values():
        for caption in captions:
            words.update(caption.split())
    vocab = ['<START>', '<END>', '<PAD>'] + sorted(words)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    logger.info("Vocabulary built — %d words", len(vocab))
    return vocab, word2idx, idx2word

class FlickrDataset(Dataset):
    def __init__(self, image_dir, captions_file, word2idx, transform=None):
        dict_captions = load_captions(captions_file)
        self.samples = [
            (filename, caption)
            for filename, captions in dict_captions.items()
            for caption in captions
        ]
        self.image_dir = image_dir
        self.word2idx = word2idx
        self.transform = transform
        logger.info("FlickrDataset ready — %d samples", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, caption = self.samples[idx]
        logger.debug("Loading image: %s", filename)
        image = Image.open(f"{self.image_dir}/{filename}").convert('RGB')
        if self.transform:
            image = self.transform(image)
        indices = [self.word2idx['<START>']] + [self.word2idx[w] for w in caption.split()] + [self.word2idx['<END>']]
        return image, torch.tensor(indices)