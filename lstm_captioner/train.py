import logging
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from functools import partial
from lstm_captioner.dataset import load_captions, build_vocabulary, FlickrDataset
from lstm_captioner.model import CaptionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def collate_fn(batch, pad_idx):
    images, captions = zip(*batch)
    images = torch.stack(images)
    captions = torch.nn.utils.rnn.pad_sequence(
        captions, batch_first=True, padding_value=pad_idx
    )
    return images, captions


def train(model, dataloader, optimizer, criterion, device):
    logger.info("Starting training epoch")
    model.train()
    total_loss = 0.0
    total_batches = len(dataloader)
    for batch_idx, (images, captions) in enumerate(dataloader):
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()
        outputs = model(images, captions[:, :-1])
        targets = captions[:, 1:]
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            logger.info("  Batch %d/%d — loss: %.4f", batch_idx + 1, total_batches, loss.item())
    avg_loss = total_loss / total_batches
    logger.info("Training epoch done — avg loss: %.4f", avg_loss)
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    logger.info("Starting evaluation")
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, captions in dataloader:
            images, captions = images.to(device), captions.to(device)
            outputs = model(images, captions[:, :-1])
            targets = captions[:, 1:]
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    logger.info("Evaluation done — avg loss: %.4f", avg_loss)
    return avg_loss


def main():
    image_dir = "image-dataset/Images"
    captions_file = "image-dataset/captions.txt"
    embed_dim = 256
    hidden_dim = 512
    batch_size = 32
    num_epochs = 10
    learning_rate = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    dict_captions = load_captions(captions_file)
    vocab, word2idx, idx2word = build_vocabulary(dict_captions)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = FlickrDataset(image_dir, captions_file, word2idx, transform)
    dataset = Subset(dataset, range(500))  # Use a smaller subset for quick testing
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_idx=word2idx["<PAD>"]),
    )
    logger.info("Dataloader ready — %d batches", len(dataloader))

    model = CaptionModel(len(vocab), embed_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])
    logger.info("Starting training for %d epochs", num_epochs)

    for epoch in range(num_epochs):
        loss = train(model, dataloader, optimizer, criterion, device)
        logger.info("Epoch %d/%d — Loss: %.4f", epoch + 1, num_epochs, loss)
    
    torch.save(model.state_dict(), "trained_model.pth")
    logger.info("Model saved at trained_model.pth")


if __name__ == "__main__":
    main()
