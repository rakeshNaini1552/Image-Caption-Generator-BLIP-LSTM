import logging
import sys
import torch
from PIL import Image
from torchvision import transforms
from lstm_captioner.dataset import load_captions, build_vocabulary
from lstm_captioner.model import CaptionModel

logger = logging.getLogger(__name__)


def generate_caption(model, image, word2idx, idx2word, device, max_len=20):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 1. Preprocess image (accepts a PIL Image)
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        # 2. Encode image with CNN
        features = model.encoder(image)

        # 3. Prime LSTM with projected image feature
        feat_proj = model.decoder.feature_proj(features).unsqueeze(1)
        _, (h, c) = model.decoder.lstm(feat_proj)

        # 4. Generate words one by one
        word_idx = torch.tensor([[word2idx['<START>']]]).to(device)
        words = []
        for _ in range(max_len):
            emb = model.decoder.embedding(word_idx)
            lstm_out, (h, c) = model.decoder.lstm(emb, (h, c))
            logits = model.decoder.fc(lstm_out.squeeze(1))
            predicted = logits.argmax(dim=-1).item()
            word = idx2word[predicted]
            # 5. Stop at <END> or max_len
            if word == '<END>':
                break
            words.append(word)
            word_idx = torch.tensor([[predicted]]).to(device)

    return ' '.join(words)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) < 2:
        logger.info("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Generating caption for image: {image_path}")
    captions_file = "image-dataset/captions.txt"
    model_path = "trained_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dict_captions = load_captions(captions_file)
    vocab, word2idx, idx2word = build_vocabulary(dict_captions)

    model = CaptionModel(len(vocab), embedding_dim=256, hidden_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    image = Image.open(image_path).convert('RGB')
    caption = generate_caption(model, image, word2idx, idx2word, device)
    logger.info("Generated caption: %s", caption)


if __name__ == "__main__":
    main()
