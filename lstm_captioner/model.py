import logging
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

logger = logging.getLogger(__name__)

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        logger.info("Loading pretrained ResNet50 encoder")
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False
        logger.info("CNNEncoder ready — weights frozen")

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        logger.debug("CNNEncoder output shape: %s", features.shape)
        return features

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        super().__init__()
        self.feature_proj = nn.Linear(2048, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        logger.info("LSTMDecoder ready — vocab_size=%d, embedding_dim=%d, hidden_dim=%d", vocab_size, embedding_dim, hidden_dim)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        features = self.feature_proj(features).unsqueeze(1)
        lstm_input = torch.cat((features, embeddings), dim=1)
        lstm_output, _ = self.lstm(lstm_input)
        output = self.fc(lstm_output)[:, 1:, :]
        logger.debug("LSTMDecoder output shape: %s", output.shape)
        return output

class CaptionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        super().__init__()
        logger.info("Initializing CaptionModel")
        self.encoder = CNNEncoder()
        self.decoder = LSTMDecoder(vocab_size, embedding_dim, hidden_dim)
        logger.info("CaptionModel ready")

    def forward(self, images, captions):
        features = self.encoder(images)
        return self.decoder(features, captions)