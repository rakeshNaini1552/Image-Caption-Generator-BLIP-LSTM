---
title: Image Caption Generator
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Image Caption Generator

A full-stack image captioning application that generates natural language descriptions of images using two models — a pretrained BLIP model and a custom-trained LSTM+CNN model built from scratch.

## Live Demo
- **Frontend**: https://image-caption-generator-blip-lstm.vercel.app/
- **Backend API**: https://rakeshn591-image-caption-generator.hf.space/docs

## Stack
- **Backend**: FastAPI + Python
- **ML Models**: BLIP (Salesforce) + Custom LSTM+CNN
- **Frontend**: React + Vite
- **Deployment**: Hugging Face Spaces (backend) + Vercel (frontend)

## Models

### BLIP (Production)
Pretrained model from Salesforce (`blip-image-captioning-base`) fine-tuned on 
hundreds of millions of image-text pairs. Generates accurate, descriptive captions 
out of the box with no additional training.

### LSTM + CNN (Conceptual)
Custom implementation built from scratch to understand how image captioning works:
- **CNN Encoder**: ResNet50 (pretrained, frozen) extracts a 2048-dim feature vector per image
- **LSTM Decoder**: Projects CNN features into embedding space, generates words one by one
- **Training**: Flickr8k dataset, teacher forcing, CrossEntropyLoss
- **Inference**: Manual (h, c) state passing between timesteps

## Project Structure
```
image-captioner/
├── app/
│   ├── main.py          # FastAPI app, lifespan, /caption endpoint
│   └── model.py         # BLIP load and inference
├── lstm_captioner/
│   ├── dataset.py       # Flickr8k loader, vocabulary builder, FlickrDataset
│   ├── model.py         # CNNEncoder + LSTMDecoder + CaptionModel
│   ├── train.py         # Training loop
│   └── inference.py     # Caption generation from PIL image
├── frontend/
│   └── src/
│       ├── App.jsx           # Main component, state, API call
│       └── components/
│           ├── ImageUploader.jsx  # Drag and drop file upload
│           └── Caption.jsx        # Caption display
├── Dockerfile
├── requirements.txt
└── README.md
```

## Running Locally

### Backend
```bash
git clone https://github.com/rakeshNaini1552/Image-Caption-Generator-BLIP-LSTM.git
cd Image-Caption-Generator-BLIP-LSTM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
API runs at `http://localhost:8000`

### Frontend
```bash
cd frontend
npm install
npm run dev
```
UI runs at `http://localhost:5173`

## Training the LSTM Model

### Requirements
- Flickr8k dataset — place images at `lstm_captioner/image-dataset/Images/`
- Captions file at `lstm_captioner/image-dataset/captions.txt`

### Steps
```bash
cd lstm_captioner
python train.py
```

Training config (editable in `train.py`):
- `embed_dim`: 256
- `hidden_dim`: 512
- `batch_size`: 32
- `num_epochs`: 10
- `learning_rate`: 3e-4
- Device: MPS (Apple Silicon) → CUDA → CPU

Training saves weights to `lstm_captioner/trained_model.pth`.

### Running LSTM Inference
```bash
python lstm_captioner/inference.py path/to/image.jpg
```

## API

### POST /caption
Accepts a multipart form with:
- `file` — image file (jpg, png, etc.)
- `model_type` — `"blip"` (default) or `"lstm"`

Returns:
```json
{
  "caption": "a dog running through a field",
  "model": "blip"
}
```

## Notes
- LSTM is trained on 500 samples by default (for pipeline verification). 
  Remove the `Subset` line in `train.py` and train on full dataset for better results.
- BLIP is downloaded from HuggingFace Hub on first startup (~990MB).
- LSTM model and dataset are not included in the Docker image.
