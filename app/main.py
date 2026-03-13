import io
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.model import generate_caption, load_model
from lstm_captioner.dataset import build_vocabulary, load_captions
from lstm_captioner.model import CaptionModel
from lstm_captioner.inference import generate_caption as lstm_generate_caption

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

_LSTM_DIR = Path(__file__).parent.parent / "lstm_captioner"
_CAPTIONS_FILE = _LSTM_DIR / "image-dataset" / "captions.txt"
_LSTM_CHECKPOINT = _LSTM_DIR / "trained_model.pth"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: load BLIP model
    model, processor = load_model(device)
    app.state.model = model
    app.state.processor = processor
    app.state.device = device

    # startup: load LSTM if available
    try:
        dict_captions = load_captions(str(_CAPTIONS_FILE))
        vocab, word2idx, idx2word = build_vocabulary(dict_captions)
        lstm_model = CaptionModel(vocab_size=len(vocab)).to(device)
        lstm_model.load_state_dict(torch.load(_LSTM_CHECKPOINT, map_location=device))
        lstm_model.eval()
        app.state.lstm_model = lstm_model
        app.state.word2idx = word2idx
        app.state.idx2word = idx2word
        app.state.lstm_available = True
    except FileNotFoundError:
        app.state.lstm_available = False

    yield
    # shutdown: cleanup here
    del model, processor


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/caption")
async def caption(request: Request, file: UploadFile = File(...), model_type: str = "blip"):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    state = request.app.state

    if model_type == "lstm":
        if not state.lstm_available:
            return {"caption": "LSTM model not available in production.", "model": "lstm"}
        result = lstm_generate_caption(
            state.lstm_model,
            image,
            state.word2idx,
            state.idx2word,
            state.device,
        )
    else:
        result = generate_caption(image, state.model, state.processor, state.device)

    return {"caption": result, "model": model_type}