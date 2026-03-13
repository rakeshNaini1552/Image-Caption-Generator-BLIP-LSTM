---
title: Image Caption Generator
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Image Caption Generator

A full-stack image captioning app using BLIP and LSTM models.

## Stack
- **Backend**: FastAPI + BLIP (Salesforce) + custom LSTM+CNN
- **Frontend**: React + Vite
- **Deployment**: Hugging Face Spaces (backend) + Vercel (frontend)

## Models
- **BLIP**: Pretrained model, accurate captions out of the box
- **LSTM**: Custom trained on Flickr8k, conceptual implementation

## Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```