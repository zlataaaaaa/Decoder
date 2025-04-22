# predict.py

import torch
from torch.utils.data import DataLoader
import pandas as pd
from model import CRNN
from dataset import MorseDataset, collate_fn
from morse_alphabet import IDX_TO_CHAR
import os
from pyctcdecode import BeamSearchDecoderCTC, Alphabet
import numpy as np
import re

# === Конфиг ===
DATA_DIR = 'data/audio/morse_dataset'
TEST_CSV = 'data/test.csv'
MODEL_PATH = 'checkpoints/model2.pt'
BATCH_SIZE = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


vocab = IDX_TO_CHAR[1:]  # без <blank>
if " " not in vocab:
    vocab.append(" ")

alphabet = Alphabet([*vocab], "<blank>")  # явное указание CTC blank
decoder = BeamSearchDecoderCTC(alphabet)

# === Постобработка ===
def postprocess(text):
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'[^А-Я0-9 ]', '', text)
    return text.strip()

# === Данные ===
df = pd.read_csv(TEST_CSV)
test_ds = MorseDataset(df, DATA_DIR)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# === Модель ===
model = CRNN(num_classes=len(IDX_TO_CHAR)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === Инференс ===
all_messages = []
with torch.no_grad():
    for features, _, input_lengths, _ in test_dl:
        features = features.to(DEVICE)
        compressed_lengths = input_lengths // 8
        logits = model(features)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2).squeeze(1)  # (T, C)
        raw = decoder.decode(log_probs.cpu().numpy())
        message = postprocess(raw)
        all_messages.append(message)

# === Сохранение результата ===
df['message'] = all_messages
df.to_csv('submission.csv', index=False)
print("✅ submission.csv готов к отправке с pyctcdecode + постобработка!")
