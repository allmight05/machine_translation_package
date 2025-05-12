# Machine Translation Package

A German-to-English machine translation system built from scratch using PyTorch.

## Installation

1. Install the package via pip:
   ```bash
   pip install machine-translation
   ```
2. Install spaCy models:
   ```bash
   python -m spacy download de_core_news_sm
   python -m spacy download en_core_web_sm
   ```

## Usage

### Training a Model
```python
from machine_translation import get_data_loaders, Encoder, Decoder, Seq2Seq, train
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, _, src_vocab, trg_vocab = get_data_loaders()
enc = Encoder(len(src_vocab), 256, 512, 2, 0.5, device)
dec = Decoder(len(trg_vocab), 256, 512, 2, 0.5)
model = Seq2Seq(enc, dec, device).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab["<PAD>"])
train(model, train_loader, optimizer, criterion, clip=1, device=device)
```

### Translating a Sentence
```python
from machine_translation import translate_sentence, Encoder, Decoder, Seq2Seq
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = Encoder(len(src_vocab), 256, 512, 2, 0.5, device)
dec = Decoder(len(trg_vocab), 256, 512, 2, 0.5)
model = Seq2Seq(enc, dec, device).to(device)
model.load_state_dict(torch.load("Seq model.pt", weights_only=True))
sentence = "Ein kleines Mädchen spielt im Park."
translation = translate_sentence(sentence, src_vocab, trg_vocab, model, device)
print(" ".join(translation))
```

## License
MIT License
