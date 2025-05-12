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

# Train for N epochs and save best model
best_val_loss = float('inf')
for epoch in range(1, num_epochs+1):
    train_loss = train(model, train_loader, optimizer, criterion, clip=1, device=device)
    val_loss = evaluate(model, val_loader, criterion, device=device)

    print(f"Epoch {epoch}: Train Loss={train_loss:.3f}, Val Loss={val_loss:.3f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'epoch': epoch,
             'val_loss': val_loss},
            "machine_translation_best.pt"
        )
        print(f"  → Saved best model (epoch {epoch}, val_loss {val_loss:.3f})")

```

### Translating a Sentence
```python
from machine_translation import translate_sentence, get_vocabularies, Encoder, Decoder, Seq2Seq
import torch

# Reconstruct vocab sizes
src_vocab, trg_vocab = get_vocabularies()  # your helper to load/save vocabs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc = Encoder(len(src_vocab), 256, 512, 2, 0.5, device)
dec = Decoder(len(trg_vocab), 256, 512, 2, 0.5)
model = Seq2Seq(enc, dec, device).to(device)

# Load saved checkpoint
checkpoint = torch.load("machine_translation_best.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

sentence = "Ein kleines Mädchen spielt im Park."
translation = translate_sentence(sentence, src_vocab, trg_vocab, model, device)
print(" ".join(translation))
```

## License
MIT License
