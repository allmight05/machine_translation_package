import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import Counter
import spacy

# Load tokenizers
spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

def tokenize_de(text):
    """Tokenize German text and reverse the order."""
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    """Tokenize English text."""
    return [tok.text for tok in spacy_en.tokenizer(text)]

def build_vocab(tokenized_data, min_freq=2):
    """Build vocabulary from tokenized data."""
    vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    idx = 4
    token_counter = Counter(token for tokens in tokenized_data for token in tokens)
    for token, freq in token_counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = idx
            idx += 1
    return vocab

def numericalize(tokens, vocab):
    """Convert tokens to numerical indices."""
    return [vocab["<SOS>"]] + [vocab.get(token, vocab["<UNK>"]) for token in tokens] + [vocab["<EOS>"]]

def collate_fn(batch):
    """Pad sequences in a batch."""
    src = [torch.tensor(item["SRC"]) for item in batch]
    trg = [torch.tensor(item["TRG"]) for item in batch]
    src_padded = pad_sequence(src, batch_first=True, padding_value=0)
    trg_padded = pad_sequence(trg, batch_first=True, padding_value=0)
    return src_padded, trg_padded

def get_data_loaders(batch_size=128):
    """Load and prepare Multi30k dataset with DataLoaders."""
    dataset = load_dataset("bentrevett/multi30k")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return {
            "SRC": [tokenize_de(example) for example in examples["de"]],
            "TRG": [tokenize_en(example) for example in examples["en"]],
        }
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Build vocabularies
    src_vocab = build_vocab(tokenized_dataset["train"]["SRC"])
    trg_vocab = build_vocab(tokenized_dataset["train"]["TRG"])
    
    # Numericalize dataset
    def numericalize_function(examples):
        return {
            "SRC": [numericalize(tokens, src_vocab) for tokens in examples["SRC"]],
            "TRG": [numericalize(tokens, trg_vocab) for tokens in examples["TRG"]]
        }
    numericalized_dataset = tokenized_dataset.map(numericalize_function, batched=True)
    
    # Create DataLoaders
    train_loader = DataLoader(numericalized_dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(numericalized_dataset["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(numericalized_dataset["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, src_vocab, trg_vocab