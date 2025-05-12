import torch
import spacy

spacy_de = spacy.load("de_core_news_sm")

def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    """Translate a German sentence to English."""
    tokens = [tok.text for tok in spacy_de.tokenizer(sentence)][::-1]
    numericalized = [src_vocab["<SOS>"]] + [src_vocab.get(token, src_vocab["<UNK>"]) for token in tokens] + [src_vocab["<EOS>"]]
    tensor = torch.LongTensor(numericalized).unsqueeze(0).to(device)
    with torch.no_grad():
        hidden, cell = model.encoder(tensor)
    trg_indexes = [trg_vocab["<SOS>"]]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab["<EOS>"]:
            break
    inv_trg_vocab = {i: w for w, i in trg_vocab.items()}
    translated_tokens = [inv_trg_vocab[idx] for idx in trg_indexes]
    return translated_tokens[1:-1]