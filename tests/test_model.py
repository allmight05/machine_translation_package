import torch
from machine_translation.model import Encoder, Decoder, Seq2Seq

def test_model_init():
    device = torch.device("cpu")
    enc = Encoder(1000, 256, 512, 2, 0.5, device)
    dec = Decoder(1000, 256, 512, 2, 0.5)
    model = Seq2Seq(enc, dec, device)
    assert model.encoder.hid_dim == model.decoder.hid_dim