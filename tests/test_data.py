from machine_translation.data import get_data_loaders

def test_data_loaders():
    train_loader, val_loader, test_loader, src_vocab, trg_vocab = get_data_loaders(batch_size=2)
    assert len(train_loader) > 0
    assert len(src_vocab) > 0
    assert len(trg_vocab) > 0