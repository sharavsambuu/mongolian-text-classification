
def wordtoken_to_id(model, word):
    token_id = model.wv.vocab[word].index
    return token_id