from tokenizer import read_raw_text, tokanize, create_vocab, SimpleTokenizerV1

if __name__ == "__main__":
    text = read_raw_text("the-verdict.txt")
    tokens = tokanize(text)

    _, vocab = create_vocab(tokens)

    tokenizer = SimpleTokenizerV1(vocab)

    text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""

    ids = tokenizer.encode(text)

    print(ids)
