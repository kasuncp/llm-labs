from tokenizer import read_raw_text, tokanize, create_vocab, SimpleTokenizer

if __name__ == "__main__":
    text = read_raw_text("the-verdict.txt")
    tokens = tokanize(text)

    _, vocab = create_vocab(tokens)

    tokenizer = SimpleTokenizer(vocab)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)


    ids = tokenizer.encode(text)
    
    print(ids)
    print(tokenizer.decode(ids))
