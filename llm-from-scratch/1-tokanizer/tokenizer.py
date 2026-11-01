import re


def read_raw_text(filename):
    with open(filename, "r", encoding="utf-8") as f:
        raw_text = f.read()

    return raw_text


def tokanize(raw_text):
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    tokens = [item.strip() for item in tokens if item.strip()]
    return tokens


def create_vocab(tokens):
    all_tokens = sorted(set(tokens))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab_size = len(all_tokens)

    vocab = {token:i for i,token in enumerate(all_tokens)}

    return [vocab_size, vocab]


class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = tokanize(text)
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[id] for id in ids])

        # removes whitespaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
