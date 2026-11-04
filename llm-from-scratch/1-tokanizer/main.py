from tokenizer import read_raw_text, tokanize, create_vocab, SimpleTokenizer
from dataloader import create_dataloader_v1

if __name__ == "__main__":
    raw_text = read_raw_text("the-verdict.txt")

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=1,
        max_length=4,
        stride=1,
        shuffle=False
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
