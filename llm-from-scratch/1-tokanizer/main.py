import re


def read_raw_text(filename):
    with open(filename, "r", encoding="utf-8") as f:
        raw_text = f.read()

    return raw_text


def split_text(raw_text):
    result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    result = [item.strip() for item in result if item.strip()]
    return result


if __name__ == "__main__":
    text = read_raw_text("the-verdict.txt")

    print(split_text(text)[:20])