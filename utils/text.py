from typing import Generator, List, Optional

import torch

from torchtext.vocab import Vocab, build_vocab_from_iterator

PAD = "<pad>"
UNK = "<unk>"
EOS = "<eos>"
BOS = "<bos>"


def text_to_tensor(
    text: str, vocab: Vocab, max_len: int, eos: bool = True, bos: bool = True
) -> torch.Tensor:
    tokenized_text = tokenize_sentence(text)
    if eos:
        tokenized_text = tokenized_text + [EOS]
    if bos:
        tokenized_text = [BOS] + tokenized_text

    tensor = torch.zeros(max_len)
    for i in range(max_len):
        if i < len(tokenized_text):
            if tokenized_text[i] in vocab:
                tensor[i] = vocab[tokenized_text[i]]
            else:
                tensor[i] = vocab[UNK]
        else:
            tensor[i] = vocab[PAD]
    return tensor.to(torch.long)


def tensor_to_text(tensor: torch.Tensor, vocab: Vocab) -> str:
    text = []
    for i in range(tensor.size(0)):
        text.append(vocab.lookup_token(tensor[i]))
    return " ".join(text)


def get_vocab(
    path_to_corpus: str,
    specials: List[str] = [
        PAD,
        UNK,
        EOS,
        BOS,
    ],
    vocab_size: Optional[int] = None,
) -> Vocab:
    return build_vocab_from_iterator(
        _yield_token(path_to_corpus), specials=specials, max_tokens=vocab_size
    )


def _yield_token(path_to_corpus: str) -> Generator[List[str], None, None]:
    with open(path_to_corpus, "r", encoding="utf-8") as f:
        for line in f:
            yield tokenize_sentence(line)


def tokenize_sentence(sentence: str) -> List[str]:
    """トークンごとに空白で区切られた文章をトークンの配列に変換する。"""
    return sentence.strip().split()
