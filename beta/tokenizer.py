"""Train/load a small BPE tokenizer for DEON-LEN 1 Beta."""
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def train_tokenizer(corpus_path: str, output_path: str, vocab_size: int = 2048) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train([corpus_path], trainer)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def encode_text(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text).ids


def decode_ids(tokenizer: Tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)
