from __future__ import annotations

from pathlib import Path

import sentencepiece as spm


def train_sentencepiece(
    input_txt: str | Path,
    model_prefix: str | Path,
    vocab_size: int = 8000,
    character_coverage: float = 0.9995,
) -> Path:
    input_txt = Path(input_txt)
    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(input_txt),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type="unigram",
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        hard_vocab_limit=False,  # small corpora may yield fewer pieces
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=[],
    )
    return Path(str(model_prefix) + ".model")


class Tokenizer:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.sp.bos_id()] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids: list[int] | list[list[int]]) -> str:
        if ids and isinstance(ids[0], list):
            return self.sp.decode(ids[0])  # type: ignore[arg-type]
        return self.sp.decode(ids)  # type: ignore[arg-type]
