import os
import random
import itertools
from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel, WhitespaceSplit
from typing import List
from dotenv import load_dotenv
load_dotenv()

class HFTokenizerTrainer:
    def __init__(self, base_tokenizer_name):
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
        self.bengali_regex_pattern = r'([\p{Bengali}\p{Latin}\p{Nd}]+)|\S'
        
    def setup_tokenizer(self):
        self.base_tokenizer.pre_tokenizer = Sequence([
            Split(pattern=self.bengali_regex_pattern, behavior="isolated"),
            WhitespaceSplit(),
            ByteLevel(add_prefix_space=False, trim_offsets=False)
            ])
    
    def _line_generator(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                yield line.strip()

    def _all_files_line_generator(self, files):
        return itertools.chain.from_iterable(
            self._line_generator(file_path) for file_path in files
        )

    def batch_generator(self, files, batch_size=1000):
        line_gen = self._all_files_line_generator(files=files)
        batch = []
        for line in line_gen:
            batch.append(line)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def train(self, files:List=None, vocab_size:int=None):
        print("Training tokenizer...")
        self.setup_tokenizer()
        self.tokenizer = self.base_tokenizer.train_new_from_iterator(
            self.batch_generator(files=files),
            vocab_size=vocab_size if vocab_size is not None else self.base_tokenizer.vocab_size,
        )
        return self.tokenizer

    def save_tokenizer(self, output_dir):
        self.tokenizer.save_pretrained(output_dir)
        print(f"Tokenizer saved in {output_dir}")

    def push_to_hub(self, repo_id, token):
        self.tokenizer.push_to_hub(repo_id, token=token)

    def train_and_push(self, files, vocab_size, output_dir, push_to_hub:bool=False, repo_id=None, token=None):
        self.train(files=files, vocab_size=vocab_size)
        self.save_tokenizer(output_dir=output_dir)
        if push_to_hub:
            self.push_to_hub(repo_id=repo_id, token=token)
            print(f"Tokenizer trained and pushed to {repo_id}")
        return self.tokenizer


if __name__ == "__main__":
    files = [
        "/storagex/Sabbir/BengaliTokenizer/characters.txt",
        '/storage2/llm_data/data_all_text/AllTextData/ai4bharat.txt',
        '/storage2/llm_data/data_all_text/AllTextData/cc_100.txt',
        ]
    trainer = HFTokenizerTrainer(
        base_tokenizer_name="Qwen/Qwen2-0.5B",
    )
    new_tokenizer = trainer.train_and_push(
        files=["demo.txt"],
        output_dir="output",
        push_to_hub=True,
        repo_id="aci-mis-team/qwen2_tokenizer_32k",
        token=os.getenv("hf_token"),
        vocab_size=32_000
    )
    