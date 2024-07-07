
import os
from dotenv import load_dotenv
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from typing import List, Optional, Union
from transformers import PreTrainedTokenizerFast
from characters import *
load_dotenv()
class BengaliBPETokenizer:
    def __init__(self):
        ## initialize the bpe tokenizer
        self._tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        # to ensure no tokens with space
        self._tokenizer.pre_tokenizer = Whitespace()

    def train_tokenizer(
        self,
        files:List[str],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: List[Union[str, AddedToken]] = ["<unk>"],
        initial_tokens: List[str] = None,
        limit_alphabet: int = 1000,
        show_progress: bool = True,
        push_to_hub:bool = False,
        hf_repo_id:str = None,
        hf_token:str = None
    ):  
        
        self.special_tokens = special_tokens
        self.initial_tokens = initial_tokens
        
        # Calculate the remaining vocab size after accounting for special and predefined tokens
        remaining_vocab_size = vocab_size - len(special_tokens) - len(initial_tokens)
        
        trainer = BpeTrainer(
            vocab_size=remaining_vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=initial_tokens,
            limit_alphabet=limit_alphabet,
            show_progress=show_progress,
        )
        
        # Train the tokenizer
        self._tokenizer.train(
            files=files, 
            trainer=trainer
            )
        
        # Add special tokens and predefined tokens to ensure they're in the vocabulary
        all_sp_tokens = self.special_tokens + self.initial_tokens
        new_tokens = [token for token in all_sp_tokens if token not in self._tokenizer.get_vocab()]
        self._tokenizer.add_tokens(new_tokens)

        transformer_tokenizer = self.create_hf_tokenizer(
            push_to_hub=push_to_hub,
            hf_repo_id=hf_repo_id,
            hf_token=hf_token
        )
        
        return transformer_tokenizer

    def create_hf_tokenizer(
        self,
        push_to_hub:bool=True,
        hf_repo_id:str=None,
        hf_token:str=None
    ):
        transformer_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self._tokenizer #tokenizer_file="tokenizer.json"
            )
        
        if push_to_hub and hf_repo_id and hf_token:
            transformer_tokenizer.push_to_hub(
                repo_id = hf_repo_id,
                token=hf_token
                )
            print(f"Tokenizer is saved to Huggingface. Repo id: {hf_repo_id}")
        
        return transformer_tokenizer

    def save(self, path: str):
        self._tokenizer.save(path)

    @classmethod
    def from_file(cls, path: str):
        tokenizer = Tokenizer.from_file(path)
        instance = cls([], [])
        instance.tokenizer = tokenizer
        return instance

    def validate_predefined_tokens(self):
        vocab = self._tokenizer.get_vocab()
        all_sp_tokens = self.special_tokens + self.initial_tokens
        
        print("Checking all special and predefined tokens existance.....")
        missing_token_list = [token for token in all_sp_tokens if token not in vocab]
        if len(missing_token_list) > 0:
            print(f"Warning: The following tokens were not found in vocabulary: '{missing_token_list}'")
        else:
            print(f"All predifined and special tokens were found in the tokenizer")

    def encode(self, text: str):
        return self._tokenizer.encode(text)


if __name__ == "__main__":

    special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
    initial_tokens = bangla_alphabets + conjunct_consonants + conj_with_kar + conj_with_fola

    tokenizer = BengaliBPETokenizer()
    files = [
        "/storagex/Sabbir/BengaliTokenizer/characters.txt",
        "/storage2/llm_data/BanglaLM_process_v2.txt"
        ]
    
    hf_tokenizer = tokenizer.train_tokenizer(
        files=files,
        vocab_size=30_000,
        special_tokens= [
            
        ],
        initial_tokens=initial_tokens,
        min_frequency=2,
        limit_alphabet=500,
        show_progress=True,
        push_to_hub=True,
        hf_repo_id="Virus-Proton/CustomTokenizer",
        hf_token=os.getenv("hf_token")
    )

    tokenizer.validate_predefined_tokens()

    # Save the tokenizer
    tokenizer.save("bengali_tokenizer.json")

    bengali_text =' আমাদের কোম্পানি ভোটদান পণ্য তৈরীর চীন মধ্যে নেতা এক।'
    encoded = hf_tokenizer.encode(bengali_text)
    print(f"Encoded Tokens: {encoded}")
    # print("Tokens:", encoded.tokens)
    # print("IDs:", encoded.ids)