import os
import re
import json
from dotenv import load_dotenv
from tokenizers import Tokenizer, pre_tokenizers, normalizers
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
load_dotenv()

bengali_regex_pattern = r"""[^\r\n\p{Bengali}\p{L}\p{Nd}]?[\p{Bengali}\u0981-\u09C4\u09C7\u09C8\u09CB\u09CC\u09CD\p{L}]*[\p{Bengali}\u0981-\u09C4\u09C7\u09C8\u09CB\u09CC\u09CD\p{L}]+|[^\r\n\p{Bengali}\p{L}\p{Nd}]?[\p{Bengali}\u0981-\u09C4\u09C7\u09C8\u09CB\u09CC\u09CD\p{L}]+[\p{Bengali}\u0981-\u09C4\u09C7\u09C8\u09CB\u09CC\u09CD\p{L}]*|[০-৯]{1,4}| ?\p{N}+| ?[^\s\p{Bengali}\p{L}\p{Nd}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


class BnBPETokenizer:
    def __init__(self, base_model_name="Qwen/Qwen2-0.5B-Instruct"):
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer = Tokenizer.from_pretrained(base_model_name)
        
        # Set the normalizer to NFC
        self.tokenizer.normalizer = normalizers.Sequence([NFC()])
        
        # Set up the pre-tokenizer
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            Split(
                pattern=bengali_regex_pattern,
                behavior="isolated"
            ),
            # ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
        ])
        
    def train(self, files, vocab_size=30000, min_frequency=2):
        
        trainer = BpeTrainer(
            vocab_size=vocab_size, 
            min_frequency=min_frequency, 
            special_tokens=["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<mask>"]
        )
        
        self.tokenizer.train(files, trainer=trainer)
        # self.tokenizer.model.unk_token = "<unk>"
        

    def save(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # self.tokenizer_config = self.base_tokenizer.tokenizer_config
        
        # save the tokenizer and tokenizer config
        self.tokenizer.save(f"{output_dir}/tokenizer.json")
        # with open(f"{output_dir}/tokenizer_config.json", "w") as f:
        #     json.dump(self.tokenizer_config, f, indent=4, ensure_ascii=False)
    
    def get_hf_tokenizer(self, model_max_length:int=512, push_to_hub:bool=False, repo_id:str=None, token:str=None):
        
        # self.hf_tokenizer = PreTrainedTokenizerFast(
        #     tokenizer_object=self.tokenizer, 
        #     model_max_length=model_max_length,
        # )
        
        self.hf_tokenizer = self.tokenizer
        
        self.hf_tokenizer.bos_token = "<|im_start|>"
        # self.hf_tokenizer.bos_token_id = self.tokenizer.token_to_id("<|im_start|>")
        self.hf_tokenizer.pad_token = self.base_tokenizer.pad_token
        # self.hf_tokenizer.pad_token_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.hf_tokenizer.eos_token = self.base_tokenizer.eos_token
        # self.hf_tokenizer.eos_token_id = self.tokenizer.token_to_id("<|im_end|>")
        self.hf_tokenizer.unk_token = "<unk>"
        # self.hf_tokenizer.unk_token_id = self.tokenizer.token_to_id("<unk>")
        self.hf_tokenizer.mask_token = "<mask>"
        self.hf_tokenizer.mask_token_id = self.tokenizer.token_to_id("<mask>")

        self.hf_tokenizer.additional_special_tokens = self.base_tokenizer.additional_special_tokens
        
        for key, value in vars(self.base_tokenizer).items():
            if not key.startswith('_'):
                setattr(self.hf_tokenizer, key, value)
        
        # if push_to_hub:
        #     self.hf_tokenizer.save_pretrained(repo_id=repo_id, token=token)
            
        # bos_token = self.hf_tokenizer.special_tokens_map['bos_token']
        # eos_token = self.hf_tokenizer.special_tokens_map['eos_token']
        # # Create a TemplateProcessing post-processor
        # post_processor = TemplateProcessing(
        #     single=f"{bos_token} $A {eos_token}",
        #     special_tokens=[
        #         (bos_token, self.hf_tokenizer.convert_tokens_to_ids(bos_token)),
        #         (eos_token, self.hf_tokenizer.convert_tokens_to_ids(eos_token)),
        #     ],
        # )
        # self.hf_tokenizer.backend_tokenizer.post_processor = post_processor

        

# Usage example
if __name__ == "__main__":
    files = ["demo.txt"]  # List of text files for training
    tokenizer_trainer = BnBPETokenizer()
    tokenizer_trainer.train(files)
    tokenizer_trainer.save("./output")
    tokenizer_trainer.get_hf_tokenizer(model_max_length=30000, push_to_hub=True, repo_id="Virus-Proton/llm_tokenizer", token=os.getenv("hf_token"))
