import os
from typing import List
from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel, WhitespaceSplit
from tqdm import tqdm

class HFTokenizerTrainer:
    def __init__(self, base_tokenizer_name: str):
        """
        Args:
            base_tokenizer_name (str): The name (hf_repo) of the base tokenizer to use from Hugging Face.
        """
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)

        self.setup_tokenizer()

    def setup_tokenizer(self, pre_tokenizer = None, 
                        post_processor = None):
        """
        if custom pre_tokenizer and post_process is required then defined there.
        Set up the tokenizer with custom pre-tokenizer and post-processor.

        Args:
            pre_tokenizer (optional): HF defined pre_tokenizers. Defaults to None.
            post_processor (optional): HF defined post-processors. Defaults to None.
        """
        print("Modifying the base tokenizer.....")
        if pre_tokenizer:
            self.base_tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizer

        if post_processor:
            self.base_tokenizer.backend_tokenizer.post_processor = post_processor

    def _line_generator(self, file_path: str):
        """
        Generate lines from a file.
        Args:
            file_path (str): path of the file

        Yields:
            str: Each line from the file
        """
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    yield line.strip()
        except IOError as e:
            print(f"Error reading file {file_path}: {e}")

    def batch_generator(self, files: List[str], batch_size: int = 1000):
        """
        Generate batches of lines from multiple files.

        Args:
            files (List[str]): A list of file paths to read from.
            batch_size (int): The number of lines to include in each batch. Defaults to 1000.

        Yields:
            List[str]: Batches of lines from the input files.
        """
        
        for file in files:
            batch = []
            for line in self._line_generator(file):
                batch.append(line)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    def train(self, files: List[str], vocab_size: int = None, output_dir: str = None, 
              push_to_hub: bool = False, repo_id: str = None, token: str = None):
        """
        Train a new tokenizer and push it to the Hugging Face Hub.

        Args:
            files (List[str]): A list of file paths to use for training the tokenizer.
            vocab_size (int, optional): The vocabulary size for the new tokenizer. If None, uses the base tokenizer's vocab size.
            output_dir (str, optional): path to save the trained tokenizer. If None, the tokenizer won't be saved locally.
            push_to_hub (bool): Whether to push the trained tokenizer to the Hugging Face Hub. Defaults to False.
            repo_id (str, optional): The repository name on the Hugging Face Hub to push the trained tokenizer. Required if push_to_hub is True.
            token (str, optional): The authentication token for the Hugging Face Hub. Required if push_to_hub is True.

        Returns:
            new trained tokenizer.

        Raises:
            ValueError: If no input files are provided or if push_to_hub is True but repo_id or token is missing.
        """
        
        if not files:
            raise ValueError("No input files provided")
        
        print("Tokenizer Training Started...")
        vocab_size = vocab_size or self.base_tokenizer.vocab_size
        self.tokenizer = self.base_tokenizer.train_new_from_iterator(
            tqdm(self.batch_generator(files), desc="Training"),
            vocab_size=vocab_size
        )

        if output_dir:
            self.tokenizer.save_pretrained(output_dir)
            print(f"Tokenizer saved in {output_dir}")

        if push_to_hub:
            if not repo_id or not token:
                raise ValueError("repo_id and token are required when push_to_hub is True")
            self.tokenizer.push_to_hub(repo_id, token=token)
            print(f"Tokenizer pushed to {repo_id} on the Hugging Face Hub")

        return self.tokenizer

if __name__ == "__main__":
    files = [
        # "/storagex/Sabbir/BengaliTokenizer/demo_edited.txt"
        '/storage2/llm_data/data_all_text/AllTextData/ai4bharat.txt',
        '/storage2/llm_data/data_all_text/AllTextData/cc_100.txt',
        '/storage2/llm_data/data_all_text/AllTextData/c4_New.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/CulturaX.txt'
        ]
    trainer = HFTokenizerTrainer("Qwen/Qwen2-0.5B")
    
    bengali_regex_pattern = r"""[^\r\n\p{Bengali}\p{L}\p{Nd}]?[\p{Bengali}\u0981-\u09C4\u09C7\u09C8\u09CB\u09CC\u09CD\p{L}]*[\p{Bengali}\u0981-\u09C4\u09C7\u09C8\u09CB\u09CC\u09CD\p{L}]+|[^\r\n\p{Bengali}\p{L}\p{Nd}]?[\p{Bengali}\u0981-\u09C4\u09C7\u09C8\u09CB\u09CC\u09CD\p{L}]+[\p{Bengali}\u0981-\u09C4\u09C7\u09C8\u09CB\u09CC\u09CD\p{L}]*|[০-৯]{1,4}| ?\p{N}+| ?[^\s\p{Bengali}\p{L}\p{Nd}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    
    pre_tokenizer = Sequence([
                Split(pattern=bengali_regex_pattern, behavior="isolated"),
                WhitespaceSplit(),
                ByteLevel(add_prefix_space=False, trim_offsets=False)
                ])
    trainer.setup_tokenizer(
        pre_tokenizer=pre_tokenizer
    )
    
    new_tokenizer = trainer.train(
        files=files,
        vocab_size=32_000,
        output_dir="output_test",
        push_to_hub=True,
        repo_id="aci-mis-team/Qwen-2-32k",
        token=os.getenv("HF_TOKEN")
    )
