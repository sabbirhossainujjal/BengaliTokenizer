
import os
from typing import List
from transformers import AutoTokenizer
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel, WhitespaceSplit
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

<<<<<<< HEAD
base_tokenizer_name = "Qwen2-0.5B/"
=======
base_tokenizer_name = "qwen_tokenizer/"
>>>>>>> 77fa1afa5fed987bb77b01cb9c6e9ab173c3f18c
base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)


def _line_generator(file_path: str):
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

def batch_generator(files: List[str], batch_size: int = 5000):

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
        for line in _line_generator(file):
            batch.append(line)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
            

vocab_size = 48_000
output_dir = "qwen_tokenizer_trained_48k"

files = [
    # "/storagex/Sabbir/BengaliTokenizer/demo_edited.txt"
    '/storage2/llm_data/data_all_text/AllTextData/ai4bharat.txt',
    '/storage2/llm_data/data_all_text/AllTextData/cc_100.txt',
    '/storage2/llm_data/data_all_text/AllTextData/c4_New.txt',
    "/storage2/llm_data/BanglaLM_process_v2.txt",
    # '/storage2/llm_data/data_all_text/AllTextData/CulturaX.txt'
    ]

tokenizer = base_tokenizer.train_new_from_iterator(
    tqdm(batch_generator(files), desc="Training"),
    vocab_size=vocab_size
)

tokenizer.save_pretrained(output_dir)

tokenizer.push_to_hub("aci-mis-team/Qwen-2-48k", token=os.getenv("HF_TOKEN"))



