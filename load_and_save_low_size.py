import os
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from dotenv import load_dotenv
import csv
import io
from tqdm import tqdm

load_dotenv()

def process_large_file(input_file, output_file, max_size):
    bytes_written = 0
    chunk_size = 1024 * 1024  # 1 MB chunks
    max_size_bytes = max_size * 1024 * 1024 * 1024  # GB conversion
    pbar = tqdm(total=max_size_bytes, unit='B', unit_scale=True, desc="Processing file")
    with open(input_file, 'rb') as infile, open(output_file, 'wb') as outfile:
        while bytes_written < max_size_bytes:
            chunk = infile.read(min(chunk_size, max_size_bytes - bytes_written))
            if not chunk:
                break
            outfile.write(chunk)
            bytes_written += len(chunk)
            pbar.update(len(chunk))
        pbar.close()
    print(f"Processed {bytes_written/(1024*1024*1024):.2f} GB")

def iter_chunks(file_path, chunk_size=1024*1024):  # 1MB chunks
    with open(file_path, 'r') as f:
        chunk = ""
        for line in f:
            chunk += line
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = ""
        if chunk:
            yield chunk

def chunk_generator(file_path):
    for chunk in iter_chunks(file_path):
        texts = chunk.split("\n\n")
        for text in texts:
            if text.strip():  # Ensure the text is not empty
                yield {"Text": text}

def load_and_push_dataset(filepath: str):

    # Create dataset in streaming mode
    total_chunks = sum(1 for _ in iter_chunks(filepath))
    dataset = Dataset.from_generator( lambda: tqdm(chunk_generator(filepath), total=total_chunks, desc="Generating dataset"))

    # Push to hub in streaming mode
    dataset.push_to_hub(
        "aci-mis-team/llm_training_data",
        split="cc_100_5GB",
        token=os.getenv("HF_TOKEN"),
        writer_batch_size=1000  # Adjust this based on your memory constraints
    )

    print("Dataset uploaded successfully!")

if __name__ == "__main__":
    input_file = '/storage2/llm_data/data_all_text/AllTextData/cc_100.txt'
    output_file = '/storage2/llm_data/data_all_text/AllTextData/cc_100_5GB.txt'
    data_size = 5   # 5 GB

    process_large_file(input_file, output_file, data_size)
    load_and_push_dataset(filepath=output_file)


