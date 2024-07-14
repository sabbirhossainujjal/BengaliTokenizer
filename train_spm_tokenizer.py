from characters import *

import sentencepiece as spm

# Predefined tokens
predefined_tokens = bangla_alphabets + conj_with_fola + conj_with_kar

text_path = "demo_1M.txt"
with open(text_path, 'r', encoding='utf-8') as file:
    bengali_text = file.read()

bengali_text = ' '.join(bengali_text.split())

# Create a SentencePiece model
model_prefix = 'bengali_tokenizer'
vocab_size = 3000
character_coverage = 1.0
model_type = 'bpe'


spm.SentencePieceTrainer.Train(
    input=bengali_text, #"/home/virus_proton/Projects/P_Projects/LLM_Mastery/Tokenizer_train/characters.txt",
    model_prefix=model_prefix,
    model_type=model_type,
    vocab_size=vocab_size,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    # user_defined_symbols=",".join(predefined_tokens),
    num_threads=5,
    split_by_whitespace=True,
    split_by_unicode_script=True,
    # required_chars=bangla_alphabets
    
)

# sp = spm.SentencePieceProcessor(model_file='/home/virus_proton/Projects/P_Projects/LLM_Mastery/Tokenizer_train/bengali_tokenizer.model')

# print(sp.encode("লিখা", out_type=str))