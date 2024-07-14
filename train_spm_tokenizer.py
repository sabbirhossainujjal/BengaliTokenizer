from characters import *

import sentencepiece as spm

# Predefined tokens
predefined_tokens = bangla_alphabets + conj_with_fola + conj_with_kar

text_path = "demo_1M.txt"
# with open(text_path, 'r', encoding='utf-8') as file:
#     bengali_text = file.read()

# bengali_text = ' '.join(bengali_text.split())

# Create a SentencePiece model
model_prefix = 'bengali_tokenizer'
vocab_size = 45_000
character_coverage = 1.0
model_type = 'bpe'
files = [
        # "/storagex/Sabbir/BengaliTokenizer/multi_char.txt",
        # "/storage2/llm_data/BanglaLM_process_v2.txt",
        '/storage2/llm_data/data_all_text/AllTextData/ai4bharat.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/banglaLM_process_v1.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/BanglaLM_raw.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2012.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2013_20.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2013_48.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2014_15.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2014_23.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2014_35.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2014_41.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2014_42.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2014_49.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2014_52.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_06.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_11.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_14.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_18.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_22.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_27.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_32.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_35.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_40.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_48.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2016_30.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2016_50.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/bn.2017_17.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/cc_100.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/output_text_file.txt',
        # '/storage2/llm_data/data_all_text/AllTextData/raw_bangla_for_BERT.txt'
        ]


spm.SentencePieceTrainer.Train(
    input=files, #"/home/virus_proton/Projects/P_Projects/LLM_Mastery/Tokenizer_train/characters.txt",
    model_prefix=model_prefix,
    model_type=model_type,
    vocab_size=vocab_size,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=predefined_tokens,
    character_coverage=1.0,
    split_by_whitespace=True,
    # split_by_unicode_script=True,
    input_sentence_size=39000000,
    max_sentence_length=5000,
    train_extremely_large_corpus=True,
    required_chars=bangla_alphabets,
    num_threads=5,
    random_seed=2024,
)
