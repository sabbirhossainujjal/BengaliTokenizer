import json
from tqdm import tqdm
import time
import re

class DataCleaning:
    def __init__(self) -> None:
        self.vocab = {}
        # Unicode ranges for Bengali, English, Arabic, and popular emojis
        BENGALI_UNICODE = r'\u0980-\u09FF'  # Bengali
        ENGLISH_UNICODE = r'A-Za-z'  # English letters (A-Z, a-z)
        ARABIC_UNICODE = r'\u0600-\u06FF'  # Arabic
        EMOJI_UNICODE = r'\U0001F600-\U0001F64F'  # Emojis range (e.g., smileys)

        self.allowed_chars_pattern = f"[{BENGALI_UNICODE}{ENGLISH_UNICODE}{ARABIC_UNICODE}{EMOJI_UNICODE}]+"


    def load_data(self, file_name):
        with open(file_name, 'r') as f:
            data =  f.read()
        
        return data
    
    def vocab_count(self, filenames):
        
        for filename in filenames:
            data = self.load_data(filename)
            for word in tqdm(data.split()):
                try:
                    self.vocab[word] += 1
                except:
                    self.vocab[word] == 1
                
        with open("manual_vocab.json", 'w') as f:
            json.dump(self.vocab, f)
       
        
    def remove_long_words(self, filenames):
        for filename in filenames:
            file_parts = filename.rsplit('.', 1)
            new_file = f"{file_parts[0]}_edited.{file_parts[1]}"
            st = time.time()
            with open(filename, 'r') as infile, open(new_file, 'w') as outfile:
                for line in tqdm(infile):
                    filtered_words = [word for word in line.split() if len(word) <= 10]
                    edited_line = ' '.join(filtered_words)
                    outfile.write(edited_line + '\n')
    
            print(f"Total time taken for edit file {filename} : {(time.time() - st)/60} m")

    def remove_invalid_chars(self, filenames):
        """
        This function processes the input text files and removes characters other than 
        Bengali, English, Arabic, and emojis, then writes the cleaned output to new files.
        """
        for filename in filenames:
            file_parts = filename.rsplit('.', 1)
            new_file = f"{file_parts[0]}_cleaned.{file_parts[1]}"
            st = time.time()
            with open(filename, 'r') as infile, open(new_file, 'w') as outfile:
                for line in tqdm(infile):
                    # Find only allowed characters and join them
                    filtered_line = ' '.join(re.findall(self.allowed_chars_pattern, line))
                    outfile.write(filtered_line + '\n')
    
            print(f"Total time taken for editing file {filename}: {(time.time() - st) / 60:.2f} minutes")

            
if __name__ == "__main__":
    files = [
        '/storage2/llm_data/data_all_text/AllTextData/ai4bharat_edited.txt',
        '/storage2/llm_data/data_all_text/AllTextData/cc_100_edited.txt',
        ]
    data_cleaner = DataCleaning()
    # data_cleaner.vocab_count(filenames=files)
    # data_cleaner.remove_long_words(files)
    data_cleaner.remove_invalid_chars(filenames=files)
    
    
    
    
   # files = [
    #    # "/storagex/Sabbir/BengaliTokenizer/demo.txt"
    #    # "/storagex/Sabbir/BengaliTokenizer/characters.txt",
    #    # "/storage2/llm_data/BanglaLM_process_v2.txt",
   #     '/storage2/llm_data/data_all_text/AllTextData/ai4bharat_edited.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/BanglaLM_raw.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2012.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2013_20.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2013_48.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2014_15.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2014_23.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2014_35.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2014_41.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2014_42.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2014_49.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2014_52.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_06.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_11.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_14.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_18.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_22.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_27.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_32.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_35.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_40.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2015_48.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2016_30.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2016_50.txt',
    #    # '/storage2/llm_data/data_all_text/AllTextData/bn.2017_17.txt',
   #     '/storage2/llm_data/data_all_text/AllTextData/cc_100_edited.txt',
   #     ]