import json
import regex as re
from typing import Union, List
from tokenizers import SentencePieceBPETokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from characters import *

# bangla_alphabets = [
#     '০',  # Zero
#     '১',  # One
#     '২',  # Two
#     '৩',  # Three
#     '৪',  # Four
#     '৫',  # Five
#     '৬',  # Six
#     '৭',  # Seven
#     '৮',  # Eight
#     '৯',# Nine
#     'অ',
#     'আ',
#     'ই',
#     'ঈ',
#     'উ',
#     'ঊ',
#     'ঋ',
#     'এ',
#     'ঐ',
#     'ও',
#     'ঔ',
#     'ক',
#     'খ',
#     'গ',
#     'ঘ',
#     'ঙ',
#     'চ',
#     'ছ',
#     'জ',
#     'ঝ',
#     'ঞ',
#     'ট',
#     'ঠ',
#     'ড',
#     'ঢ',
#     'ণ',
#     'ত',
#     'থ',
#     'দ',
#     'ধ',
#     'ন',
#     'প',
#     'ফ',
#     'ব',
#     'ভ',
#     'ম',
#     'য',
#     'র',
#     'ল',
#     'শ',
#     'ষ',
#     'স',
#     'হ',
#     'ড়',
#     'ঢ়',
#     'য়',
#     'ৎ',
#     'ং',
#     'ঃ',
#     'ঁ',
#     'া',  #  - আ কার (aa)
#     'ি',  # - ই কার (i)
#     'ী',  #  - ঈ কার (ii)
#     'ু',  #  - উ কার (u)
#     'ূ',  #  - ঊ কার (uu)
#     'ৃ',  #  - ঋ কার (rri)
#     'ে',  #  - এ কার (e)
#     'ৈ',  #  - ঐ কার (oi)
#     'ো',  #  - ও কার (o)
#     'ৌ',  #  - ঔ কার (ou)
#     '্',  # হসন্ত (hasanta)
#     '়',  # নুক্তা (nukta)
#     '।',  # দাঁড়ি (daari)
#     '॥',  # ডাবল দাঁড়ি (double daari)
#     '“',  # উদ্ধৃতি খোলা (opening quotation mark)
#     '”',  # উদ্ধৃতি বন্ধ (closing quotation mark)
#     '‘',  # একক উদ্ধৃতি খোলা (single opening quote)
#     '’',  # একক উদ্ধৃতি বন্ধ (single closing quote)
#     '–',  # এন ড্যাশ (en dash)
#     '—',  # এম ড্যাশ (em dash)
#     '৳',
#     ]

bangla_alphabets = bangla_alphabets + conjunct_consonants + conj_with_kar + conj_with_fola

bengali_regex_pattern = r"""[^\r\n\p{Bengali}\p{L}\p{Nd}]?[\p{Bengali}\u0981-\u09C4\u09C7\u09C8\u09CB\u09CC\u09CD\p{L}]*[\p{Bengali}\u0981-\u09C4\u09C7\u09C8\u09CB\u09CC\u09CD\p{L}]+|[^\r\n\p{Bengali}\p{L}\p{Nd}]?[\p{Bengali}\u0981-\u09C4\u09C7\u09C8\u09CB\u09CC\u09CD\p{L}]+[\p{Bengali}\u0981-\u09C4\u09C7\u09C8\u09CB\u09CC\u09CD\p{L}]*|[০-৯]{1,4}| ?\p{N}+| ?[^\s\p{Bengali}\p{L}\p{Nd}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# initial_merges = {
#     (224, 166): 256,
#     (224, 167): 257,
#     (224, 165): 258,
#     (226, 128): 259
# }

def load_data(filepath='./small_data_1MB.txt'):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read().replace('\n', '')
    return text

def split_text_with_regex(pattern: str, text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
    """
    Splits a given text or list of texts into pieces based on the provided regex pattern.

    Parameters:
    pattern (str): The regex pattern to apply for splitting.
    text (Union[str, List[str]]): The input text or list of texts to be split.

    Returns:
    Union[List[str], List[List[str]]]: The split pieces as a list of strings or a list of lists of strings.
    """
    # def split_single_text(single_text: str) -> List[str]:
    #     return re.findall(pattern, single_text)
    
    def split_single_text(single_text: str) -> List[str]:
        # pattern = r'\S+'  # Adjust this pattern according to your splitting needs
        split_list = re.findall(pattern, f" {single_text}")
        
        # Define a pattern to match an initial space followed by a non-space character
        initial_space_pattern = re.compile(r'^ (\S)')

        # Iterate through the split list and replace the initial space with '_'
        modified_list = [
            initial_space_pattern.sub(r'_\1', item) if initial_space_pattern.match(item) else item
            for item in split_list
        ]
        
        return modified_list
    
    if isinstance(text, str):
        return split_single_text(text)
    elif isinstance(text, list):
        return [split_single_text(t) for t in text]
    else:
        raise TypeError("Input must be a string or a list of strings.")

def merge_initial_tokens(ids_list, sequence, idx):
    newids_list = []
    seq_len = len(sequence)

    # Ensure ids_list is a list of lists
    if not(isinstance(ids_list[0], list)):
        ids_list = [ids_list]
    # Iterate over each list in ids_list
    for ids in ids_list:
        i = 0
        newids = []
        # Iterate over the elements in the list
        while i < len(ids):
            # Check for the sequence and replace with idx if found
            if i <= len(ids) - seq_len and all(ids[i+j] == sequence[j] for j in range(seq_len)):
                newids.append(idx)
                i += seq_len
            else:
                newids.append(ids[i])
                i += 1
        newids_list.append(newids)
    return newids_list

def merge_tokens(ids_list, pair, idx):
    """
    Merges a specified pair of consecutive elements in a list or lists of integers with a new value.

    Args:
        ids_list (list of int or list of list of int): The list or lists of integers to be merged.
        pair (tuple of int): A pair of integers which, if found consecutively in the list, will be merged.
        idx (int): The new value with which the consecutive pair will be replaced.

    Returns:
        list of list of int: A list of lists where each sublist has the specified consecutive pair replaced by the new value.

    Example:
        >>> merge_tokens([[1, 2, 3, 2, 3], [2, 3, 4]], (2, 3), 9)
        [[1, 9, 9], [9, 4]]
    """
    newids_list = []

    # Ensure ids_list is a list of lists
    if not(isinstance(ids_list[0], list)):
        ids_list = [ids_list]

    # Iterate over each list in ids_list
    for ids in ids_list:
        i = 0
        newids = []
        # Iterate over the elements in the list
        while i < len(ids):
            # Check for the pair and replace with idx if found
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        newids_list.append(newids)
    return newids_list

def expand_a_token(merges, id):
    """
    Recursively expands a token into a list of sub-tokens based on a merges dictionary.

    Args:
        merges (dict): A dictionary where keys are token IDs and values are tuples representing merged tokens.
        id (int): The token ID to be expanded.

    Returns:
        list: A list of integers representing the expanded sub-tokens of the original token.

    Example:
        >>> merges = {256: (101, 102), 257: (256, 103)}
        >>> expand_a_token(merges, 257)
        [101, 102, 103]
    """
    merges_map = {v: k for k, v in merges.items()}
    if id < 256:
        return [id]
    else:
        mapped_token_list = list(merges_map[id])
        final_list = []
        for token in mapped_token_list:
            final_list.extend(expand_a_token(merges, token))
        return final_list

def decode_utf8_from_integers(int_list):
    """
    Decodes a list of integers into a UTF-8 string.

    This function attempts to convert a list of integers, where each integer
    represents a byte value, into a UTF-8 encoded string. If the conversion
    fails due to a UnicodeDecodeError, an empty string is returned.

    Args:
        int_list (list of int): A list of integers representing byte values.

    Returns:
        str: The decoded UTF-8 string or an empty string if decoding fails.

    Example:
        >>> decode_utf8_from_integers([72, 101, 108, 108, 111])
        'Hello'
    """
    try:
        # Convert list of integers to bytes
        byte_data = bytes(int_list)
        # Decode the byte data to a UTF-8 string
        decoded_string = byte_data.decode('utf-8')
        return decoded_string
    except UnicodeDecodeError as e:
        return ""

def decode_a_token(merges, id):
    """
    Decodes a single token ID into its corresponding UTF-8 string representation.

    This function takes a token ID and a dictionary of merges, expands the token ID
    into a list of sub-token IDs using the `expand_a_token` function, and then decodes
    the list of sub-token IDs into a UTF-8 string using the `decode_utf8_from_integers` function.

    Args:
        merges (dict): A dictionary where keys are token IDs and values are tuples representing merged tokens.
        id (int): The token ID to be decoded.

    Returns:
        str: The decoded UTF-8 string representation of the token ID.

    Example:
        >>> merges = {256: (101, 102), 257: (256, 103)}
        >>> decode_a_token(merges, 257)
        'eac'  # Assuming 'eac' is the string representation of the token ID 257
    """
    expanded_list = expand_a_token(merges, id)
    decoded_str = decode_utf8_from_integers(expanded_list)
    return decoded_str

def get_ids_list(str_list):
    """
    Converts a list of strings into a list of lists, where each inner list contains the integer byte values of the UTF-8 encoded characters of a string.

    Args:
        str_list (list of str or str): A list of strings or a single string to be converted.

    Returns:
        list of list of int: A list where each element is a list of integers representing the UTF-8 encoded byte values of the characters of the strings.

    Example:
        >>> get_ids_list(["hello", "world"])
        [[104, 101, 108, 108, 111], [119, 111, 114, 108, 100]]
    """
    if isinstance(str_list, str):
        str_list = [str_list]
    ids_list = [list(map(int, line.encode("utf-8"))) for line in str_list]
    return ids_list

def num_total_tokens(ids_list):
    """
    Calculates the total number of tokens (integer byte values) in a list of lists.

    Args:
        ids_list (list of list of int or list of int): A list where each element is a list of integers representing tokens. If a single list of integers is provided, it is converted to a list of lists.

    Returns:
        int: The total number of tokens across all lists.

    Example:
        >>> num_total_tokens([[104, 101, 108, 108, 111], [119, 111, 114, 108, 100]])
        10
    """
    if not(isinstance(ids_list[0], list)):
        ids_list = [ids_list]
    return sum(len(ids) for ids in ids_list)

def get_stats(ids_list):
    """
    Counts the occurrences of consecutive byte pairs in a list of lists, while ignoring certain byte pairs where the first byte is not allowed at the start or the second byte is not allowed at the end.

    Args:
        ids_list (list of list of int): A list where each element is a list of integers representing the UTF-8 encoded byte values of characters.

    Returns:
        dict: A dictionary with the byte pairs as keys and their counts as values.

    Example:
        >>> get_stats([[224, 165, 314], [224, 165, 315]])
        {(224, 165): 2}

    Note:
        The function defines a list of byte values that are not allowed at the start of a byte pair. Currently, the list of bytes not allowed at the end is empty.
    """
    tokens_not_at_start = [
        314,  # => ং
        315,  # => ঃ
        316,  # => ঁ
        317,  # => া
        318,  # => ি
        319,  # => ী
        320,  # => ু
        321,  # => ূ
        322,  # => ৃ
        323,  # => ে
        324,  # => ৈ
        325,  # => ো
        326,  # => ৌ
        327,  # => ্
        328,  # => ়
    ]
    tokens_not_at_end = [
        # 327  # => ্
        ]
    if not(isinstance(ids_list[0], list)):
        ids_list = [ids_list]
    counts = {}
    for ids in ids_list:
        for pair in zip(ids, ids[1:]):
            if (
                (pair[0] in tokens_not_at_start) or
                (pair[1] in tokens_not_at_end)
            ):
                continue
            counts[pair] = counts.get(pair, 0) + 1
    return counts

def get_initial_merges(ids_list) -> dict:
    """
    Args:
        initial_merges (Union[dict, None], optional): initial merge dict if any initial merges needed . Defaults to None.

    Returns:
        all alphabet and initial special merges
    """
    
    merges = {}
    # Add predetermined Bangla Alphabet to merges
    for char in bangla_alphabets:
        encoded = list(map(int, char.encode("utf-8")))
        merges[tuple(encoded)] = 256 + len(merges)
    
    for sequence, idx in merges.items():
        ids_list = merge_initial_tokens(ids_list, sequence, idx)
    
    return merges, ids_list

def get_merges(text, vocab_size=10_000, min_frequency:int=1):
    
    ## split the given data according to regex pattern
    patterned_texts = split_text_with_regex(bengali_regex_pattern, text)
    ## get tokens ids according to regex extraction
    ids_list = get_ids_list(patterned_texts)

    ## get the initial merges and alphabets in the merges dict
    merges, ids_list = get_initial_merges(ids_list=ids_list)

    # loop, merge and expand vocab untill vocab size is fullfilled
    while (len(merges) + 255) < vocab_size:
        stats = get_stats(ids_list)
        pair = max(stats, key=stats.get)
        # max_freq = stats[pair]
        # if max_freq < min_frequency:
        #     break
        idx = 256 + len(merges)
        ids_list = merge_tokens(ids_list, pair, idx)
        merges[pair] = idx
        
    return merges

def get_vocab(merges:dict):
    vocab = {idx: bytes([idx]) for idx in range(256)} # byte converted initial tokens (all alphabets)
    for tup, idx in merges.items():
        if len(tup) == 2:
            vocab[idx] = vocab[tup[0]]+vocab[tup[1]]
        elif len(tup) == 3:
            vocab[idx] = vocab[tup[0]]+vocab[tup[1]]+vocab[tup[2]]
        else: # solve for 'ঢ়
            for i in range(len(tup)):
                if i==0:
                    char = vocab[tup[i]]
                else:
                    char += vocab[tup[i]]
    
    return vocab

def decode(ids:Union[List, int], vocab:dict) -> str:
    """
    Args:
        ids (list): List of ids
        vocab (dict): Generated Vocab from merges

    Returns:
        str: decoded string
    """
    if isinstance(ids, int):
        ids = [ids]
    tokens = b"".join(vocab[id] for id in ids)
    return tokens.decode("utf-8", errors="backslashreplace") #backslashreplace

def save_tokenizer_files(vocab:dict, merges:dict, special_tokens:Union[list, None]=['<unk>']):
    ## save vocab as vocab.json file as required for hf tokenizer    
    # format : {'token': id, ...}
    if special_tokens is not None and len(special_tokens) > 0:
        vocab_n = {special_tokens[i]:i for i in range(0, len(special_tokens))}
        vocab_n.update({decode(v, vocab):k+len(special_tokens) for k, v in vocab.items()})
    else:
        vocab_n = {decode(v, vocab):k for k, v in vocab.items()}
    
    with open('vocab.json', 'w', encoding="utf-8") as json_file:
        json.dump(vocab_n, json_file, indent=4, ensure_ascii=False)
    
    ## save merges as merge.txt file as required for hf tokenizer
    merge_list = list(merges.keys())[len(bangla_alphabets):]
    with open('merges.txt', 'w') as file:
        for item in merge_list:
            file.write(f"{decode(vocab[item[0]], vocab=vocab)} {decode(vocab[item[1]], vocab=vocab)}\n")

def train_tokenizer(filepath:str, vocab_size:int=100, special_tokens:Union[list, None]=None):
    text = load_data(filepath=filepath)
    merges = get_merges(text=text, vocab_size=vocab_size)
    vocab = get_vocab(merges=merges)

    save_tokenizer_files(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    return vocab, merges

def get_hf_tokenizer(vocab_filepath, merges_filepath):
    tokenizer = SentencePieceBPETokenizer(
        vocab=vocab_filepath,
        merges=merges_filepath,
        unk_token='<unk>',
    )
    tokenizer.save("tokenizer.json")
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, #tokenizer_file="tokenizer.json"
        model_max_length=512,
        
    )
    hf_tokenizer.bos_token = "<s>"
    hf_tokenizer.bos_token_id = tokenizer.token_to_id("<s>")
    hf_tokenizer.pad_token = "<pad>"
    hf_tokenizer.pad_token_id = tokenizer.token_to_id("<pad>")
    hf_tokenizer.eos_token = "</s>"
    hf_tokenizer.eos_token_id = tokenizer.token_to_id("</s>")
    hf_tokenizer.unk_token = "<unk>"
    hf_tokenizer.unk_token_id = tokenizer.token_to_id("<unk>")
    hf_tokenizer.mask_token = "<mask>"
    hf_tokenizer.mask_token_id = tokenizer.token_to_id("<mask>")
    
    hf_tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )
    hf_tokenizer.save_pretrained("./hf_tokenizer/")
    return hf_tokenizer


# if __name__ == "__main__":
#     train_tokenizer('demo.txt', vocab_size=2000, special_tokens=['<unk>'])
#     tokenizer = get_hf_tokenizer(
#         vocab_filepath='vocab.json',
#         merges_filepath='merges.txt'
#     )
    



