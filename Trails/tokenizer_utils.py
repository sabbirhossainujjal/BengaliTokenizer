import regex as re
from typing import List, Iterator
import tokenizers
from transformers import PreTrainedTokenizerFast
from tokenizers import SentencePieceBPETokenizer


bangla_alphabets = [
    '০',  # Zero
    '১',  # One
    '২',  # Two
    '৩',  # Three
    '৪',  # Four
    '৫',  # Five
    '৬',  # Six
    '৭',  # Seven
    '৮',  # Eight
    '৯',# Nine
    'অ',
    'আ',
    'ই',
    'ঈ',
    'উ',
    'ঊ',
    'ঋ',
    'এ',
    'ঐ',
    'ও',
    'ঔ',
    'ক',
    'খ',
    'গ',
    'ঘ',
    'ঙ',
    'চ',
    'ছ',
    'জ',
    'ঝ',
    'ঞ',
    'ট',
    'ঠ',
    'ড',
    'ঢ',
    'ণ',
    'ত',
    'থ',
    'দ',
    'ধ',
    'ন',
    'প',
    'ফ',
    'ব',
    'ভ',
    'ম',
    'য',
    'র',
    'ল',
    'শ',
    'ষ',
    'স',
    'হ',
    'ৎ',
    'ং',
    'ঃ',
    'ঁ',
    'া',  #  - আ কার (aa)
    'ি',  # - ই কার (i)
    'ী',  #  - ঈ কার (ii)
    'ু',  #  - উ কার (u)
    'ূ',  #  - ঊ কার (uu)
    'ৃ',  #  - ঋ কার (rri)
    'ে',  #  - এ কার (e)
    'ৈ',  #  - ঐ কার (oi)
    'ো',  #  - ও কার (o)
    'ৌ',  #  - ঔ কার (ou)
    '্',  # হসন্ত (hasanta)
    '়',  # নুক্তা (nukta)
    '।',  # দাঁড়ি (daari)
    '॥',  # ডাবল দাঁড়ি (double daari)
    '“',  # উদ্ধৃতি খোলা (opening quotation mark)
    '”',  # উদ্ধৃতি বন্ধ (closing quotation mark)
    '‘',  # একক উদ্ধৃতি খোলা (single opening quote)
    '’',  # একক উদ্ধৃতি বন্ধ (single closing quote)
    '–',  # এন ড্যাশ (en dash)
    '—',  # এম ড্যাশ (em dash)
    '৳',
    ]

conjunct_consonants = [
    "ক্ক",
    "ক্ট",
    "ক্ট্র",
    "ক্ত",
    "ক্ত্র",
    "ক্ন",
    "ক্ব",
    "ক্ম",
    "ক্য",
    "ক্র",
    "ক্ল",
    "ক্ষ",
    "ক্ষ্ণ",
    "ক্ষ্ব",
    "ক্ষ্ম",
    "ক্ষ্ম্য",
    "ক্ষ্য",
    "ক্স",
    "খ্য",
    "খ্র",
    "গ্‌ণ",
    "গ্ধ",
    "গ্ধ্য",
    "গ্ধ্র",
    "গ্ন",
    "গ্ন্য",
    "গ্ব",
    "গ্ম",
    "গ্য",
    "গ্র",
    "গ্র্য",
    "গ্ল",
    "ঘ্ন",
    "ঘ্য",
    "ঘ্র",
    "ঙ্ক",
    "ঙ্‌ক্ত",
    "ঙ্ক্য",
    "ঙ্ক্ষ",
    "ঙ্খ",
    "ঙ্খ্য",
    "ঙ্গ",
    "ঙ্গ্য",
    "ঙ্ঘ",
    "ঙ্ঘ্য",
    "ঙ্ঘ্র",
    "ঙ্ম",
    "চ্চ",
    "চ্ছ",
    "চ্ছ্ব",
    "চ্ছ্র",
    "চ্ঞ",
    "চ্ব",
    "চ্য",
    "জ্জ",
    "জ্জ্ব",
    "জ্ঝ",
    "জ্ঞ",
    "জ্ব",
    "জ্য",
    "জ্র",
    "ঞ্চ",
    "ঞ্ছ",
    "ঞ্জ",
    "ঞ্ঝ",
    "ট্ট",
    "ট্ব",
    "ট্ম",
    "ট্য",
    "ট্র",
    "ড্ড",
    "ড্ব",
    "ড্ম",
    "ড্য",
    "ড্র",
    "ড়্গ",
    "ঢ্য",
    "ঢ্র",
    "ণ্ট",
    "ণ্ঠ",
    "ণ্ঠ্য",
    "ণ্ড",
    "ণ্ড্য",
    "ণ্ড্র",
    "ণ্ঢ",
    "ণ্ণ",
    "ণ্ব",
    "ণ্ম",
    "ণ্য",
    "ত্ত",
    "ত্ত্র",
    "ত্ত্ব",
    "ত্ত্য",
    "ত্থ",
    "ত্ন",
    "ত্ব",
    "ত্ম",
    "ত্ম্য",
    "ত্য",
    "ত্র",
    "ত্র্য",
    "থ্ব",
    "থ্য",
    "থ্র",
    "দ্গ",
    "দ্ঘ",
    "দ্দ",
    "দ্দ্ব",
    "দ্ধ",
    "দ্ব",
    "দ্ভ",
    "দ্ভ্র",
    "দ্ম",
    "দ্য",
    "দ্র",
    "দ্র্য",
    "ধ্ন",
    "ধ্ব",
    "ধ্ম",
    "ধ্য",
    "ধ্র",
    "ন্ট",
    "ন্ট্র",
    "ন্ঠ",
    "ন্ড",
    "ন্ড্র",
    "ন্ত",
    "ন্ত্ব",
    "ন্ত্য",
    "ন্ত্র",
    "ন্ত্র্য",
    "ন্থ",
    "ন্থ্র",
    "ন্দ",
    "ন্দ্য",
    "ন্দ্ব",
    "ন্দ্র",
    "ন্ধ",
    "ন্ধ্য",
    "ন্ধ্র",
    "ন্ন",
    "ন্ব",
    "ন্ম",
    "ন্য",
    "প্ট",
    "প্ত",
    "প্ন",
    "প্প",
    "প্য",
    "প্র",
    "প্র্য",
    "প্ল",
    "প্স",
    "ফ্র",
    "ফ্ল",
    "ব্জ",
    "ব্দ",
    "ব্ধ",
    "ব্ব",
    "ব্য",
    "ব্র",
    "ব্ল",
    "ভ্ব",
    "ভ্য",
    "ভ্র",
    "ভ্ল",
    "ম্ন",
    "ম্প",
    "ম্প্র",
    "ম্ফ",
    "ম্ব",
    "ম্ব্র",
    "ম্ভ",
    "ম্ভ্র",
    "ম্ম",
    "ম্য",
    "ম্র",
    "ম্ল",
    "য্য",
    "র্ক",
    "র্ক্য",
    "র্গ",
    "র্গ্য",
    "র্ঘ্য",
    "র্ঙ্গ",
    "র্চ্য",
    "র্জ্য",
    "র্জ্জ",
    "র্জ্ঞ",
    "র্ণ্য",
    "র্ত্য",
    "র্থ্য",
    "র্ব্য",
    "র্ম্য",
    "র্শ্য",
    "র্ষ্য",
    "র্হ্য",
    "র্খ",
    "র্গ",
    "র্গ্র",
    "র্ঘ",
    "র্চ",
    "র্ছ",
    "র্জ",
    "র্ঝ",
    "র্ট",
    "র্ড",
    "র্ণ",
    "র্ত",
    "র্ৎ",
    "র্ত্ম",
    "র্ত্র",
    "র্থ",
    "র্দ",
    "র্দ্ব",
    "র্দ্র",
    "র্ধ",
    "র্ধ্ব",
    "র্ন",
    "র্প",
    "র্ফ",
    "র্ব",
    "র্ভ",
    "র্ম",
    "র্য",
    "র্ল",
    "র্শ",
    "র্শ্ব",
    "র্ষ",
    "র্ষ্ট",
    "র্ষ্ণ",
    "র্ষ্ণ্য",
    "র্স",
    "র্হ",
    "র্হ্য",
    "র্ঢ্য",
    "ল্ক",
    "ল্ক্য",
    "ল্গ",
    "ল্ট",
    "ল্ড",
    "ল্প",
    "ল্ফ",
    "ল্ব",
    "ল্ভ",
    "ল্ম",
    "ল্য",
    "ল্ল",
    "শ্চ",
    "শ্ছ",
    "শ্ন",
    "শ্ব",
    "শ্ম",
    "শ্য",
    "শ্র",
    "শ্ল",
    "ষ্ক",
    "ষ্ক্ব",
    "ষ্ক্র",
    "ষ্ট",
    "ষ্ট্য",
    "ষ্ট্র",
    "ষ্ঠ",
    "ষ্ঠ্য",
    "ষ্ণ",
    "ষ্ণ্ব",
    "ষ্প",
    "ষ্প্র",
    "ষ্ফ",
    "ষ্ব",
    "ষ্ম",
    "ষ্য",
    "স্ক",
    "স্ক্র",
    "স্খ",
    "স্ট",
    "স্ট্র",
    "স্ত",
    "স্ত্ব",
    "স্ত্য",
    "স্ত্র",
    "স্থ",
    "স্থ্য",
    "স্ন",
    "স্ন্য",
    "স্প",
    "স্প্র",
    "স্প্‌ল",
    "স্ফ",
    "স্ব",
    "স্ম",
    "স্য",
    "স্র",
    "স্ল",
    "হ্ণ",
    "হ্ন",
    "হ্ব",
    "হ্ম",
    "হ্য",
    "হ্র",
    "হ্ল",
]

initial_alphabets = bangla_alphabets + conjunct_consonants

def load_data(filepath='./small_data_1MB.txt'):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read().replace('\n', '')
    return text

def make_hf_tokenizer(tokenizer):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer #tokenizer_file="tokenizer.json"
    )
    return tokenizer

# this will prevent forming tokens with whitespace
# def bengali_word_iterator(text:str):
#     for line in text:
#         yield re.findall(r'\S+', line, flags=re.UNICODE)

def make_addToken(token:str, token_id:int, is_special=False):
        token = {
            "id":token_id, 
            "special":is_special,
            "content":token,
            "single_word":False,
            "lstrip":False,
            "rstrip":False,
            "normalized":False
            }
        
        return tokenizers.AddedToken(**token)

class SpBPETokenizer(SentencePieceBPETokenizer):
    def __init__(self,  forbidden_start_chars=None):
        super().__init__()
        
        self.forbidden_start_chars = forbidden_start_chars or {'ৎ', 'ং', 'ঃ'}
        self.initial_alphabets = bangla_alphabets + conjunct_consonants
        self.init_token_counter = 0
    
    def train_from_iterator(
        self,
        iterator: Iterator[str],
        vocab_size: int = 30000,
        min_frequency: int = 5,
        show_progress: bool = True,
        special_tokens: List[str] = None,
        initial_alphabet: List[str] = None,
        limit_alphabet: int = None,
    ) -> None:
        # # First, add the initial alphabet to the vocabulary
        if initial_alphabet:
            self.add_tokens(initial_alphabet)
        # if special_tokens is not None:
        #     for sp in special_tokens:
        #         token = make_addToken(token=sp, token_id=self.init_token_counter, is_special=True)
        #         self.add_tokens([token])
        #         self.init_token_counter += 1
                
        # if initial_alphabet is not None:
        #     self.initial_alphabets = initial_alphabet
            
        # for ia in self.initial_alphabets:
        #     token = make_addToken(token=ia, token_id=self.init_token_counter, is_special=False)
        #     self.add_tokens([token])
        #     self.init_token_counter += 1
            
        # tt = {"id":101,"special":False,"content":self.initial_alphabets[0],"single_word":False,"lstrip":False,"rstrip":False,"normalized":False}
    
        # tt = tokenizers.AddedToken(**tt)    
        # self.add_tokens([tt])
        
        print(sorted(self.get_vocab().items(), key=lambda x:x[1]))
        # modify the iterator for whitespace issue
        # iterator = bengali_word_iterator(text=iterator)
        
        super().train_from_iterator(
            iterator,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=[],
            limit_alphabet=limit_alphabet,
        )
        print(self.get_vocab().items())
        # remove the forbidden char so that they don't merge
        filtered_vocab = {token: score for token, score in self.get_vocab().items()
                        if not any(token.startswith(char) for char in self.forbidden_start_chars)}
        
        # Reset the vocabulary and add the filtered tokens
        self.add_tokens(list(filtered_vocab.keys()))





