from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors, trainers, models
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import Whitespace, Metaspace
from tokenizers.processors import TemplateProcessing
from tokenizers.implementations import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast

class BnSpBpeTokenizer:
    def __init__(self):
        self.tokenizer = SentencePieceBPETokenizer(
            # vocab="/home/virus_proton/Projects/P_Projects/LLM_Mastery/Tokenizer_train/Trails/tokenizer_jul_13/test_vocab.json",
            add_prefix_space=True,
            unk_token='<unk>',
            replacement='_'
        )

        self.tokenizer.normalizer = normalizers.Sequence([NFC()])

        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Metaspace()])

    def train(self, files:list=None, vocab_size:int=1000, min_frequency:int=2, special_tokens:list=None, predefined_tokens=None):
        self.predefined_words = predefined_tokens if predefined_tokens else []

        self.tokenizer.train(
            files=files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens
        )
        
        # Add predefined words to the tokenizer vocabulary
        self.tokenizer.add_tokens(self.predefined_words)
        
        self.hf_tokenizer = self.get_hf_tokenizer(model_max_length=512)
        
        return self.hf_tokenizer
    
    def get_hf_tokenizer(self, model_max_length:int=512):
        self.hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer, 
            # tokenizer_file="tokenizer.json",
            model_max_length=model_max_length,
        )
        self.hf_tokenizer.bos_token = "<s>"
        self.hf_tokenizer.bos_token_id = self.tokenizer.token_to_id("<s>")
        self.hf_tokenizer.pad_token = "<pad>"
        self.hf_tokenizer.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.hf_tokenizer.eos_token = "</s>"
        self.hf_tokenizer.eos_token_id = self.tokenizer.token_to_id("</s>")
        self.hf_tokenizer.unk_token = "<unk>"
        self.hf_tokenizer.unk_token_id = self.tokenizer.token_to_id("<unk>")
        self.hf_tokenizer.mask_token = "<mask>"
        self.hf_tokenizer.mask_token_id = self.tokenizer.token_to_id("<mask>")

        bos_token = self.hf_tokenizer.special_tokens_map['bos_token']
        eos_token = self.hf_tokenizer.special_tokens_map['eos_token']
        # Create a TemplateProcessing post-processor
        post_processor = TemplateProcessing(
            single=f"{bos_token} $A {eos_token}",
            special_tokens=[
                (bos_token, self.hf_tokenizer.convert_tokens_to_ids(bos_token)),
                (eos_token, self.hf_tokenizer.convert_tokens_to_ids(eos_token)),
            ],
        )
        self.hf_tokenizer.backend_tokenizer.post_processor = post_processor
        
        return self.hf_tokenizer


if __name__=="__main__":
    _tokenizer = BnSpBpeTokenizer()
    hf_tokenizer = _tokenizer.train(
        files=['demo.txt'],
        vocab_size=52_000,
        min_frequency=2,
        special_tokens=["<unk>", "<pad>", "<s>", "</s>", "<mask>"],
        predefined_tokens=None
    )
    hf_tokenizer.save_pretrained('/home/virus_proton/Projects/P_Projects/LLM_Mastery/Tokenizer_train/Trails/hf_tokenizer')