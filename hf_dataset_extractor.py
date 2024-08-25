import os
import time
from datasets import load_dataset


class LargeDatasetTextExtractor:
    def __init__(self, dataset_name, split=None, language:str='bn', output_path:str='output.txt', batch_size=1000):
        self.dataset_name = dataset_name
        self.split = split
        self.language = language
        self.output_path = output_path
        self.batch_size = batch_size

    def extract_text(self):
        
        dataset_kwargs = {}
        if self.split:
            dataset_kwargs['split'] = self.split
        if self.language:
            dataset_kwargs['name'] = self.language
            

        dataset = load_dataset(path=self.dataset_name,streaming=True, **dataset_kwargs )
        st = time.time()
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for i, batch in enumerate(dataset.iter(batch_size=self.batch_size)):
                for item in batch['text']:
                    f.write(item + '\n')
                
                if (i + 1) % self.batch_size * 10 == 0:
                    print(f"Processed {(i + 1) * self.batch_size} rows")

        print(f"Finished extracting text to {self.output_path}. Total time taken for extraction: {(time.time() - st)/60} m")

if __name__ == "__main__":
    extractor = LargeDatasetTextExtractor(
        dataset_name="allenai/c4", 
        split='train',
        language='bn',
        output_path="/storage2/llm_data/data_all_text/AllTextData/c4_New.txt", 
        batch_size=1000)
    extractor.extract_text()