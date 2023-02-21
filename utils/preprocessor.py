import random
from more_itertools import chunked
from transformers import BertTokenizer

class Preprocessor():
    def __init__(self, config, id2doc):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name)
        self.id2doc = id2doc
        
    def process_all(self, query):
        doc_ids, inputs = self.concat_all(query)
        chunck_size = self.config.batch_size * 64
        input_chuncks = list(chunked(inputs, chunck_size))
        doc_id_chuncks = list(chunked(doc_ids, chunck_size))
        tensor_chuncks = []
        for input_chunck in input_chuncks:
            tensor_chunck = self.text2tensor(input_chunck)
            tensor_chuncks.append(tensor_chunck)
        return doc_id_chuncks, tensor_chuncks
    
    def process_partial(self, query, targets):
        inputs = self.concat_partial(query, targets)
        encoded_inputs = self.text2tensor(inputs)
        return encoded_inputs
        
    def process(self, queries, targets):
        positive_inputs, negative_inputs = self.concat(queries, targets)
        encoded_positive_inputs = self.text2tensor(positive_inputs)
        encoded_negative_inputs = self.text2tensor(negative_inputs)
        return encoded_positive_inputs, encoded_negative_inputs
    
    def text2tensor(self, texts):
        encoded = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="max_length",
            max_length = self.config.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.config.device)
        attention_mask = encoded["attention_mask"].to(self.config.device)
        return (input_ids, attention_mask)
    
    def sample(self, targets):
        """
        input all targets with relevance score
        output sampled positive and negative sample documents
        """
        positive_doc = self.id2doc[targets[0][0]]
        positive_score = targets[0][1]
        negative_docs = [id for id, target in enumerate(targets) if target[1] < positive_score]
        if len(negative_docs) > 0:
            negative_doc = self.id2doc[targets[random.choice(negative_docs)][0]]
        else:
            negative_doc = random.choice(list(self.id2doc.values()))
        return positive_doc, negative_doc
    
    def concat_all(self, query):
        doc_ids = []
        inputs = []
        for doc_id, doc in self.id2doc.items():
            doc_ids.append(doc_id)
            inputs.append(f"{query}[SEP]{doc}")
        return doc_ids, inputs
    
    def concat_partial(self, query, targets):
        inputs = []
        for doc_id, _ in targets:
            inputs.append(f"{query}[SEP]{self.id2doc[doc_id]}")
        return inputs
    
    def concat(self, queries, targets):
        positive_inputs = []
        negative_inputs = []
        for query, target in zip(queries, targets):
            positive_doc, negative_doc = self.sample(target)
            positive_input = f"{query}[SEP]{positive_doc}"
            negative_input = f"{query}[SEP]{negative_doc}"
            positive_inputs.append(positive_input)
            negative_inputs.append(negative_input)
        return positive_inputs, negative_inputs
    
    
    
        
if __name__ == "__main__":
    class Config():
        def __init__(self) -> None:
            self.max_length = 512
            self.model_name = "bert-base-multilingual-cased"
    config = Config()
    preprocseeor = Preprocessor(config)
    texts = [" ".join(["a"]*1024)]
    print(len(texts[0]))
    preprocseeor.text2tensor(texts)