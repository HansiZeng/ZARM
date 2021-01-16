from tqdm import tqdm

class Vocab():
    def __init__(self, special_tokens, list_of_str, preprocessor=None, tokenizer=None, stop_words=None, max_size=50000):
        self.special_tokens = special_tokens
        self.list_of_str = list_of_str
        self._preprocessor = preprocessor
        self._tokenizer = tokenizer
        self._stop_words = stop_words
        self._max_size = max_size
        
        self._token2id = {}
        self._id2token = {}
        self._token_freqs = {}

        self._oov = set()
        
        if special_tokens != None:
            for tok in special_tokens:
                cur_id = self.__len__()
                self._token2id[tok] = cur_id 
                self._id2token[cur_id] = tok 
                
    def _build_from_list_of_str(self):
        list_of_str = self.list_of_str
        
        if self._preprocessor != None:
            list_of_str = list(map(self._preprocessor, list_of_str))
        else:
            raise ValueError("need preprocessor")
        
        if self._tokenizer != None:
            list_of_tokens = list(map(self._tokenizer, list_of_str))
        else:
            list_of_tokens = list(map(str.split, list_of_str))
        
        self._list_of_tokens = list_of_tokens
        self._build_from_list_of_tokens()
        
    def _build_from_list_of_tokens(self):
        list_of_tokens = self._list_of_tokens
        flatten_tokens = [tok for tokens in list_of_tokens for tok in tokens]

        # count token freqs 
        for tok in flatten_tokens:
            if tok in self._token_freqs:
                self._token_freqs[tok] += 1
            else:
                self._token_freqs[tok] = 1
        
        # words to oov 
        self._token_freqs = {k:v for k,v in sorted(self._token_freqs.items(), key=lambda x: x[1], reverse=True)}

        ranked_tokens = [t for t in self._token_freqs]
        self._oov = ranked_tokens[self._max_size:]
        print(f"oov size: {len(self._oov)}")

        for tok in tqdm(self._token_freqs):
            if tok in self._oov or tok in self._stop_words:
                continue
            if tok not in self._token2id:
                cur_id = self.__len__()
                self._token2id[tok] = cur_id
                self._id2token[cur_id] = tok 
                    
    def get_list_of_tokens(self):
        return self._list_of_tokens                      
        
    def build(self):
        self._build_from_list_of_str()
        
    def __len__(self):
        return len(self._token2id)

class Indexlizer():
    def __init__(self, list_of_str, special_tokens=["<pad>", "<unk>"], preprocessor=None, tokenizer=None, stop_words=[],
                pad_token="<pad>", max_len=50):
        self._special_tokens = special_tokens
        self._list_of_str = list_of_str
        self._preprocessor = preprocessor 
        self._tokenizer = tokenizer
        self._stop_words = stop_words
        self._max_len = max_len

        self._pad_token = special_tokens[0]
        self._unk_token = special_tokens[1]
        
        self._vocab = Vocab(self._special_tokens, self._list_of_str, self._preprocessor, self._tokenizer,
                            self._stop_words)
        self._vocab.build()
        
        self._token2id = self._vocab._token2id
        self._id2token = self._vocab._id2token
        
        self.review_lengths = []
        
        assert pad_token == self._id2token[0]
    
    def _pad_and_truncate_sequence(self, x):
        if len(x) > self._max_len:
            x = x[:self._max_len]
        else:
            pad_id = self._token2id[self._pad_token]
            pad_len = self._max_len - len(x)
            x = x + [pad_id] * pad_len
        
        return x
    
    def _list_of_str_to_list_of_tokens(self, list_of_str):
        if self._preprocessor != None:
            list_of_str = list(map(self._preprocessor, list_of_str))
        else:
            raise ValueError("need preprocessor")
        
        if self._tokenizer != None:
            list_of_tokens = list(map(self._tokenizer, list_of_str))
        else:
            list_of_tokens = list(map(str.split, list_of_str))
            
        return list_of_tokens
    
    def transform_idxed_review(self, idxed_review, ignore_pad=True):
        assert self._vocab != None 
        pad_id = self._token2id[self._pad_token]

        out_tokens = []
        for tokid in idxed_review:
            if tokid == pad_id:
                continue
            else:
                tok = self._id2token[tokid] if tokid in self._id2token else self._unk_token
                out_tokens.append(tok)
        return out_tokens

    def transform(self, reviews):
        """
        reviews: list of str
        """
        review_ids = []

        list_of_tokens = self._list_of_str_to_list_of_tokens(reviews)
        unk_id = self._token2id[self._unk_token]

        # transform `list of str` to `dict(str, int)` to speed up 
        oov = {w:i for i, w in enumerate(self._vocab._oov)}
        sws = {w: i for i, w in enumerate(self._vocab._stop_words)}
        
        for tokens in tqdm(list_of_tokens):
            token_ids = [] 
            for tok in tokens:
                if tok in oov:
                    token_ids.append(unk_id)
                elif tok in sws:
                    continue
                elif tok in self._token2id:
                    token_ids.append(self._token2id[tok])
                else:
                    raise ValueError(f"{tok} not in oov, stopwords, vocab.")

            self.review_lengths.append(len(token_ids)) # statistics
            padded_token_ids = self._pad_and_truncate_sequence(token_ids)
         
            review_ids.append(padded_token_ids)
            
        return review_ids