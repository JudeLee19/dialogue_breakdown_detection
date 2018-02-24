import numpy as np


class BowEmbed():
    
    def __init__(self, train_data, test_data, max_vocab_size):
        self.data = train_data + test_data
        self.max_vocab_size = max_vocab_size
        self.vocab = self.get_vocab()
        self.vocab_size = len(self.vocab)
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_vocab(self):
        vocab = {}
        for dialog in self.data:
            for each in dialog:
                utterance_list = each[0]
                for utterance in utterance_list:
                    utterance = utterance.lower()
                    for word in utterance.split(' '):
                        if "'" in word:
                            pre_word = word.split("'")[0]
                            after_word = "'" + word.split("'")[1]
                            if pre_word in vocab:
                                vocab[pre_word] += 1
                            else:
                                vocab[pre_word] = 1
                            
                            if after_word in vocab:
                                vocab[after_word] += 1
                            else:
                                vocab[after_word] = 1
                        else:
                            if word in vocab:
                                vocab[word] += 1
                            else:
                                vocab[word] = 1
        vocab = sorted(vocab, key=vocab.get, reverse=True)
        
        max_vocab_size = self.max_vocab_size
        if len(vocab) > max_vocab_size:
            vocab = vocab[:max_vocab_size]
        return vocab
    
    def embed_utterance(self, utterance):
        bow = np.zeros([self.vocab_size], dtype=np.int32)
        utterance = utterance.lower()
        for word in utterance.split(' '):
            if "'" in word:
                pre_word = word.split("'")[0]
                after_word = "'" + word.split("'")[1]
                if pre_word in self.vocab:
                    idx = self.vocab.index(pre_word)
                    bow[idx] += 1
                if after_word in self.vocab:
                    idx = self.vocab.index(after_word)
                    bow[idx] += 1
            else:
                if word in self.vocab:
                    idx = self.vocab.index(word)
                    bow[idx] += 1
        return bow