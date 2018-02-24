from config import Config
from utterance_embed import UtteranceEmbed
from data_process import Data
from memory_net import Memory_net
import random
import joblib
import sys
import numpy as np
import string
from bow import BowEmbed
printable = set(string.printable)


class Trainer():

    def __init__(self):
        config = Config()
        self.emb = UtteranceEmbed(config.word2vec_filename)
        train_dataset = Data(config.train_filename, config.test_filename).train_set
        random.shuffle(train_dataset)
        self.train_dataset = train_dataset[:361]
        self.test_dataset = train_dataset[361:]
        self.cate_mapping_dict = joblib.load(config.cate_mapping_dict)
        self.bow_embed = BowEmbed(self.train_dataset, self.train_dataset, 200)
        
        nb_hidden = 128
        # obs_size = self.emb.dim + self.bow_embed.get_vocab_size()
        obs_size = self.emb.dim
        
        self.memory_net = Memory_net(obs_size, nb_hidden)
        
    def train(self):
        print(len(self.train_dataset))
        print('\n:: training started\n')
        print('bow get_vocab_size :', self.bow_embed.get_vocab_size())
        epochs = 100
        for j in range(epochs):
            num_tr_examples = len(self.train_dataset)
            loss = 0.
            
            for i, dialog in enumerate(self.train_dataset):
                loss += self.dialog_train(dialog)
                sys.stdout.write('\r{}.[{}/{}]'.format(j + 1, i + 1, num_tr_examples))
            
            self.evaluate()
            
    def dialog_train(self, dialog):
        """
        실험 두가지 가능.
        1. hidden 이용할 때 user만 이용하는것
        2. hidden 이용할 때 user, system 둘 다 이용하는것.
        """

        self.memory_net.reset_state()
        loss = 0
        for (u, s), label in dialog:
            u = u.encode("ascii", errors="ignore").decode()
            s = s.encode("ascii", errors="ignore").decode()
            
            utter_emb = self.emb.embed_utterance(u + ' ' + s)
            # u_emb = self.emb.embed_utterance(u)
            s_emb = self.emb.embed_utterance(s)
            
            class_label = self.cate_mapping_dict[label]
            
            loss += self.memory_net.train_step(utter_emb, s_emb, class_label)
            
        return loss / len(dialog)
    
    def evaluate(self):
        
        accuracy = 0
        len_dialog = 0
        correct_examples = 0
        
        for dialog in self.test_dataset:
            self.memory_net.reset_state()
            
            for (u, s), label in dialog:
                
                class_label = self.cate_mapping_dict[label]

                utter_emb = self.emb.embed_utterance(u + ' ' + s)
                # u_emb = self.emb.embed_utterance(u)
                s_emb = self.emb.embed_utterance(s)

                prediction = self.memory_net.forward(utter_emb, s_emb)

                correct_examples += int(prediction == class_label)

            len_dialog += len(dialog)
        accuracy += correct_examples / len_dialog
        #
        print('=============================')
        print('Accuracy')
        print(accuracy * 100)
        print('=============================\n')
        

if __name__ == '__main__':
    
    trainer = Trainer()
    trainer.train()