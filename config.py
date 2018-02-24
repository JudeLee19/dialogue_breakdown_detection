import os


class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        # self.logger = get_logger(self.log_path)
    
    output_path = 'results/word2vec_lstm_1003/'
    model_output = output_path + 'model.weights/'
    log_path = output_path + "log.txt"

    lr = 0.01
    lr_decay = 0.9
    clip = -1
    nepoch_no_imprv = 3
    reload = True
    
    num_epochs = 30
    batch_size = 10
    
    embed_method = 'word2vec'
    
    data_root_dir = '/root/jude/data/dbdc3/data/'
    # file name lists for training
    word2vec_filename = data_root_dir + 'word2vec/wiki_en_model'
    
    train_filename = data_root_dir + 'new_train_test_dir/train_dataset'
    dev_filename = data_root_dir + 'new_train_test_dir/dev_dataset'
    test_filename = data_root_dir + 'new_train_test_dir/test_dataset'
    cate_mapping_dict = data_root_dir + 'cate_mapping_dict'
