class Data():
    
    def __init__(self, train_file_name, test_file_name):
        self.train_set = self.prepare_data(train_file_name)
        self.test_set = self.prepare_data(test_file_name)
    
    def prepare_data(self, file_name):
        # get dialogs from file
        dialog_set = []
        train_set = []
        with open(file_name, 'r', encoding='utf-8') as f_r:
            for line in f_r:
                line = line.strip()
                if line == '<EOD>':
                    dialog_set.append(train_set)
                    train_set = []
                else:
                    line_split = line.split('||')
                    user_utter = line_split[0]
                    system_utter = line_split[1]
                    ground_label = line_split[2]
                    concat_utter_list = [user_utter, system_utter]
                    train_set.append((concat_utter_list, ground_label))
        return dialog_set
    