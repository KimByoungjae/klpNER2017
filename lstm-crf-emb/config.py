import os
from general_utils import get_logger


class Config():
    def __init__(self):
        # directory for training outputs
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # create instance of logger
        self.logger = get_logger(self.log_path)


    # general config
    output_path = "results/crf/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"

    # embeddings
    dim = 50    #300
    dim_char = 50   #100
    dim_morph = 60
    dim_pos = 16
    dic_dim = 5
    syl_dim = 12
    glove_filename = "data/wiki/wikiCorpus_word2vector.txt".format(dim)    #wikiCorpus_word2vector.txt".format(dim)
    dic_filename = "data/dic/output.txt".format(dim)
    syl_filename = "data/dic/syllable2.txt".format(dim)
    morph_vec_filename = "data/morph/morph_vector.txt".format(dim)
    pos_vec_filename = "data/wiki/komoran_kbsnews_posW2V.txt".format(dim)
    trimmed_filename = "data/wiki.trimmed.npz".format(dim)
    exported_filename = "data/dic.exported.npz".format(dim)
    exported_mfilename = "data/morph.exported.npz".format(dim)
    exported_sfilename = "data/syl.exported.npz".format(dim)
    exported_pfilename = "data/pos.exported.npz".format(dim)

    # dataset
    dev_filename = "data/corpus/kor/dev_shuffle.txt"
    test_filename = "data/corpus/kor/2017klpner_test_anal.txt"
    train_filename = "data/corpus/kor/train_shuffle.txt"
    max_iter = None # if not None, max number of examples

    # vocab (created from dataset with build_data.py)
    words_filename = "data/words.txt"
    tags_filename = "data/tags.txt"
    chars_filename = "data/chars.txt"
    morphs_filename = "data/morphs.txt"
    posTag_filename = "data/pos.txt"
    word_dic_filename = 'data/word_dic.txt'
    word_syl_filename = "data/syl.txt"

    # training
    train_embeddings = True
    nepochs = 100
    dropout = 0.5
    batch_size = 32
    lr = 0.005  #0.001
    lr_decay = 0.92   #0.9
    nepoch_no_imprv = 5

    # model hyperparameters
    hidden_size = 150    #300
    char_hidden_size = 50    #100
    morph_hidden_size = 12
    #posTag_size = 46
    lm_size = 50
    lm_gamma = 0.1

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    crf = True # if crf, training is 1.7x slower on CPU
    chars = True # if char embedding, training is 3.5x slower on CPU
    morphs = True # if morph embedding
    posTag = True  # if posTag embedding
    posLM = False
    dic_flag = True
