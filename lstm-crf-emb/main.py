from data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word, CoNLLDataset, get_exported_dic_vectors, \
    get_exported_morph_vectors, get_exported_pos_vectors
from model import NERModel
from config import Config
import pickle
import numpy as np
# create instance of config
config = Config()

# load vocabs
vocab_words  = load_vocab(config.words_filename)
vocab_tags   = load_vocab(config.tags_filename)
vocab_chars  = load_vocab(config.chars_filename)
vocab_morphs = load_vocab(config.morphs_filename)  #morphs add
vocab_syls   = load_vocab(config.word_syl_filename)
pos_tags     = load_vocab(config.posTag_filename)  #pos tag adding----
dic_words    = load_vocab(config.word_dic_filename)  #dic add

# get processing functions
processing_word = get_processing_word(vocab_words, dic_words, vocab_chars, vocab_morphs, vocab_syls, pos_tags,
                lowercase=True, chars=config.chars, morphs=config.morphs, posflag=config.posTag,
                pos_lm=config.posLM, dic_flag=config.dic_flag)
processing_tag  = get_processing_word(vocab_tags,
                lowercase=False)
processing_pos  = get_processing_word(pos_tags = pos_tags, posflag= True,
                lowercase=True, pos_lm = True)

# get pre trained embeddings
embeddings = get_trimmed_glove_vectors(config.trimmed_filename)
dic_embeddings = get_exported_dic_vectors(config.exported_filename)
morph_embeddings = get_exported_morph_vectors(config.exported_mfilename)
syl_embeddings = get_exported_dic_vectors(config.exported_sfilename)
pos_embeddings = get_exported_pos_vectors(config.exported_pfilename)


# create dataset
dev   = CoNLLDataset(config.dev_filename, processing_word,
                    processing_tag, processing_pos, config.max_iter)
test  = CoNLLDataset(config.test_filename, processing_word,
                    processing_tag, processing_pos, config.max_iter)
train = CoNLLDataset(config.train_filename, processing_word,
                    processing_tag, processing_pos, config.max_iter)

# build model
lmwords = len(vocab_words)
lmposs = len(pos_tags)

model = NERModel(config, embeddings, dic_embeddings, pos_embeddings, syl_embeddings, morph_embeddings,
                                    ntags=len(vocab_tags), nchars=len(vocab_chars), nsyls=len(vocab_syls),
                                    nmorphs=len(vocab_morphs), nwords=lmwords, nposs= lmposs)
model.build()

# train, evaluate and interact
model.train(train, dev, vocab_tags)
model.evaluate(test, vocab_tags, test_flag=1)
#model.interactive_shell(vocab_tags, processing_word)

