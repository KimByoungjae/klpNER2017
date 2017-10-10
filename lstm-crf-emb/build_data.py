from config import Config
from data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word, get_dic_vocab, \
    export_dic_vectors, export_morph_vectors, export_syl_vectors, get_morph_vocab, \
    export_pos_vectors


def build_data(config):
    """
    Procedure to build data

    Args:
        config: defines attributes needed in the function
    Returns:
        creates vocab files from the datasets
        creates a npz embedding file from trimmed glove vectors
    """
    processing_word = get_processing_word(lowercase=True)

    # Generators
    dev   = CoNLLDataset(config.dev_filename, processing_word)
    test  = CoNLLDataset(config.test_filename, processing_word)
    train = CoNLLDataset(config.train_filename, processing_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags, vocab_pos = get_vocabs([train, dev, test])  #pos adding-----
    vocab_glove = get_glove_vocab(config.glove_filename)
    vocab_dic = get_dic_vocab(config.dic_filename, 1)  #add dic vector get
    vocab_syl = get_dic_vocab(config.syl_filename, 1)  #add syl vector
    vocab_morph = get_morph_vocab(config.morph_vec_filename) #morph vector get

    vocab = vocab_words & vocab_glove
    vocab.add(UNK.decode('utf-8'))
    vocab.add(NUM.decode('utf-8'))

    word_dic = vocab_dic  #add dic
    word_dic.add(UNK.decode('utf-8'))
    word_dic.add(NUM.decode('utf-8'))

    word_syl = vocab_syl  #add syl
    word_syl.add(UNK.decode('utf-8'))
    word_syl.add(NUM.decode('utf-8'))

    word_morph = vocab_morph # add morph
    word_morph.add(UNK.decode('utf-8'))
    word_morph.add(NUM.decode('utf-8'))

    vocab_pos.add(UNK.decode('utf-8'))

    # Save vocab
    write_vocab(vocab, config.words_filename)
    write_vocab(vocab_tags, config.tags_filename)
    write_vocab(word_dic, config.word_dic_filename)  #add dic
    write_vocab(word_syl, config.word_syl_filename)  #add syl
    write_vocab(word_morph, config.morphs_filename)  #add morph
    write_vocab(vocab_pos, config.posTag_filename)  #add pos

    # Trim GloVe Vectors(pretrain vector)
    vocab = load_vocab(config.words_filename)
    export_trimmed_glove_vectors(vocab, config.glove_filename,
                                config.trimmed_filename, config.dim)
    word_dic = load_vocab(config.word_dic_filename)  #dic add
    export_dic_vectors(word_dic, config.dic_filename, config.exported_filename, config.dic_dim)
    word_syl = load_vocab(config.word_syl_filename)  #syl add
    export_syl_vectors(word_syl, config.syl_filename, config.exported_sfilename, config.syl_dim)
    word_morph = load_vocab(config.morphs_filename)  #morph add
    export_morph_vectors(word_morph, config.morph_vec_filename, config.exported_mfilename, config.dim_morph)
    vocab_pos = load_vocab(config.posTag_filename)   #pos add
    export_pos_vectors(vocab_pos, config.pos_vec_filename, config.exported_pfilename, config.dim_pos)


    # Build and save char vocab, morph vocab
    train = CoNLLDataset(config.train_filename)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.chars_filename)

if __name__ == "__main__":
    config = Config()
    build_data(config)
