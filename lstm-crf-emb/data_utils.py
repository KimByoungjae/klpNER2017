import numpy as np
import os
from config import Config
from kor_char import word_to_char

config = Config()
UNK = b"$UNK$"
NUM = b"$NUM$"
NONE = b"O"

class CoNLLDataset(object):
    """
    Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags
    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```
    """
    def __init__(self, filename, processing_word=None, processing_tag=None, processing_pos=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield
        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.processing_pos = processing_pos
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename, 'rb') as f:
            words, tags, pos = [], [], [] #pos add
            sentence = []
            print_line = []
            for line in f:
                line = line.strip()
                if (line == b'' or line.startswith(b"-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags, pos, sentence, print_line
                        words, tags, pos = [], [], [] # pos add
                        sentence = []
                        print_line = []
                else:
                    #word, tag = line.split()
                    """input related code"""
                    print_line.append(line.decode())
                    word_tok = line.split()
                    if word_tok[0] == b';':
                        #print_line.append(line)
                        continue
                    elif word_tok[0][0:1] == b'$':
                        #print_line.append(line)
                        continue
                    if config.posTag:
                        #word = word_tok[0]
                        #pos_tok = word_tok[0].split(b'/')[-1]
                        #print(word_tok)
                        word = word_tok[1] + b'/' + word_tok[2]
                        pos_tok = word_tok[2]
                    else:
                        #word = word_tok[0].split(b'/')[0]
                        word = word_tok[1]
                    #tag = word_tok[1]
                    tag = word_tok[3]

                    #sentence += [word_tok[0].decode().split('/')[0]]
                    sentence += [word_tok[1].decode()]
                    """change code"""
                    if self.processing_word is not None:
                        # print (self.processing_word(word))
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)

                    words += [word]
                    tags += [tag]

                    if config.posTag:
                        pos_tok = pos_tok.lower()
                        pos += [pos_tok]


    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

"""input form change"""
def get_vocabs(datasets):
    """
    Args:
        datasets: a list of dataset objects
    Return:
        a set of all the words in the dataset
    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    vocab_pos = set()
    for dataset in datasets:
        for words, tags, poss, _, _ in dataset: #adding pos--
            words = [word.decode('utf-8') for word in words]
            tags = [tag.decode('utf-8') for tag in tags]
            poss = [pos.decode('utf-8') for pos in poss]
            vocab_words.update(words)
            vocab_tags.update(tags)
            vocab_pos.update(poss) #--


    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags, vocab_pos  #--

def get_char_vocab(dataset):
    """
    Args:
        dataset: a iterator yielding tuples (sentence, tags)
    Returns:
        a set of all the characters in the dataset
    """
    vocab_char = set()
    vocab_morph = set()
    for words, _, _, _, _ in dataset:
        words = [word.decode('utf-8') for word in words]
        for word in words:
    #        vocab_morph.update(word) # add morph
            chars = word_to_char(word)
            for char in chars:
                vocab_char.update(char) # add char
    return vocab_char


def get_glove_vocab(filename):  #loading embedding( cnn embedding)
    """
    Args:
        filename: path to the glove vectors
    """
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split()[0]
            if not config.posTag:
                word = word.split('/')[0]
                word = word.lower()
            else:
                word = word.lower()
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab

def get_dic_vocab(filename, ds_flag):  #add get_dic_vocab
    print("building dic vocab...")
    vocab = set()
    first_line_pass = 0
    if ds_flag == 0:
        first_line_pass = 1

    with open(filename, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            #print(line)
            if first_line_pass == 0:
                first_line_pass = 1
                continue
            word = line.split('\t')[0]
            word = word.lower()
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab

def get_morph_vocab(filename): #add get_morph_vocab
    print("building morph vocab...")
    vocab = set()
    #i = 0
    with open(filename, 'rb') as f:
        for line in f:
            #i = i+1
            line = line.decode('utf-8')
            word = line.split('\t')
            #if len(word) == 1:
            #    print(i)
            #print(str(i)+' : '+word[0])
            word = word[0]
            word = word.lower()
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """
    Writes a vocab to a file

    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with open(filename, 'rb') as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx

    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """
    Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split('\t') #when file change fix
            word = line[0]
            if not config.posTag:
                word = word.split('/')[0]
                word = word.lower()
            else:
                word = word.lower()
            word = word.encode('utf-8')
            #print(word)
            #print("123123")
            #print(vocab)
            line_b = line[1].split()
            #line_b = line[1:]
            embedding = [float(x) for x in line_b[0:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)

def export_dic_vectors(vocab, dic_filename, exported_filename, dim):
    embeddings = np.zeros([len(vocab), dim])
    first_line_pass = 0
    with open(dic_filename) as f:
        for line in f:
            if first_line_pass == 0:
                first_line_pass = 1
                continue
            line = line.strip().split('\t')
            word = line[0]
            word = word.lower()
            word = word.encode('utf-8')
            line_b = line[1:]
            for i in range(0,5):
                if float(line_b[i]) >= 1:
                    line_b[i] = 1
                else:
                    line_b[i] = 0
            embedding = [float(x) for x in line_b[0:5]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    np.savez_compressed(exported_filename, embeddings=embeddings)

def export_syl_vectors(vocab, syl_filename, exported_filename, dim):
    embeddings = np.zeros([len(vocab), dim])
    first_line_pass = 0
    with open(syl_filename) as f:
        for line in f:
            if first_line_pass == 0:
                first_line_pass = 1
                continue
            line = line.split('\t')
            word = line[0]
            word = word.lower()
            word = word.encode('utf-8')
            line_b = line[1:]
            embedding = [float(x) for x in line_b[0:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    np.savez_compressed(exported_filename, embeddings=embeddings)

def export_morph_vectors(vocab, morph_vec_filename, exported_mfilename, dim):
    embeddings = np.zeros([len(vocab), dim])
    with open(morph_vec_filename) as f:
        for line in f:
            line = line.split('\t')
            word = line[0]
            word = word.lower()
            word = word.encode('utf-8')
            line_b = line[1].split()
            embedding = [float(x) for x in line_b[0:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    np.savez_compressed(exported_mfilename, embeddings=embeddings)

def export_pos_vectors(vocab, pos_vec_filename, exported_pfilename, dim):
    embeddings = np.zeros([len(vocab), dim])
    with open(pos_vec_filename) as f:
        for line in f:
            line = line.split()
            word = line[0]
            word = word.lower()
            word = word.encode('utf-8')
            line_b = line[1:]
            embedding = [float(x) for x in line_b[0:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    np.savez_compressed(exported_pfilename, embeddings=embeddings)

def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    with np.load(filename) as data:
        return data["embeddings"]

def get_exported_dic_vectors(filename):  #add for dic
    with np.load(filename) as data:
        return data["embeddings"]

def get_exported_morph_vectors(filename):
    with np.load(filename) as data:
        return data["embeddings"]

def get_exported_pos_vectors(filename):
    with np.load(filename) as data:
        return data["embeddings"]

def get_processing_word(vocab_words=None, dic_words=None, vocab_chars=None, vocab_morphs=None, vocab_syls=None, pos_tags=None,
                    lowercase=False, chars=False, morphs=False, posflag=False, pos_lm = False, dic_flag = False):
    """
    Args:
        vocab: dict[word] = idx
    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            uword = word.decode('utf-8')
            uword = word_to_char(uword)
            for char in uword:
                # ignore chars out of vocabulary
                char = char.encode('utf-8')
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 0-1. get morphs of words
        if vocab_morphs is not None and morphs == True:
            morph_ids = []
            syl_ids = []
            uword = word.decode('utf-8')
            for morph in uword:
                morph = morph.encode('utf-8')
                if morph in vocab_morphs:
                    morph_ids += [vocab_morphs[morph]]
                else:
                    morph_ids += [vocab_morphs[UNK]]
                if morph in vocab_syls:
                    syl_ids += [vocab_syls[morph]]
                else:
                    syl_ids += [vocab_syls[UNK]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        """input related code"""
        # 1-2. get id of pos
        if pos_tags is not None and posflag == True: #input data processing
            #pos_ids = []
            #print("pos")
            if b'/' in word:
                pos_tag = word.split(b'/')[-1]
            if word in pos_tags:  #only use pos embedding test
                pos_ids = pos_tags[word]
            elif pos_tag in pos_tags:
                pos_ids = pos_tags[pos_tag]
            else:
                pos_ids = pos_tags[UNK]
        #print("pos end")

        #1-3. get id of dic
        if dic_words is not None and dic_flag == True:
            if b'/' in word:
                dic_w = word.split(b'/')[0]
            if dic_w in dic_words:
                dic_ids = dic_words[dic_w]
            else:
                dic_ids = dic_words[UNK]

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                word = vocab_words[UNK]

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True and posflag == False and dic_flag == False and morphs == False:
            # print("1")
            return char_ids, word
        elif pos_tags is not None and vocab_chars is not None and chars == True and posflag == True and dic_flag == False and morphs == False:
            #print("2")
            return pos_ids, char_ids, word
        elif pos_tags is not None and vocab_chars is not None and chars == True and posflag == True and dic_flag == True and morphs == False:
            return pos_ids, char_ids, word, dic_ids
        elif pos_tags is not None and vocab_chars is not None and chars == True and posflag == True and dic_flag == True and morphs == True:
            return pos_ids, char_ids, word, dic_ids, morph_ids, syl_ids
        elif morphs == True:
            return word, morph_ids
        else:
            #print("3")
            return word

    return f

def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length

def _pad_sequences2(sequences, pad_tok, max_length, nlevels = 1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    if nlevels == 1:
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
            sequence_padded +=  [seq_]
            sequence_length += [min(len(seq), max_length)]

        return sequence_padded, sequence_length

    elif nlevels == 4:
        for seq in sequences:
            tmp_padded = []
            for s in seq:
                s = list(s)
                s = s[:max_length] + [pad_tok]*max(max_length - len(s), 0)
                tmp_padded +=  [s]
                sequence_length += [min(len(s), max_length)]

            sequence_padded.append(tmp_padded)

        return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok]*max_length_word,
                                            max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)

    elif nlevels == 3:
        max_length_word = config.max_word_length
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok]*max_length_word,
                                            max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)

    elif nlevels == 4:
        # max_length = max_lens
        max_length = max(map(lambda x: len(x), sequences[0]))
        sequence_padded, sequence_length = _pad_sequences2(sequences,
                                                          pad_tok, max_length, nlevels=nlevels)
    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Returns:
        list of tuples
    """
    x_batch, fw_x_batch, bw_x_batch, y_batch, z_batch = [], [], [], [], []
    sentences = []
    print_line = []

    for (x, y, z, sentence, print_l) in data:
        if len(x_batch) == minibatch_size:

            yield x_batch, fw_x_batch, bw_x_batch, y_batch, z_batch, sentences, print_line
            x_batch, fw_x_batch, bw_x_batch, y_batch, z_batch = [], [], [], [], []
            sentences = []
            print_line = []

        chars_batch = []

        #rearrange word forward by one index
        fw_x = []
        fw_x = x[1:]
        fw_x.append(x[0])

        #rearrange word backward by one index
        bw_x = []
        bw_x = x[0:-1]
        bw_x.insert(0,x[-1])

        if type(x[0]) == tuple:
            x = zip(*x)
        if type(fw_x[0]) == tuple:
            fw_x = zip(*fw_x)
        if type(bw_x[0]) == tuple:
            bw_x = zip(*bw_x)

        x_batch    += [x]
        fw_x_batch += [fw_x]
        bw_x_batch += [bw_x]
        y_batch    += [y]
        z_batch    += [z]

        sentences += [sentence]

        print_line += [print_l]

    if len(x_batch) != 0:
        yield x_batch, fw_x_batch, bw_x_batch, y_batch, z_batch, sentences, print_line


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split(b'-')[0]
    tag_type = tag_name.split(b'-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks
