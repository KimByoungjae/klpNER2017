import tensorflow as tf
import pickle


char_W = tf.get_variable("embeddings",
              [58, 11])

init = tf.global_variables_initializer()
sess = tf.Session()
saver = tf.train.Saver()
# saver.restore(sess, "/home/nlp908/data/cm/workspace/lstm-char-cnn-tensorflow/checkpoint/wiki_64/LSTMTDNN")

saver.restore(sess, "/home/nlp908/data/cm/workspace/word2vec-tensorflow/checkpoint/pos_embd-25")
sess.run(init)
char_embed = char_W.eval(sess)

pos2idx = {}
with open('/home/nlp908/data/cm/workspace/word2vec-tensorflow/pos_vocab.txt', 'rb') as f:
    lines = f.readlines()
    for line in lines:
        pos, idx = line.split(b'\t')
        pos2idx[pos] = idx

# f = open('/home/nlp908/data/cm/workspace/lstm-char-cnn-tensorflow/data/wiki/vocab.pkl', 'rb')
# vocab = pickle.load(f)
# a=1

# dict = {'char2idx':vocab[3], 'char_embed':char_embed}

dict = {'pos2idx' : pos2idx}
with open('char_embed.pkl', 'wb') as f:
    pickle.dump(dict,f)
