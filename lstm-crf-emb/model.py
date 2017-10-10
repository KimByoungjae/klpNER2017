import numpy as np
import os
import tensorflow as tf
from data_utils import minibatches, pad_sequences, get_chunks
from general_utils import Progbar, print_sentence
import trie

class NERModel(object):
    def __init__(self, config, embeddings, dic_embeddings, pos_embeddings, syl_embeddings, morph_embeddings, ntags, nchars=None, nsyls=None, nmorphs=None, nwords=None, nposs = None):
        """
        Args:
            config: class with hyper parameters
            embeddings: np array with embeddings
            nchars: (int) size of chars vocabulary
        """
        self.config     = config
        self.embeddings = embeddings
        self.dic_embeddings = dic_embeddings
        self.syl_embeddings = syl_embeddings
        self.morph_embeddings = morph_embeddings
        self.pos_embeddings = pos_embeddings
        self.nchars     = nchars
        self.nsyls      = nsyls
        self.nmorphs    = nmorphs
        self.ntags      = ntags
        self.logger     = config.logger # now instantiated in config
        self.nwords     = nwords
        self.nposs = nposs
        self.gpu_nums   = 2
        self.cur_gpu    = 0

    def _next_device(self):
        """round robin gpu device"""
        if self.gpu_nums == 0:
            return ''
        dev = '/gpu:%d' % self.cur_gpu
        if self.gpu_nums > 1:
            self.cur_gpu = (self.cur_gpu + 1) % (self.gpu_nums - 1)
        return dev


    def add_placeholders(self):
        """
        Adds placeholders to self
        """
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                            name="word_ids")


        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                            name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                            name="char_ids")

        # shape = (batch size, max length of sentence, max length of word)
        self.morph_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                            name="morph_ids")
        self.syl_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                            name="syl_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                            name="word_lengths")
        self.morph_lengths = tf.placeholder(tf.int32, shape=[None, None],
                            name="morph_lengths")
        self.syl_lengths = tf.placeholder(tf.int32, shape=[None, None],
                            name="syl_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                            name="labels")

        # shape = (batch size, max length of sentence in batch)
        self.fw_lm_ids = tf.placeholder(tf.int32, shape=[None, None], name="fw_lm_ids")
        self.bw_lm_ids = tf.placeholder(tf.int32, shape=[None, None], name="bw_lm_ids")

        self.fw_pos_ids = tf.placeholder(tf.int32, shape=[None, None], name="fw_pos_ids")
        self.bw_pos_ids = tf.placeholder(tf.int32, shape=[None, None], name="bw_pos_ids")

        # shape = (batch size, max length of sentence in batch)
        self.posTag_ids = tf.placeholder(tf.int32, shape=[None, None], name="posTag_ids")

        self.dic_ids = tf.placeholder(tf.float32, shape=[None, None, 6], name="dic_ids")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                            name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                            name="lr")


    def get_feed_dict(self, words,fw_words, bw_words, dict_labels, labels=None, lr=None, dropout=None, test_flag=0):
        """words, fw_words, bw_words, labels, postags,  fw_postags, bw_postags
        Given some data, pad it and build a feed dictionary
        Args:
            words: list of sentences. A sentence is a list of ids of a list of words.
                A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob
        Returns:
            dict {placeholder: value}
        """
        # perform padding of the given data
        if self.config.chars and not self.config.posTag and not self.config.dic_flag and not self.config.morphs:
            char_ids, word_ids = zip(*words)
            _, fw_lm_ids = zip(*fw_words)
            _, bw_lm_ids = zip(*bw_words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            fw_lm_ids, sequence_lengths = pad_sequences(fw_lm_ids, 0)
            bw_lm_ids, sequence_lengths = pad_sequences(bw_lm_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
        elif self.config.chars and self.config.posTag: #--------adding posTag padding-----------
            if self.config.dic_flag and not self.config.morphs:
                posTag_ids, char_ids, word_ids, dic_ids = zip(*words)
                fw_postag_ids, _, fw_lm_ids, _ = zip(*fw_words)
                bw_postag_ids, _, bw_lm_ids, _ = zip(*bw_words)
                dic_ids = []
                for dict in dict_labels:
                    tmp_dic1 = []
                    tmp_dic2 = []
                    tmp_dic3 = []
                    tmp_dic4 = []
                    for d_i, d in enumerate(dict['labels1']):
                        tmp_dic1.append(dict['labels1'][d_i])
                        tmp_dic2.append(dict['labels2'][d_i])
                        tmp_dic3.append(dict['labels3'][d_i])
                        tmp_dic4.append(dict['labels4'][d_i])
                        tmp_dic5.append(dict['labels5'][d_i])

                    dic_ids.append([tmp_dic1, tmp_dic2, tmp_dic3, tmp_dic4, tmp_dic5])
            elif self.config.dic_flag and self.config.morphs:
                posTag_ids, char_ids, word_ids, dic_ids, morph_ids, syl_ids = zip(*words)
                fw_postag_ids, _, fw_lm_ids, _, _, _ = zip(*fw_words)
                bw_postag_ids, _, bw_lm_ids, _, _, _ = zip(*bw_words)

                dic_ids = []
                for dict in dict_labels:
                    tmp_dic1 = []
                    tmp_dic2 = []
                    tmp_dic3 = []
                    tmp_dic4 = []
                    tmp_dic5 = []
                    for d_i, d in enumerate(dict['labels1']):
                        tmp_dic1.append(dict['labels1'][d_i])
                        tmp_dic2.append(dict['labels2'][d_i])
                        tmp_dic3.append(dict['labels3'][d_i])
                        tmp_dic4.append(dict['labels4'][d_i])
                        tmp_dic5.append(dict['labels5'][d_i])

                    dic_ids.append([tmp_dic1, tmp_dic2, tmp_dic3, tmp_dic4, tmp_dic5])
            else:
                posTag_ids, char_ids, word_ids = zip(*words)
                fw_postag_ids, _, _, fw_lm_ids = zip(*fw_words)
                bw_postag_ids, _, _, bw_lm_ids = zip(*bw_words)


            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            fw_lm_ids, sequence_lengths = pad_sequences(fw_lm_ids, 0)
            bw_lm_ids, sequence_lengths = pad_sequences(bw_lm_ids, 0)

            # if self.config.dic_flag:
            #     dic_ids, sequence_lengths = pad_sequences(dic_ids, 0)

            if self.config.dic_flag:
                dic_ids, sequence_lengths = pad_sequences(dic_ids, pad_tok=0, nlevels=4)
                # if last_flag == False:
                dic_embeddings = np.zeros((len(word_ids),len(word_ids[0]), 6), dtype=np.float32)
                # elif last_flag == True:
                #     dic_embeddings = np.zeros((3, len(word_ids[0]), 7), dtype=np.float32)
                for batch_i, batch_dict in enumerate(dic_ids):
                    for word_i, word_dict in enumerate(batch_dict[0]):

                        dic_embeddings[batch_i][word_i][int(batch_dict[0][word_i])] = 1
                        dic_embeddings[batch_i][word_i][int(batch_dict[1][word_i])] = 1
                        dic_embeddings[batch_i][word_i][int(batch_dict[2][word_i])] = 1
                        dic_embeddings[batch_i][word_i][int(batch_dict[3][word_i])] = 1
                        dic_embeddings[batch_i][word_i][int(batch_dict[4][word_i])] = 1

                dic_ids = dic_embeddings
            if self.config.morphs:
                morph_ids, morph_lengths = pad_sequences(morph_ids, pad_tok=0, nlevels=2)
                syl_ids, syl_lengths = pad_sequences(syl_ids, pad_tok=0, nlevels=2)
            fw_postag_ids, sequence_lengths = pad_sequences(fw_postag_ids, 0)
            bw_postag_ids, sequence_lengths = pad_sequences(bw_postag_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
            posTag_ids, _ = pad_sequences(posTag_ids, 0)
        else:
            word_ids, morph_ids = zip(*words)
            fw_lm_ids, _ = zip(*fw_words)
            bw_lm_ids, _ = zip(*bw_words)
            morph_ids, morph_lengths = pad_sequences(morph_ids, pad_tok=0, nlevels=2)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            fw_lm_ids, sequence_lengths= pad_sequences(fw_lm_ids, 0)
            bw_lm_ids, sequence_lenghts= pad_sequences(bw_lm_ids, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.fw_lm_ids: fw_lm_ids,
            self.bw_lm_ids: bw_lm_ids,
            self.sequence_lengths: sequence_lengths
        }
        #if test_flag == 1:
        #    print(word_ids)

        if self.config.posLM:
            feed[self.fw_pos_ids] = fw_postag_ids
            feed[self.bw_pos_ids] = bw_postag_ids


        if self.config.chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if self.config.morphs:
            feed[self.morph_ids] = morph_ids
            feed[self.morph_lengths] = morph_lengths
            feed[self.syl_ids] = syl_ids
            feed[self.syl_lengths] = syl_lengths


        if self.config.dic_flag:
            feed[self.dic_ids] = dic_ids

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout
        #postag add--------------------------
        if self.config.posTag:
            feed[self.posTag_ids] = posTag_ids

        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        """
        Adds word embeddings to self
        """

        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings, name="_word_embeddings", dtype=tf.float32,
                                    trainable=self.config.train_embeddings)
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids,
                    name="word_embeddings")
        with tf.variable_scope("dics"):
            if self.config.dic_flag:
                # _dic_embeddings = tf.Variable(self.dic_embeddings, name="_dic_embeddings", dtype = tf.float32,
                #                         trainable=False)
                # dic_embeddings = tf.nn.embedding_lookup(_dic_embeddings, self.dic_ids, name="dic_embeddings")
                # word_embeddings = tf.concat([word_embeddings, dic_embeddings], axis=-1)
                word_embeddings = tf.concat([word_embeddings, self.dic_ids], axis=-1)

        with tf.variable_scope("chars"):
            if self.config.chars:
                # get embeddings matrix
                _char_embeddings = tf.get_variable(name="_char_embeddings", dtype=tf.float32,
                                                        shape=[self.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids,
                                                             name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[-1])
                # bi lstm on chars
                # need 2 instances of cells since tf 1.1
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_hidden_size,
                                                      state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_hidden_size,
                                                      state_is_tuple=True)

                _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                                          cell_bw, char_embeddings,
                                                                                          sequence_length=word_lengths,
                                                                                          dtype=tf.float32)

                output = tf.concat([output_fw, output_bw], axis=-1)
                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output, shape=[-1, s[1], 2 * self.config.char_hidden_size])

                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        with tf.variable_scope("morphs"):
            if self.config.morphs:
                # get embeddings matrix
                #_morph_embeddings = tf.Variable(self.morph_embeddings, name = "_morph_embeddings", dtype=tf.float32,
                #                                          trainable = self.config.train_embeddings)
                #morph_embeddings = tf.nn.embedding_lookup(_morph_embeddings, self.morph_ids,
                #                                             name="morph_embeddings")
                _syl_embeddings = tf.Variable(self.syl_embeddings, name = "_syl_embeddings", dtype = tf.float32,
                                                          trainable = False)
                syl_embeddings = tf.nn.embedding_lookup(_syl_embeddings, self.syl_ids, name="syl_embeddings")

                #morph_embeddings = tf.concat([morph_embeddings, syl_embeddings], axis=-1)
                #self.config.dim_morph
                morph_embeddings = syl_embeddings

                s = tf.shape(morph_embeddings)
                morph_embeddings = tf.reshape(morph_embeddings, shape=[-1, s[-2], (self.config.syl_dim)])
                morph_lengths = tf.reshape(self.morph_lengths, shape=[-1])

                cell_fw = tf.contrib.rnn.LSTMCell(self.config.morph_hidden_size,
                                                   state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.morph_hidden_size,
                                                   state_is_tuple=True)

                _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                                      cell_bw, morph_embeddings,
                                                                                      sequence_length=morph_lengths,
                                                                                      dtype=tf.float32)
                output = tf.concat([output_fw, output_bw], axis=-1)
                output = tf.reshape(output, shape=[-1, s[1], 2 * self.config.morph_hidden_size])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)


        #-------------pos tag one-hot encoding added to embedding-------------------------
        with tf.variable_scope("posTag"):
            if self.config.posTag:
                #get pos embeddings matrix
                _pos_embeddings = tf.Variable(self.pos_embeddings, name="_pos_embeddings", dtype=tf.float32)
                pos_embeddings = tf.nn.embedding_lookup(_pos_embeddings, self.posTag_ids, name="pos_embeddings")
                #output = tf.one_hot(indices = self.posTag_ids, depth = self.config.posTag_size)
                #output = tf.reshape(output, shape=[-1, s[1], self.config.posTag_size])

                word_embeddings = tf.concat([word_embeddings, pos_embeddings], axis=-1)
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        """
        Adds logits to self
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                    cell_bw, self.word_embeddings, sequence_length=self.sequence_lengths,
                    dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

            fw_M = tf.get_variable("fw_M", shape=[self.config.hidden_size, self.config.lm_size], dtype=tf.float32)
            fw_b = tf.get_variable("fw_b", shape=[self.config.lm_size], dtype=tf.float32, initializer = tf.zeros_initializer())

            bw_M = tf.get_variable("bw_M", shape=[self.config.hidden_size, self.config.lm_size], dtype=tf.float32)
            bw_b = tf.get_variable("bw_b", shape=[self.config.lm_size], dtype=tf.float32, initializer=tf.zeros_initializer())

            if self.config.posLM:
                fw_POS = tf.get_variable("fw_POS", shape=[self.config.hidden_size, self.config.lm_size], dtype=tf.float32)
                fw_pos = tf.get_variable("fw_pos", shape=[self.config.lm_size], dtype=tf.float32, initializer=tf.zeros_initializer())

                bw_POS = tf.get_variable("bw_POS", shape=[self.config.hidden_size, self.config.lm_size], dtype=tf.float32)
                bw_pos = tf.get_variable("bw_pos", shape=[self.config.lm_size], dtype=tf.float32, initializer=tf.zeros_initializer())

            fw_m = tf.nn.relu(tf.matmul(tf.reshape(output_fw,[-1,self.config.hidden_size]), fw_M) + fw_b, name='fw_pred')
            bw_m = tf.nn.relu(tf.matmul(tf.reshape(output_bw,[-1,self.config.hidden_size]), bw_M) + bw_b, name='bw_pred')

            if self.config.posLM:
                fw_pos_m = tf.tanh(tf.matmul(tf.reshape(output_fw, [-1, self.config.hidden_size]), fw_POS) + fw_pos,
                               name='fw_pos_m')
                bw_pos_m = tf.tanh(tf.matmul(tf.reshape(output_bw, [-1, self.config.hidden_size]), bw_POS) + bw_pos,
                               name='bw_pos_m')


        with tf.variable_scope("proj"):
            W = tf.get_variable("W", shape=[2*self.config.hidden_size, self.ntags],
                        dtype=tf.float32)

            b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32, initializer=tf.zeros_initializer())


            Wd = tf.get_variable("Wd", shape=[2*self.config.hidden_size, 2*self.config.hidden_size],
                                                dtype=tf.float32)
            bd = tf.get_variable("bd", shape=[2*self.config.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer())

            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size])
            output = tf.nn.relu(tf.matmul(output, Wd) + bd, name="tanh_d_layer")
            #output = tf.matmul(output, Wd) + bd

            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, ntime_steps, self.ntags])


            fw_lm_W = tf.get_variable("fw_lm_W", shape=[self.config.lm_size, self.nwords], dtype=tf.float32)
            fw_lm_b = tf.get_variable("fw_lm_b", shape=[self.nwords], dtype=tf.float32, initializer=tf.zeros_initializer())

            bw_lm_W = tf.get_variable("bw_lm_W", shape=[self.config.lm_size, self.nwords], dtype=tf.float32)
            bw_lm_b = tf.get_variable("bw_lm_b", shape=[self.nwords], dtype=tf.float32, initializer=tf.zeros_initializer())

            if self.config.posLM:
                fw_pos_W = tf.get_variable("fw_pos_W", shape=[self.config.lm_size, self.nposs], dtype=tf.float32)
                fw_pos_w = tf.get_variable("fw_pos_w", shape=[self.nposs], dtype=tf.float32, initializer=tf.zeros_initializer())

                bw_pos_W = tf.get_variable("bw_pos_W", shape=[self.config.lm_size, self.nposs], dtype=tf.float32)
                bw_pos_w = tf.get_variable("bw_pos_w", shape=[self.nposs], dtype=tf.float32, initializer=tf.zeros_initializer())

            fw_pred = tf.matmul(fw_m, fw_lm_W) + fw_lm_b
            bw_pred = tf.matmul(bw_m, bw_lm_W) + bw_lm_b

            if self.config.posLM:
                fw_pos_pred = tf.matmul(fw_pos_m, fw_pos_W) + fw_pos_w
                bw_pos_pred = tf.matmul(bw_pos_m, bw_pos_W) + bw_pos_w

            self.fw_logits = tf.reshape(fw_pred, [-1, ntime_steps, self.nwords])
            self.bw_logits = tf.reshape(bw_pred, [-1, ntime_steps, self.nwords])

            if self.config.posLM:
                self.fw_pos_logits = tf.reshape(fw_pos_pred, [-1, ntime_steps, self.nposs])
                self.bw_pos_logits = tf.reshape(bw_pos_pred, [-1, ntime_steps, self.nposs])


    def add_pred_op(self):
        """
        Adds labels_pred to self
        """
        if not self.config.crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)


    def add_loss_op(self):
        """
        Adds loss to self
        """
        if self.config.crf:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.labels, self.sequence_lengths)
            self.tag_loss = tf.reduce_mean(-log_likelihood)

            fw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.fw_logits, labels = tf.squeeze(self.fw_lm_ids))
            bw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.bw_logits, labels = tf.squeeze(self.bw_lm_ids))

            if self.config.posLM:
                fw_pos_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.fw_pos_logits,
                                                                         labels=tf.squeeze(self.fw_pos_ids))
                bw_pos_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.bw_pos_logits,
                                                                         labels=tf.squeeze(self.bw_pos_ids))

            self.fw_loss = tf.reduce_mean(fw_loss)
            self.bw_loss = tf.reduce_mean(bw_loss)

            if self.config.posLM:
                self.fw_pos_loss = tf.reduce_mean(fw_pos_loss)
                self.bw_pos_loss = tf.reduce_mean(bw_pos_loss)

            if self.config.posLM:
                self.loss = self.tag_loss + self.config.lm_gamma * (self.fw_loss + self.bw_loss + self.fw_pos_loss + self.bw_pos_loss)
            else:
                self.loss = self.tag_loss + self.config.lm_gamma * (self.fw_loss + self.bw_loss)

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("tag_loss", self.tag_loss)
        tf.summary.scalar("fw_loss", self.fw_loss)
        tf.summary.scalar("bw_loss", self.bw_loss)
        if self.config.posLM:
            tf.summary.scalar("fw_loss", self.fw_pos_loss)
            tf.summary.scalar("bw_loss", self.bw_pos_loss)


    def add_train_op(self):
        """
        Add train_op to self
        """
        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.loss)


    def add_init_op(self):
        self.init = tf.global_variables_initializer()


    def add_summary(self, sess):
        # tensorboard stuff
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, sess.graph)


    def build(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op()
        self.add_init_op()


    def predict_batch(self, sess, words, fw_words, bw_words,dict_labels, labels, print_line, test_flag):

        """
        Args:
            sess: a tensorflow session
            words: list of sentences
        Returns:
            labels_pred: list of labels for each sentence
            sequence_length
        """

        fd, sequence_lengths = self.get_feed_dict(words, fw_words, bw_words, dict_labels, labels, dropout=1.0, test_flag=1)
        # fd, sequence_lengths = self.get_feed_dict(words, fw_words, bw_words, labels, dropout= 0.5)

        if self.config.crf:
            viterbi_sequences = []
            logits, transition_params = sess.run([self.logits, self.transition_params],
                    feed_dict=fd)

            #if test_flag == 1:
            #    print(print_line)
            # iterate over the sentences
            for logit, sequence_length in zip(logits, sequence_lengths):
                # keep only the valid time steps
                logit = logit[:sequence_length]
                viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                                logit, transition_params)
                viterbi_sequences += [viterbi_sequence]
            #print(viterbi_sequences)
            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def run_epoch(self, sess, train, dev, tags, epoch):
        """
        Performs one complete pass over the train set and evaluate on dev
        Args:
            sess: tensorflow session
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            tags: {tag: index} dictionary
            epoch: (int) number of the epoch
        """

        #trie setting
        self.lis1 = []
        self.lis2 = []
        self.lis3 = []
        self.lis4 = []
        self.lis5 = []

        trie.gazette(self.lis1, "data/dic/gazette.txt")
        trie.gazette(self.lis2, "data/dic/thres3.txt")
        trie.gazette_DTTI(self.lis3, "data/dic/DT_analysis.txt")
        trie.gazette_DTTI(self.lis4, "data/dic/TI_analysis.txt")
        trie.gazette(self.lis5, "data/dic/wiki_PS.txt")


        nbatches = (len(train) + self.config.batch_size - 1) // self.config.batch_size
        prog = Progbar(target=nbatches)
        for i, (words, fw_words, bw_words, labels, postags, sentences, _) in enumerate(minibatches(train, self.config.batch_size)):

            dict_labels = self.dict_trie(sentences)

            fd, _ = self.get_feed_dict(words, fw_words, bw_words, dict_labels, labels, self.config.lr, self.config.dropout, test_flag=0)

            _, train_loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        acc, f1, p, r = self.run_evaluate(sess, dev, tags, test_flag=0)
        self.logger.info("- dev acc {:04.2f} - f1 {:04.2f} - p {:04.2f} - r {:04.2f}".format(100*acc, 100*f1, 100*p, 100*r))
        return acc, f1


    def run_evaluate(self, sess, test, tags, test_flag):
        """
        Evaluates performance on test set
        Args:
            sess: tensorflow session
            test: dataset that yields tuple of sentences, tags
            tags: {tag: index} dictionary
        Returns:
            accuracy
            f1 score
        """

        #trie setting
        self.lis1 = []
        self.lis2 = []
        self.lis3 = []
        self.lis4 = []
        self.lis5 = []

        trie.gazette(self.lis1,"data/dic/gazette.txt")
        trie.gazette(self.lis2,"data/dic/thres3.txt")
        trie.gazette_DTTI(self.lis3, "data/dic/DT_analysis.txt")
        trie.gazette_DTTI(self.lis4, "data/dic/TI_analysis.txt")
        trie.gazette(self.lis5,"data/dic/wiki_PS.txt")
        fresult = open("results/result.txt", "w")

        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        # for i, (words, fw_words, bw_words, labels, postags) in enumerate(minibatches(train, self.config.batch_size)):
        #     fd, _ = self.get_feed_dict(words, fw_words, bw_words, labels, self.config.lr, self.config.dropout)

        total_chunks = []

        for words, fw_words, bw_words, labels, postags, sentences, print_line in minibatches(test, self.config.batch_size):

            dict_labels = self.dict_trie(sentences)

            labels_pred, sequence_lengths = self.predict_batch(sess, words, fw_words, bw_words, dict_labels,labels, print_line, test_flag)


            line_num=0
            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a==b for (a, b) in zip(lab, lab_pred)]
                lab_chunks = set(get_chunks(lab, tags))
                lab_pred_chunks = set(get_chunks(lab_pred, tags))
                #-------------------------------------------------------
                #print(lab_pred_chunks)
                if test_flag == 1:
                    #print(print_line[line_num][1])
                    fresult.write(print_line[line_num][0]+'\n')
                    #fresult.write(print_line[line_num][1]+'\n')
                    print_chunks = list(lab_pred_chunks)
                    print_chunks.sort(key=lambda chunks: chunks[1])
                    #print(print_chunks)
                    for tag, start, end in print_chunks:
                        print_tag = ''
                        if tag.decode() == 'B_PS':
                            print_tag = 'PS'
                        elif tag.decode() == 'B_LC':
                            print_tag = 'LC'
                        elif tag.decode() == 'B_DT':
                            print_tag = 'DT'
                        elif tag.decode() == 'B_TI':
                            print_tag = 'TI'
                        elif tag.decode() == 'B_OG':
                            print_tag = 'OG'
                        else:
                            print_tag = tag.decode()
                        #print(print_tag+'\t'+str(start)+'\t'+str(end)+'\t'+print_line[line_num][start+2].split()[1])
                        fresult.write(print_line[line_num][start+1].split()[1]+'\t'+print_tag+'\n')
                    #print("")
                    fresult.write('\n')
                    line_num = line_num + 1
                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        #self.print_results(total_chunks)

        return acc, f1, p, r

    def print_results(self, total_chunks):
        fresult = open("data/result.txt", 'w')
        fword   = open("data/words.txt", 'rb')
        ftag    = open("data/tags.txt", 'rb')



    def train(self, train, dev, tags):
        """
        Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            tags: {tag: index} dictionary
        """
        best_score = 0
        saver = tf.train.Saver()
        # for early stopping
        nepoch_no_imprv = 0
        session_conf = tf.ConfigProto(allow_soft_placement=True)
        session_conf.gpu_options.allow_growth = True

        # tf.identity(variables_to_restore[0], name="pre_trained_char_embed")
        with tf.Session(config=session_conf) as sess:

            sess.run(self.init)
            # tensorboard
            self.add_summary(sess)
            for epoch in range(self.config.nepochs):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))

                acc, f1 = self.run_epoch(sess, train, dev, tags, epoch)

                # decay learning rate
                self.config.lr *= self.config.lr_decay

                # early stopping and saving best parameters
                if f1 >= best_score:
                    nepoch_no_imprv = 0
                    if not os.path.exists(self.config.model_output):
                        os.makedirs(self.config.model_output)
                    saver.save(sess, self.config.model_output)
                    best_score = f1
                    self.logger.info("- new best score!")

                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                        self.logger.info("- early stopping {} epochs without improvement".format(
                                        nepoch_no_imprv))
                        break


    def evaluate(self, test, tags, test_flag):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.logger.info("Testing model over test set")
            saver.restore(sess, self.config.model_output)
            acc, f1, p, r = self.run_evaluate(sess, test, tags, test_flag)
            self.logger.info("- test acc {:04.2f} - f1 {:04.2f} - p {:04.2f} - r {:04.2f}".format(100*acc, 100*f1, 100*p, 100*r))


    def interactive_shell(self, tags, processing_word):
        idx_to_tag = {idx: tag for tag, idx in tags.items()}
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.config.model_output)
            self.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")
            while True:
                try:
                    try:
                        # for python 2
                        sentence = raw_input("input> ")
                    except NameError:
                        # for python 3
                        sentence = input("input> ")

                    words_raw = sentence.strip().split(" ")

                    if words_raw == ["exit"]:
                        break

                    words = [processing_word(w) for w in words_raw]

                    if type(words[0]) == tuple:
                        words = zip(*words)

                    pred_ids, _ = self.predict_batch(sess, [words])

                    preds = [idx_to_tag[idx] for idx in list(pred_ids[0])]
                    print_sentence(self.logger, {"x": words_raw, "y": preds})

                except Exception:
                    pass



    def dict_trie(self, sentences):

        #PS, OG, LC ,DT, TI, O
        # 1, 2,  3,  4,  5,  0

        # trie to words
        labels1 = []
        labels2 = []
        labels3 = []
        labels4 = []
        labels5 = []

        return_dict = []

        for sentence in sentences:
            tmp_dict = {}
            new_setence = []
            labels1 = len(sentence) * [0.0]
            labels2 = len(sentence) * [0.0]
            labels3 = len(sentence) * [0.0]
            labels4 = len(sentence) * [0.0]
            labels5 = len(sentence) * [0.0]

            for j in sentence:

                if j == u'월요일':
                    new_setence.append(u'u')
                elif j == u'화요일':
                    new_setence.append(u'u')
                elif j == u'수요일':
                    new_setence.append(u'u')
                elif j == u'목요일':
                    new_setence.append(u'u')
                elif j == u'금요일':
                    new_setence.append(u'u')
                elif j == u'토요일':
                    new_setence.append(u'u')
                elif j == u'일요일':
                    new_setence.append(u'u')
                elif j == u'화':
                    new_setence.append(u'y')
                elif j == u'수':
                    new_setence.append(u'y')
                elif j == u'목':
                    new_setence.append(u'y')
                elif j == u'금':
                    new_setence.append(u'y')
                elif j == u'토':
                    new_setence.append(u'y')
                elif j.isdigit() == True:
                    new_setence.append(u'p')
                elif j.isdigit() == False:
                    new_setence.append(j)

            trie.find_trie(sentence, self.lis1, labels1)
            trie.find_trie(sentence, self.lis2, labels2)
            trie.find_trie(new_setence, self.lis3, labels3)
            trie.find_trie(new_setence, self.lis4, labels4)
            trie.find_trie(sentence, self.lis5, labels5)

            tmp_dict['labels1'] = labels1
            tmp_dict['labels2'] = labels2
            tmp_dict['labels3'] = labels3
            tmp_dict['labels4'] = labels4
            tmp_dict['labels5'] = labels5

            return_dict.append(tmp_dict)

        return return_dict