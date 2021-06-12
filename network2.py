# coding=utf-8
import numpy as np
import random
import time
import os
import sys
from datetime import datetime
from scipy import signal
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
import threading
from PyQt5.QtWidgets import *
from utils import epoch_buffer, batch_data, flatten, sample_arr, filter, down_sample, print_n_samples_each_class
from eeg_ui3 import MainDialogImgBW
# from inlet import *
from inlet import save_data

filter_buffer = []
pre_labels = []
sample_rate = 100


# output :network
def build_firstPart_model(input_var, keep_prob_=0.5):
    # List to store the output of each CNNs，一个小的滤波器和一个大的滤波器
    output_conns = []

    # CNNs with small filter size at the first layer #

    # Convolution
    network = tf.layers.conv1d(inputs=input_var, filters=64, kernel_size=50, strides=6,
                               padding='same', activation=tf.nn.relu)

    network = tf.layers.max_pooling1d(inputs=network, pool_size=8, strides=8, padding='same')

    # Dropout
    network = tf.nn.dropout(network, keep_prob_)

    # Convolution
    network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=8, strides=1,
                               padding='same', activation=tf.nn.relu)  # 一维卷积，padding有三种模式

    network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=8, strides=1,
                               padding='same', activation=tf.nn.relu)
    network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=8, strides=1,
                               padding='same', activation=tf.nn.relu)

    # Max pooling
    network = tf.layers.max_pooling1d(inputs=network, pool_size=4, strides=4, padding='same')

    # Flatten Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
    network = flatten(name="flat1", input_var=network)

    output_conns.append(network)

    # CNNs with large filter size at the first layer #

    # Convolution
    network = tf.layers.conv1d(inputs=input_var, filters=64, kernel_size=400, strides=50,
                               padding='same', activation=tf.nn.relu)

    network = tf.layers.max_pooling1d(inputs=network, pool_size=4, strides=4, padding='same')

    # Dropout
    network = tf.nn.dropout(network, keep_prob_)

    # Convolution
    network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=6, strides=1,
                               padding='same', activation=tf.nn.relu)

    network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=6, strides=1,
                               padding='same', activation=tf.nn.relu)
    network = tf.layers.conv1d(inputs=network, filters=128, kernel_size=6, strides=1,
                               padding='same', activation=tf.nn.relu)

    # Max pooling
    network = tf.layers.max_pooling1d(inputs=network, pool_size=2, strides=2, padding='same')

    # Flatten
    network = flatten(name="flat2", input_var=network)

    output_conns.append(network)

    # Concat
    network = tf.concat(output_conns, 1, name="concat1")

    # Dropout
    network = tf.nn.dropout(network, keep_prob_)

    return network


# input: keep_prob dropout的概率 char2numY {'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, 'W': 0, '<SOD>': 5, '<EOD>': 6}
# output: logits, pred_outputs, _final_state
def build_network(hparams, char2numY, inputs, dec_inputs, keep_prob_=0.5, ):
    if hparams.akara2017 is True:
        _inputs = tf.reshape(inputs, [-1, hparams.input_depth, 1])
        network = build_firstPart_model(_inputs, keep_prob_)

        shape = network.get_shape().as_list()
        data_input_embed = tf.reshape(network, (-1, hparams.max_time_step, shape[1]))

    # Embedding layers
    with tf.variable_scope("embeddin") as embedding_scope:
        decoder_embedding = tf.Variable(tf.random_uniform((len(char2numY), hparams.embed_size), -1.0, 1.0),
                                        name='dec_embedding')  # +1 to consider <EOD>
        decoder_emb_inputs = tf.nn.embedding_lookup(decoder_embedding, dec_inputs)

    with tf.variable_scope("encoding") as encoding_scope:
        if not hparams.bidirectional:

            # Regular approach with LSTM units
            # encoder_cell = tf.contrib.rnn.LSTMCell(hparams.num_units)
            # encoder_cell = tf.nn.rnn_cell.MultiRNNCell([encoder_cell] * hparams.lstm_layers)
            def lstm_cell():
                lstm = tf.contrib.rnn.LSTMCell(hparams.num_units)
                return lstm

            encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(hparams.lstm_layers)])
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, inputs=data_input_embed, dtype=tf.float32)

        else:

            # Using a bidirectional LSTM architecture instead
            # enc_fw_cell = tf.contrib.rnn.LSTMCell(hparams.num_units)
            # enc_bw_cell = tf.contrib.rnn.LSTMCell(hparams.num_units)

            def lstm_cell():
                lstm = tf.contrib.rnn.LSTMCell(hparams.num_units)
                return lstm

            stacked_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(hparams.lstm_layers)],
                                                          state_is_tuple=True)
            stacked_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(hparams.lstm_layers)],
                                                          state_is_tuple=True)

            ((enc_fw_out, enc_bw_out), (enc_fw_final, enc_bw_final)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=stacked_cell_fw,
                cell_bw=stacked_cell_bw,
                inputs=data_input_embed,
                dtype=tf.float32)
            encoder_final_state = []
            for layer in range(hparams.lstm_layers):
                enc_fin_c = tf.concat((enc_fw_final[layer].c, enc_bw_final[layer].c), 1)
                enc_fin_h = tf.concat((enc_fw_final[layer].h, enc_bw_final[layer].h), 1)
                encoder_final_state.append(tf.contrib.rnn.LSTMStateTuple(c=enc_fin_c, h=enc_fin_h))

            encoder_state = tuple(encoder_final_state)
            encoder_outputs = tf.concat((enc_fw_out, enc_bw_out), 2)

    with tf.variable_scope("decoding") as decoding_scope:

        output_layer = Dense(len(char2numY), use_bias=False)
        decoder_lengths = np.ones((hparams.batch_size), dtype=np.int32) * (hparams.max_time_step + 1)
        training_helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inputs, decoder_lengths)

        if not hparams.bidirectional:
            # decoder_cell = tf.contrib.rnn.LSTMCell(hparams.num_units)
            def lstm_cell():
                lstm = tf.contrib.rnn.LSTMCell(hparams.num_units)
                return lstm

            decoder_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(hparams.lstm_layers)])

        else:
            # decoder_cell = tf.contrib.rnn.LSTMCell(2 * hparams.num_units)
            def lstm_cell():
                lstm = tf.contrib.rnn.LSTMCell(2 * hparams.num_units)
                return lstm

            decoder_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(hparams.lstm_layers)])

        if hparams.use_attention:
            # Create an attention mechanism
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                hparams.num_units * 2 if hparams.bidirectional else hparams.num_units, encoder_outputs,
                memory_sequence_length=None)

            decoder_cells = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cells, attention_mechanism,
                attention_layer_size=hparams.attention_size, alignment_history=True)

            encoder_state = decoder_cells.zero_state(hparams.batch_size, tf.float32).clone(cell_state=encoder_state)

        # Basic Decoder and decode
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cells, training_helper, encoder_state,
                                                  output_layer=output_layer)

        dec_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                               impute_finished=True)

        # dec_outputs, _ = tf.nn.dynamic_rnn(decoder_cell, inputs=decoder_emb_inputs, initial_state=encoder_state)

    logits = dec_outputs.rnn_output  # logits是输入softmax之前的层的，是未进入softmax的概率，就是未归一化的概率

    # Inference
    start_tokens = tf.fill([hparams.batch_size], char2numY['<SOD>'])
    end_token = char2numY['<EOD>']
    if not hparams.use_beamsearch_decode:  # beam search只在预测的时候需要。训练的时候因为知道正确答案，并不需要再进行这个搜索。

        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            decoder_embedding,
            start_tokens, end_token)

        # Inference Decoder
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cells, inference_helper, encoder_state,
            output_layer=output_layer)
    else:

        encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=hparams.beam_width)
        decoder_initial_state = decoder_cells.zero_state(hparams.batch_size * hparams.beam_width, tf.float32).clone(
            cell_state=encoder_state)

        inference_decoder = beam_search_decoder.BeamSearchDecoder(cell=decoder_cells,
                                                                  embedding=decoder_embedding,
                                                                  start_tokens=start_tokens,
                                                                  end_token=end_token,
                                                                  initial_state=decoder_initial_state,
                                                                  beam_width=hparams.beam_width,
                                                                  output_layer=output_layer)

    # Dynamic decoding
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        inference_decoder, impute_finished=False, maximum_iterations=hparams.output_max_length)
    pred_outputs = outputs.sample_id
    if hparams.use_beamsearch_decode:
        # [batch_size, max_time_step, beam_width]
        pred_outputs = pred_outputs[0]
    return logits, pred_outputs, _final_state


random.seed(
    654)  # to make have the same training set and test set each time the code is run, we use a fixed random seed


# output: logits, pred_outputs, loss, optimizer, dec_states
# input: char2numY {'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4, 'W': 0, '<SOD>': 5, '<EOD>': 6}
def build_whole_model(hparams, char2numY, inputs, targets, dec_inputs, keep_prob_):
    # logits = build_network(inputs,dec_inputs=dec_inputs)
    logits, pred_outputs, dec_states = build_network(hparams, char2numY, inputs, dec_inputs, keep_prob_)
    decoder_prediction = tf.argmax(logits, 2)  # 针对传入函数的axis参数,去选取array中相对应轴元素值大的索引

    # optimization operation
    with tf.name_scope("optimization"):
        # Loss function
        vars = tf.trainable_variables()
        beta = 0.001
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * beta

        # class_ratio = [0.1,0.4, 0.1, 0.1, 0.1, 0.1,0.1]
        # class_weight = tf.constant(class_ratio)
        # weighted_logits = tf.multiply(logits, class_weight)

        loss_is = []
        for i in range(logits.get_shape().as_list()[-1]):
            class_fill_targets = tf.fill(tf.shape(targets), i)
            # tf.equal()逐个元素进行判断，如果相等就是True，不相等，就是False
            # tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换
            weights_i = tf.cast(tf.equal(targets, class_fill_targets), "float")
            loss_is.append(tf.contrib.seq2seq.sequence_loss(logits, targets, weights_i, average_across_batch=False))

        loss = tf.reduce_sum(loss_is, axis=0)

        # loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([hparams.batch_size, hparams.max_time_step+1])) #+1 is because of the <EOD> token

        # Optimizer
        loss = tf.reduce_mean(loss) + lossL2
        # optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

    return logits, pred_outputs, loss, dec_states


def run_program(hparams, FLAGS):
    classes = FLAGS.classes  # default ['W', 'N1', 'N2', 'N3', 'REM']

    print(str(datetime.now()))

    # output：y_pred
    def evaluate_model(hparams, X):
        # X.shape(1,1,3000)
        # 这里是预测结果 pred_outputs_ → _y_pred
        # start_time = time.time()
        pred_outputs_ = sess.run(pred_outputs, feed_dict={inputs: X, keep_prob_: 1.0})
        # end_time = time.time()
        # print("the predict time is: ", end_time - start_time,)  # type(pred_outputs_) ndarray
        pred_outputs_ = pred_outputs_[:, :hparams.max_time_step]  # remove the last prediction <EOD>
        _y_pred = pred_outputs_.flatten()
        # print("y_pred: ", _y_pred, type(_y_pred), _y_pred.tolist())  # array([0],dtype=int32), np.array, [0]
        return _y_pred, _y_pred.tolist()

    # 预测
    # preprocessing
    char2numY = dict(zip(classes, range(len(classes))))  # zip打包为元组的列表[('w',0),('N1',1)……]
    # <SOD> is a token to show start of decoding  and <EOD> is a token to indicate end of decoding
    char2numY['<SOD>'] = len(char2numY)
    char2numY['<EOD>'] = len(char2numY)
    # Placeholders
    inputs = tf.placeholder(tf.float32, [1, hparams.max_time_step, hparams.input_depth], name='inputs')
    targets = tf.placeholder(tf.int32, (None, None), 'targets')
    dec_inputs = tf.placeholder(tf.int32, (None, None), 'decoder_inputs')
    keep_prob_ = tf.placeholder(tf.float32, name='keep')
    # model
    logits, pred_outputs, loss, dec_states = build_whole_model(hparams, char2numY, inputs, targets, dec_inputs,
                                                               keep_prob_)
    sess = tf.Session()
    saver = tf.train.Saver()

    ckpt_name = "model_fold{:02d}.ckpt".format(3)
    ckpt_name = os.path.join(FLAGS.checkpoint_dir, ckpt_name)
    # load the parameter checkpoints-seq2seq-sleep-EDF/model_fold03.ckpt
    saver.restore(sess, ckpt_name)

    global pre_labels, filter_buffer  # 主函数部分
    i = 0  # 数据集移动index
    j = 0  # the index of npz files
    start_timestamp = time.time()
    while True:
        if epoch_buffer.get_raw_data_state(i):
            raw_data = epoch_buffer.get_raw_data(i)  # filtered raw data type=list
            i = i + 1
            filter_buffer = filter_buffer + raw_data  # save dict x
            # down sample: 1.self-defined method 2.signal.resample
            # data = down_sample(raw_data)
            data = signal.resample(raw_data, 3000)  # type=array
            # normalize each 30s sample such that each has zero mean and unit variance
            tmp_data = np.reshape(data, (1, 3000))
            tmp_data = (tmp_data - np.expand_dims(tmp_data.mean(axis=1), axis=1)) / np.expand_dims(tmp_data.std(axis=1),
                                                                                                   axis=1)
            x = np.reshape(tmp_data, (1, 1, 3000))  # predict: input the normalized data (1,1,3000)
            # 替换为预测输出，y_pred, y_list = output
            y_pred, y_list = evaluate_model(hparams, x)  # type list

            pre_labels = pre_labels + y_list  # save dict y
            print_n_samples_each_class(y_pred, classes)
            epoch_buffer.set_label(y_list)
            if time.time() - start_timestamp >= 600:  # 开始保存文件
                save_dict = {
                    'x': filter_buffer,
                    'y': pre_labels,
                    'fs': sample_rate
                }
                filename = "{:02d}".format(j) + "eeg_data.npz"
                if (os.path.exists('results') == False):
                    os.mkdir('results')
                save_file = os.path.join("./results", filename)
                np.savez(save_file, **save_dict)
                print("save success!")
                j = j + 1


def main(args=None):
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('output_dir', 'outputs_2013/outputs_eeg_fpz_cz',
                               """Directory where to save trained models """
                               """and outputs.""")
    tf.app.flags.DEFINE_list('classes', ['W', 'N1', 'N2', 'N3', 'REM'], """classes""")
    tf.app.flags.DEFINE_string('checkpoint_dir', './ckpt',
                               """Directory to save checkpoints""")
    # hyperparameters
    hparams = tf.contrib.training.HParams(
        epochs=120,  # 300
        batch_size=1,  # 10 batch size = 1一次跑一个结果
        num_units=128,
        embed_size=10,
        input_depth=3000,
        n_channels=100,
        bidirectional=False,
        use_attention=True,
        lstm_layers=2,
        attention_size=64,
        beam_width=4,
        use_beamsearch_decode=False,
        max_time_step=1,  # 5 3 second best 10# 40 # 100 这里就设置为 1 即可
        output_max_length=1 + 2,  # max_time_step +1
        akara2017=True,
        test_step=5,  # each 10 epochs
    )
    # classes = ['W', 'N1', 'N2', 'N3', 'REM']
    run_program(hparams, FLAGS)


if __name__ == "__main__":
    thread_pre = threading.Thread(target=tf.app.run)
    thread_save = threading.Thread(target=save_data)
    thread_pre.start()
    thread_save.start()
    app = QApplication(sys.argv)
    main = MainDialogImgBW()
    main.show()
    sys.exit(app.exec_())
