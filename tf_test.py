import tensorflow_addons as tfa
import tensorflow as tf

bs = 4
cell = tf.keras.layers.LSTMCell(3)
mechanism = tfa.seq2seq.LuongAttention(units=3)
cell = tfa.seq2seq.AttentionWrapper(cell, mechanism)
layer = tf.keras.layers.RNN(cell)



def test(masked=False):
    seq_len = tf.random.uniform(shape=(), minval=2, maxval=10, dtype=tf.int32)
    data = tf.ones(shape=(bs, seq_len, 3))
    mechanism.setup_memory(data)
    if masked:
        mask = tf.sequence_mask(lengths=bs * [seq_len])
    else:
        mask = None
    return layer(data, mask=mask)


test(masked=False)
print('Test 1 passed')
test(masked=True)
print('Test 2 passed')
