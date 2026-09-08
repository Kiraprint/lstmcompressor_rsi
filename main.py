import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISM'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import json
import tensorflow as tf
import numpy as np
import random
import time
import math
import contextlib
import hashlib

from ArithmeticCoder import (
    ArithmeticEncoder,
    ArithmeticDecoder,
    BitOutputStream,
    BitInputStream
)


# ==============================
# Hyperparameters
# ==============================

# The batch size for training
batch_size = 140
seq_length = 2
rnn_units = 288
num_layers = 1
embedding_size = 512
start_learning_rate = 0.006
end_learning_rate = 0.0001
mode = 'both'

FREQ_SCALE = 16000000
path_to_file = "data/enwik5"
path_to_compressed = path_to_file + "_compressed.dat"
path_to_decompressed = path_to_file + "_decompressed.dat"


class LearnableTemperature(tf.keras.layers.Layer):
    """Learnable softmax temperature.
    Идея из калибровки вероятностей: позволить модели немного менять
    остроту softmax, но в узких пределах, чтобы не испортить онлайн-обучение.
    """

    def __init__(self, min_tau=0.8, max_tau=1.25, **kwargs):
        super().__init__(**kwargs)
        self.min_tau = float(min_tau)
        self.max_tau = float(max_tau)
        self.log_min = math.log(self.min_tau)
        self.log_max = math.log(self.max_tau)

    def build(self, input_shape):
        self.log_tau = self.add_weight(
            name='log_tau',
            shape=(),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, logits):
        log_tau = tf.clip_by_value(self.log_tau, self.log_min, self.log_max)
        tau = tf.exp(log_tau)
        return logits / tau


def build_model(vocab_size: int) -> tf.keras.Model:
    """Builds the model architecture.

    Args:
        vocab_size: Int, size of the vocabulary.
    """
    inputs = [
        tf.keras.Input(shape=(seq_length,), batch_size=batch_size)
    ]

    # In addition to the primary input, there are also two "state" inputs
    # for each layer of the network.
    for _ in range(num_layers):
        inputs.append(tf.keras.Input(shape=(rnn_units,), batch_size=batch_size))
        inputs.append(tf.keras.Input(shape=(rnn_units,), batch_size=batch_size))

    embedding = tf.keras.layers.Embedding(
        vocab_size,
        embedding_size
    )(inputs[0])

    # Skip connections will be used to connect each LSTM layer output to the
    # final output layer. Each LSTM layer will get as input both the original
    # input and the output of the previous layer.
    skip_connections = []

    # In addition to the softmax output, there are also two "state" outputs
    # for each layer of the network.
    outputs = []

    predictions, state_h, state_c = tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        return_state=True,
        recurrent_initializer='glorot_uniform'
    )(
        embedding,
        initial_state=[inputs[1], inputs[2]]
    )

    skip_connections.append(predictions)
    outputs.append(state_h)
    outputs.append(state_c)

    for i in range(num_layers - 1):
        layer_input = tf.keras.layers.concatenate([embedding, skip_connections[-1]])

        predictions, state_h, state_c = tf.keras.layers.LSTM(
            rnn_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )(
            layer_input,
            initial_state=[inputs[i * 2 + 3], inputs[i * 2 + 4]]
        )

        skip_connections.append(predictions)
        outputs.append(state_h)
        outputs.append(state_c)

    # The dense output layer only needs to be computed for the last timestep.
    last_timestep = []
    for i in range(num_layers):
        last_timestep.append(
            tf.keras.layers.Lambda(lambda x: x[:, seq_length - 1, :])(
                skip_connections[i]
            )
        )

    if num_layers == 1:
        layer_input = last_timestep[0]
    else:
        layer_input = tf.keras.layers.concatenate(last_timestep)

    logits = tf.keras.layers.Dense(vocab_size, name='dense_logits')(layer_input)

    # Обучаемая температура для калибровки вероятностей.
    logits = LearnableTemperature(
        min_tau=0.8,
        max_tau=1.25,
        name='learnable_temperature'
    )(logits)

    output = tf.keras.layers.Activation(
        'softmax',
        dtype='float32',
        name='predictions'
    )(logits)

    outputs.insert(0, output)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_symbol(index, length, freq, coder, compress, data):
    """Runs arithmetic coding and returns the next symbol.

    Args:
        index: Int, position of the symbol in the file.
        length: Int, size limit of the file.
        freq: ndarray, predicted symbol probabilities.
        coder: this is the arithmetic coder.
        compress: Boolean, True if compressing, False if decompressing.
        data: List containing each symbol in the file.

    Returns:
        The next symbol, or 0 if "index" is over the file size limit.
    """
    symbol = 0

    if index < length:
        if compress:
            symbol = data[index]
            coder.write(freq, symbol)
        else:
            symbol = coder.read(freq)
            data[index] = symbol

    return symbol


def train(pos, seq_input, length, vocab_size, coder, model, optimizer, compress,
          data, states):
    """Runs one training step.

    Args:
        pos: Int, position in the file for the current symbol for the first batch.
        seq_input: Tensor, containing the last seq_length inputs for the model.
        length: Int, size limit of the file.
        vocab_size: Int, size of the vocabulary.
        coder: this is the arithmetic coder.
        model: the model to generate predictions.
        optimizer: optimizer used to train the model.
        compress: Boolean, True if compressing, False if decompressing.
        data: List containing each symbol in the file.
        states: List containing state information for the layers of the model.

    Returns:
        seq_input: Tensor, containing the last seq_length inputs for the model.
        cross_entropy: cross entropy numerator.
        denom: cross entropy denominator.
    """
    split = math.ceil(length / batch_size)

    with tf.GradientTape() as tape:
        # The model inputs contain both seq_input and the states for each layer.
        inputs = states.pop(0)
        inputs.insert(0, seq_input)

        # Run the model (for all batches in parallel) to get predictions for the
        # next characters.
        outputs = model(inputs)
        predictions = outputs.pop(0)

        states.append(outputs)

        p = predictions.numpy()

        # Быстрее считать кумулятивные частоты сразу для всего батча.
        freqs = np.cumsum(p * FREQ_SCALE + 1, axis=1)

        symbols = []
        mask = []

        # Go over each batch to run the arithmetic coding and prepare the next
        # input.
        for i in range(batch_size):
            index = pos + 1 + i * split
            symbol = get_symbol(index, length, freqs[i], coder, compress, data)
            symbols.append(symbol)

            if index < length:
                mask.append(1.0)
            else:
                mask.append(0.0)

        symbols_arr = np.array(symbols, dtype=np.int64)
        mask_arr = np.array(mask, dtype=np.float32)

        # For logging.
        probs_selected = p[np.arange(batch_size), symbols_arr]
        probs_selected = np.clip(probs_selected, 1e-7, 1.0)

        cross_entropy = float(np.sum(np.log2(probs_selected) * mask_arr))
        denom = int(np.sum(mask_arr))

        # Loss without one_hot: take probability of the correct symbol.
        symbols_tensor = tf.convert_to_tensor(symbols, dtype=tf.int32)
        mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

        probs = tf.gather(predictions, symbols_tensor, batch_dims=1)
        probs = tf.clip_by_value(probs, 1e-7, 1.0)

        loss = tf.reduce_sum(-tf.math.log(probs) * mask_tensor)

        # Remove the oldest input and append the new one.
        seq_input = tf.slice(
            seq_input,
            [0, 1],
            [batch_size, seq_length - 1]
        )
        seq_input = tf.concat(
            [seq_input, tf.expand_dims(symbols_tensor, 1)],
            1
        )

    # Run the backwards pass to update model weights.
    gradients = tape.gradient(loss, model.trainable_variables)

    # Gradient clipping to make training more robust.
    capped_grads = []
    for grad, var in zip(gradients, model.trainable_variables):
        if grad is None:
            grad = tf.zeros_like(var)
        capped_grads.append(tf.clip_by_norm(grad, 4.0))

    optimizer.apply_gradients(zip(capped_grads, model.trainable_variables))

    return seq_input, cross_entropy, denom


def reset_seed():
    """Initializes various random seeds to help with determinism."""
    SEED = 1234

    os.environ['PYTHONHASHSEED'] = str(SEED)

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def process(compress, length, vocab_size, coder, data):
    """This runs compression/decompression.

    Args:
        compress: Boolean, True if compressing, False if decompressing.
        length: Int, size limit of the file.
        vocab_size: Int, size of the vocabulary.
        coder: this is the arithmetic coder.
        data: List containing each symbol in the file.
    """
    start = time.time()

    reset_seed()

    model = build_model(vocab_size=vocab_size)
    model.summary()

    # Try to split the file into equal size pieces for the different batches.
    split = math.ceil(length / batch_size)
    lr_decay_steps = max(1, split)

    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        start_learning_rate,
        lr_decay_steps,
        end_learning_rate,
        power=1.0
    )

    # Возвращаем оригинальный более стабильный beta_2.
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_fn,
        beta_1=0,
        beta_2=0.9999,
        epsilon=1e-5
    )

    # Use a uniform distribution for predicting the first batch of symbols.
    freq = np.cumsum(
        np.full(vocab_size, (1.0 / vocab_size)) * FREQ_SCALE + 1
    )

    # Construct the first set of input characters for training.
    symbols = []
    for i in range(batch_size):
        symbols.append(
            get_symbol(i * split, length, freq, coder, compress, data)
        )

    # Replicate the input tensor seq_length times, to match the input format.
    seq_input = tf.tile(
        tf.expand_dims(symbols, 1),
        [1, seq_length]
    )

    pos = cross_entropy = denom = 0

    # This will keep track of layer states. Initialize them to zeros.
    states = []
    for _ in range(seq_length):
        states.append(
            [tf.zeros([batch_size, rnn_units]) for _ in range(num_layers * 2)]
        )

    # Keep repeating the training step until we get to the end of the file.
    while pos < split:
        seq_input, ce, d = train(
            pos,
            seq_input,
            length,
            vocab_size,
            coder,
            model,
            optimizer,
            compress,
            data,
            states
        )

        cross_entropy += ce
        denom += d
        pos += 1

        if pos % 5 == 0:
            percentage = 100 * pos / split

            if percentage >= 100:
                continue

            print(
                '\rProgress: %.2f%%, Cross entropy: %.2f, Time = %i' % (
                    percentage,
                    -cross_entropy / denom,
                    time.time() - start
                ),
                end=''
            )

    if compress:
        coder.finish()

    print('\n')


def compression():
    text = open(path_to_file, 'rb').read()

    vocab = sorted(set(text))

    # Используем реальный размер словаря.
    # Если уникальных байтов нет, берём 1, чтобы модель могла работать.
    vocab_size = max(1, len(vocab))

    # Creating a mapping from unique characters to indexes.
    char2idx = {u: i for i, u in enumerate(vocab)}

    int_list = [char2idx[c] for c in text]

    file_len = len(int_list)

    print('Length of file: {} symbols'.format(file_len))
    print('Vocabulary size: {}'.format(vocab_size))

    with open(path_to_compressed, "wb") as out, \
         contextlib.closing(BitOutputStream(out)) as bitout:

        length = len(int_list)

        # Write the original file length to the compressed file header.
        out.write(length.to_bytes(5, byteorder='big', signed=False))

        # Write 256 bits to the compressed file header to keep track
        # of the vocabulary.
        for i in range(256):
            if i in char2idx:
                bitout.write(1)
            else:
                bitout.write(0)

        enc = ArithmeticEncoder(32, bitout)

        process(
            True,
            length,
            vocab_size,
            enc,
            int_list
        )


# Compatibility alias for the original typo.
compession = compression


def decompression():
    with open(path_to_compressed, "rb") as inp, \
         open(path_to_decompressed, "wb") as out:

        # Read the original file size from the header.
        length = int.from_bytes(inp.read(5), byteorder='big')

        # Create a list to store the file characters.
        output = [0] * length

        bitin = BitInputStream(inp)

        # Get the vocabulary from the file header.
        vocab = []
        for i in range(256):
            if bitin.read():
                vocab.append(i)

        # Используем реальный размер словаря.
        vocab_size = max(1, len(vocab))

        dec = ArithmeticDecoder(32, bitin)

        process(
            False,
            length,
            vocab_size,
            dec,
            output
        )

        # Convert indexes back to the original characters.
        idx2char = np.array(vocab)

        for i in range(length):
            out.write(bytes((idx2char[output[i]],)))


def main():
    start = time.time()

    if mode == 'compress' or mode == 'both':
        compression()

        print(f"Original size: {os.path.getsize(path_to_file)} bytes")
        print(f"Compressed size: {os.path.getsize(path_to_compressed)} bytes")

    if mode == 'decompress' or mode == 'both':
        decompression()

        hash_dec = hashlib.md5(open(path_to_decompressed, 'rb').read()).hexdigest()
        hash_orig = hashlib.md5(open(path_to_file, 'rb').read()).hexdigest()

        assert hash_dec == hash_orig

    if os.path.exists(path_to_file) and os.path.exists(path_to_compressed):
        original_size = os.path.getsize(path_to_file)
        compressed_size = os.path.getsize(path_to_compressed)

        if compressed_size > 0:
            print('Compression ratio: %.2f' % (original_size / compressed_size))
        else:
            print('Compression ratio: inf')

    print('Time spent: %i' % (time.time() - start))


    result = {
        "compression_ratio": round(original_size / compressed_size, 2) if compressed_size > 0 else None,
        "time_spent": round(time.time() - start, 2)
    }
    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)

if __name__ == '__main__':
    main()