# https://www.tensorflow.org/text/tutorials/transformer
import argparse

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Translate pt in en with transformers')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
                        default='.', help='Directory for storing data')
    return parser


MAX_TOKENS = 128
BUFFER_SIZE = 20000
BATCH_SIZE = 64


def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)  # Output is ragged.
    pt = pt[:, :MAX_TOKENS]  # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS + 1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()  # Drop the [START] tokens

    return (pt, en_inputs), en_labels


def make_batches(ds):
    return (
        ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE))


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positional_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = keras.Sequential([
            keras.layers.Dense(dff, activation='relu'),
            keras.layers.Dense(d_model),
            keras.layers.Dropout(dropout_rate)
        ])
        self.add = keras.layers.Add()
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.


class DecoderLayer(keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class Decoder(keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                 d_model=d_model)
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x


class Transformer(keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)

        self.final_layer = keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits


class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


class Translator(tf.Module):
    def __init__(self, tokenizers, transformer):
        super().__init__()
        self.tokenizers = tokenizers
        self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_TOKENS):
        # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

        encoder_input = sentence

        # As the output language is English, initialize the output with the
        # English `[START]` token.
        start_end = self.tokenizers.en.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a Python list), so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)

            # Select the last token from the `seq_len` dimension.
            predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.
            output_array = output_array.write(i + 1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        # The output shape is `(1, tokens)`.
        text = tokenizers.en.detokenize(output)[0]  # Shape: `()`.

        tokens = tokenizers.en.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop.
        # So, recalculate them outside the loop.
        self.transformer([encoder_input, output[:, :-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores

        return text, tokens, attention_weights


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')


def plot_attention_head(in_tokens, translated_tokens, attention):
    # The model didn't generate `<START>` in the output. Skip it.
    translated_tokens = translated_tokens[1:]

    ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    labels = [label.decode('utf-8') for label in in_tokens.numpy()]
    ax.set_xticklabels(
        labels, rotation=90)

    labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
    ax.set_yticklabels(labels)
    plt.show()


def plot_attention_weights(sentence, translated_tokens, attention_heads):
    in_tokens = tf.convert_to_tensor([sentence])
    in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
    in_tokens = tokenizers.pt.lookup(in_tokens)[0]

    fig = plt.figure(figsize=(16, 8))

    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(2, 4, h + 1)

        plot_attention_head(in_tokens, translated_tokens, head)

        ax.set_xlabel(f'Head {h + 1}')

    plt.tight_layout()
    plt.show()


class ExportTranslator(tf.Module):
    def __init__(self, translator):
        super().__init__()
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        (result,
         tokens,
         attention_weights) = self.translator(sentence, max_length=MAX_TOKENS)

        return result


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                   data_dir=args.input_dir,
                                   with_info=True,
                                   as_supervised=True)

    train_examples, val_examples = examples['train'], examples['validation']
    for pt_examples, en_examples in train_examples.batch(3).take(1):
        print('> Examples in Portuguese:')
        for pt in pt_examples.numpy():
            print(pt.decode('utf-8'))
        print()

        print('> Examples in English:')
        for en in en_examples.numpy():
            print(en.decode('utf-8'))

    model_name = 'ted_hrlr_translate_pt_en_converter'
    # just download once and the renamed and moved to correct directory
    # keras.utils.get_file(
    #    f'{model_name}.zip',
    #    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
    #    cache_dir=args.input_dir + "/models", cache_subdir='', extract=True
    # )
    tokenizers = tf.saved_model.load(args.input_dir + "/models/" + model_name)
    print([item for item in dir(tokenizers.en) if not item.startswith('_')])
    print('> This is a batch of strings:')
    for en in en_examples.numpy():
        print(en.decode('utf-8'))
    encoded = tokenizers.en.tokenize(en_examples)

    print('> This is a padded-batch of token IDs:')
    for row in encoded.to_list():
        print(row)
    round_trip = tokenizers.en.detokenize(encoded)

    print('> This is human-readable text:')
    for line in round_trip.numpy():
        print(line.decode('utf-8'))

    print('> This is the text split into tokens:')
    tokens = tokenizers.en.lookup(encoded)
    print(tokens)

    lengths = []

    for pt_examples, en_examples in train_examples.batch(1024):
        pt_tokens = tokenizers.pt.tokenize(pt_examples)
        lengths.append(pt_tokens.row_lengths())

        en_tokens = tokenizers.en.tokenize(en_examples)
        lengths.append(en_tokens.row_lengths())
        print('.', end='', flush=True)

    all_lengths = np.concatenate(lengths)

    plt.hist(all_lengths, np.linspace(0, 500, 101))
    plt.ylim(plt.ylim())
    max_length = max(all_lengths)
    plt.plot([max_length, max_length], plt.ylim())
    plt.title(f'Maximum tokens per example: {max_length}')
    plt.show()

    # Create training and validation set batches.
    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)

    for (pt, en), en_labels in train_batches.take(1):
        break

    print(pt.shape)
    print(en.shape)
    print(en_labels.shape)
    print(en[0][:10])
    print(en_labels[0][:10])

    # DEFINE THE COMPONENTS
    # test positional encoding
    pos_encoding = positional_encoding(length=2048, depth=512)

    # Check the shape.
    print(pos_encoding.shape)

    # Plot the dimensions.
    plt.pcolormesh(pos_encoding.numpy().T, cmap='RdBu')
    plt.ylabel('Depth')
    plt.xlabel('Position')
    plt.colorbar()
    plt.show()
    pos_encoding /= tf.norm(pos_encoding, axis=1, keepdims=True)
    p = pos_encoding[1000]
    dots = tf.einsum('pd,d -> p', pos_encoding, p)
    plt.subplot(2, 1, 1)
    plt.plot(dots)
    plt.ylim([0, 1])
    plt.plot([950, 950, float('nan'), 1050, 1050],
             [0, 1, float('nan'), 0, 1], color='k', label='Zoom')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(dots)
    plt.xlim([950, 1050])
    plt.ylim([0, 1])
    plt.show()

    # embedding
    embed_pt = PositionalEmbedding(vocab_size=tokenizers.pt.get_vocab_size().numpy(), d_model=512)
    embed_en = PositionalEmbedding(vocab_size=tokenizers.en.get_vocab_size().numpy(), d_model=512)

    pt_emb = embed_pt(pt)
    en_emb = embed_en(en)
    print(en_emb._keras_mask)

    # test and run sample inputs
    sample_ca = CrossAttention(num_heads=2, key_dim=512)

    print(pt_emb.shape)
    print(en_emb.shape)
    print(sample_ca(en_emb, pt_emb).shape)

    sample_gsa = GlobalSelfAttention(num_heads=2, key_dim=512)

    print(pt_emb.shape)
    print(sample_gsa(pt_emb).shape)

    sample_csa = CausalSelfAttention(num_heads=2, key_dim=512)

    print(en_emb.shape)
    print(sample_csa(en_emb).shape)

    out1 = sample_csa(embed_en(en[:, :3]))
    out2 = sample_csa(embed_en(en))[:, :3]

    print(tf.reduce_max(abs(out1 - out2)).numpy())

    sample_ffn = FeedForward(512, 2048)

    print(en_emb.shape)
    print(sample_ffn(en_emb).shape)

    sample_encoder_layer = EncoderLayer(d_model=512, num_heads=8, dff=2048)

    print(pt_emb.shape)
    print(sample_encoder_layer(pt_emb).shape)

    # Instantiate the encoder.
    sample_encoder = Encoder(num_layers=4,
                             d_model=512,
                             num_heads=8,
                             dff=2048,
                             vocab_size=8500)

    sample_encoder_output = sample_encoder(pt, training=False)

    # Print the shape.
    print(pt.shape)
    print(sample_encoder_output.shape)  # Shape `(batch_size, input_seq_len, d_model)`.

    sample_decoder_layer = DecoderLayer(d_model=512, num_heads=8, dff=2048)

    sample_decoder_layer_output = sample_decoder_layer(
        x=en_emb, context=pt_emb)

    print(en_emb.shape)
    print(pt_emb.shape)
    print(sample_decoder_layer_output.shape)  # `(batch_size, seq_len, d_model)`

    # Instantiate the decoder.
    sample_decoder = Decoder(num_layers=4,
                             d_model=512,
                             num_heads=8,
                             dff=2048,
                             vocab_size=8000)

    output = sample_decoder(
        x=en,
        context=pt_emb)

    # Print the shapes.
    print(en.shape)
    print(pt_emb.shape)
    print(output.shape)
    print(sample_decoder.last_attn_scores.shape)  # (batch, heads, target_seq, input_seq)
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        dropout_rate=dropout_rate)
    output = transformer((pt, en))

    print(en.shape)
    print(pt.shape)
    print(output.shape)
    attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
    print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)
    # transformer.summary()

    # TRAIN THE MODEL
    learning_rate = CustomSchedule(d_model)

    optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    plt.plot(learning_rate(tf.range(40000, dtype=tf.float32)))
    plt.ylabel('Learning Rate')
    plt.xlabel('Train Step')
    plt.show()

    is_loaded = True
    if is_loaded:
        translator = tf.saved_model.load(args.input_dir + '/models/translator')
    else:
        transformer.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
        transformer.fit(train_batches, epochs=1, validation_data=val_batches)
        translator = Translator(tokenizers, transformer)
        # export here the translator not at the end because with cpu it takes very long time to train
        translatorExported = ExportTranslator(translator)
        print(translatorExported('este é o primeiro livro que eu fiz.').numpy())
        tf.saved_model.save(translatorExported, export_dir=args.input_dir + '/models/translator')

    # examples
    sentence = 'este é um problema que temos que resolver.'
    ground_truth = 'this is a problem we have to solve .'

    if is_loaded:
        translated_text = translator(sentence)
        print_translation(sentence, translated_text, ground_truth)
    else:
        translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
        print_translation(sentence, translated_text, ground_truth)

    sentence = 'os meus vizinhos ouviram sobre esta ideia.'
    ground_truth = 'and my neighboring homes heard about this idea .'

    if is_loaded:
        translated_text = translator(sentence)
        print_translation(sentence, translated_text, ground_truth)
    else:
        translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
        print_translation(sentence, translated_text, ground_truth)

    sentence = 'vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.'
    ground_truth = "so i'll just share with you some stories very quickly of some magical things that have happened."

    if is_loaded:
        translated_text = translator(sentence)
        print_translation(sentence, translated_text, ground_truth)
    else:
        translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
        print_translation(sentence, translated_text, ground_truth)

    sentence = 'este é o primeiro livro que eu fiz.'
    ground_truth = "this is the first book i've ever done."

    if is_loaded:
        translated_text = translator(sentence)
        print_translation(sentence, translated_text, ground_truth)
    else:
        translated_text, translated_tokens, attention_weights = translator(tf.constant(sentence))
        print_translation(sentence, translated_text, ground_truth)

        head = 0
        # Shape: `(batch=1, num_heads, seq_len_q, seq_len_k)`.
        attention_heads = tf.squeeze(attention_weights, 0)
        attention = attention_heads[head]
        print(attention.shape)

        in_tokens = tf.convert_to_tensor([sentence])
        in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
        in_tokens = tokenizers.pt.lookup(in_tokens)[0]
        print(in_tokens)
        print(translated_tokens)

        #plot_attention_head(in_tokens, translated_tokens, attention)
        #plot_attention_weights(sentence, translated_tokens, attention_weights[0])

    # translator = ExportTranslator(translator)
    # print(translator('este é o primeiro livro que eu fiz.').numpy())
    # tf.saved_model.save(translator, export_dir=args.input_dir + '/models/translator')
    reloaded = tf.saved_model.load(args.input_dir + '/models/translator')
    print(reloaded('este é o primeiro livro que eu fiz.').numpy())
