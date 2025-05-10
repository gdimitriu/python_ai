import argparse
import pathlib
import re

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab


def add_start_end(ragged):
    global START
    global END
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], START)
    ends = tf.fill([count, 1], END)
    return tf.concat([starts, ragged, ends], axis=1)


def write_vocab_file(filepath, vocab):
    with open(filepath, 'w') as f:
        for token in vocab:
            print(token, file=f)


def cleanup_text(reserved_tokens, token_txt):
    # Drop the reserved tokens, except for "[UNK]".
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    # Join them into strings.
    result = tf.strings.reduce_join(result, separator=' ', axis=-1)

    return result


class CustomTokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        super().__init__()
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        ## Create the signatures for export:

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
        enc = add_start_end(enc)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Subwords')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
                        default='.', help='Directory for storing data')
    return parser


if __name__ == "__main__":
    global START
    global END
    args = build_arg_parser().parse_args()
    tf.get_logger().setLevel('ERROR')
    pwd = pathlib.Path.cwd()
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                                   data_dir=args.input_dir,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']
    for pt, en in train_examples.take(1):
        print("Portuguese: ", pt.numpy().decode('utf-8'))
        print("English:   ", en.numpy().decode('utf-8'))
    train_en = train_examples.map(lambda pt, en: en)
    train_pt = train_examples.map(lambda pt, en: pt)
    bert_tokenizer_params = dict(lower_case=True)
    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size=8000,
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params=bert_tokenizer_params,
        # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
        learn_params={},
    )
    pt_vocab = bert_vocab.bert_vocab_from_dataset(
        train_pt.batch(1000).prefetch(2),
        **bert_vocab_args
    )
    print(pt_vocab[:10])
    print(pt_vocab[100:110])
    print(pt_vocab[1000:1010])
    print(pt_vocab[-10:])
    write_vocab_file('../text/pt_vocab.txt', pt_vocab)
    en_vocab = bert_vocab.bert_vocab_from_dataset(
        train_en.batch(1000).prefetch(2),
        **bert_vocab_args
    )
    print(en_vocab[:10])
    print(en_vocab[100:110])
    print(en_vocab[1000:1010])
    print(en_vocab[-10:])
    write_vocab_file('../text/en_vocab.txt', en_vocab)

    # build the tokenizer
    pt_tokenizer = text.BertTokenizer('pt_vocab.txt', **bert_tokenizer_params)
    en_tokenizer = text.BertTokenizer('en_vocab.txt', **bert_tokenizer_params)
    for pt_examples, en_examples in train_examples.batch(3).take(1):
        for ex in en_examples:
            print(ex.numpy())

    # Tokenize the examples -> (batch, word, word-piece)
    token_batch = en_tokenizer.tokenize(en_examples)
    # Merge the word and word-piece axes -> (batch, tokens)
    token_batch = token_batch.merge_dims(-2, -1)

    for ex in token_batch.to_list():
        print(ex)

    # Lookup each token id in the vocabulary.
    txt_tokens = tf.gather(en_vocab, token_batch)
    # Join with spaces.
    print(tf.strings.reduce_join(txt_tokens, separator=' ', axis=-1))
    words = en_tokenizer.detokenize(token_batch)
    print(tf.strings.reduce_join(words, separator=' ', axis=-1))
    START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
    END = tf.argmax(tf.constant(reserved_tokens) == "[END]")
    words = en_tokenizer.detokenize(add_start_end(token_batch))
    print(tf.strings.reduce_join(words, separator=' ', axis=-1))
    print(en_examples.numpy())
    token_batch = en_tokenizer.tokenize(en_examples).merge_dims(-2, -1)
    words = en_tokenizer.detokenize(token_batch)
    print(words)
    print(cleanup_text(reserved_tokens, words).numpy())
    tokenizers = tf.Module()
    tokenizers.pt = CustomTokenizer(reserved_tokens, '../text/pt_vocab.txt')
    tokenizers.en = CustomTokenizer(reserved_tokens, '../text/en_vocab.txt')
    model_name = 'ted_hrlr_translate_pt_en_converter'
    tf.saved_model.save(tokenizers, model_name)
    reloaded_tokenizers = tf.saved_model.load(model_name)
    print(reloaded_tokenizers.en.get_vocab_size().numpy())
    tokens = reloaded_tokenizers.en.tokenize(['Hello TensorFlow!'])
    print(tokens.numpy())
    text_tokens = reloaded_tokenizers.en.lookup(tokens)
    print(text_tokens)
