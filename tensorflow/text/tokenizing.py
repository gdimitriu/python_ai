# https://www.tensorflow.org/text/guide/tokenizers
import requests
import tensorflow as tf
import tensorflow_text as tf_text

def decode_list(x):
  if type(x) is list:
    return list(map(decode_list, x))
  return x.decode("UTF-8")

def decode_utf8_tensor(x):
  return list(map(decode_list, x.to_list()))


if __name__ == "__main__":
    tokenizer = tf_text.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
    print(tokens.to_list())
    tokenizer = tf_text.UnicodeScriptTokenizer()
    tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
    print(tokens.to_list())
    tokenizer = tf_text.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
    print(tokens.to_list())
    url = "https://github.com/tensorflow/text/blob/master/tensorflow_text/python/ops/test_data/test_wp_en_vocab.txt?raw=true"
    r = requests.get(url)
    filepath = "vocab.txt"
    print(open(filepath, 'wb').write(r.content))
    subtokenizer = tf_text.UnicodeScriptTokenizer(filepath)
    subtokens = tokenizer.tokenize(tokens)
    print(subtokens.to_list())
    tokenizer = tf_text.BertTokenizer(filepath, token_out_type=tf.string, lower_case=True)
    tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
    print(tokens.to_list())
    url = "https://github.com/tensorflow/text/blob/master/tensorflow_text/python/ops/test_data/test_oss_model.model?raw=true"
    sp_model = requests.get(url).content
    tokenizer = tf_text.SentencepieceTokenizer(sp_model, out_type=tf.string)
    tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
    print(tokens.to_list())
    tokenizer = tf_text.UnicodeCharTokenizer()
    tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
    print(tokens.to_list())
    characters = tf.strings.unicode_encode(tf.expand_dims(tokens, -1), "UTF-8")
    bigrams = tf_text.ngrams(characters, 2, reduction_type=tf_text.Reduction.STRING_JOIN, string_separator='')
    print(bigrams.to_list())
    #MODEL_HANDLE = "https://tfhub.dev/google/zh_segmentation/1"
    #segmenter = tf_text.HubModuleTokenizer(MODEL_HANDLE)
    #tokens = segmenter.tokenize(["新华社北京"])
    #print(tokens.to_list())
    #print(decode_utf8_tensor(tokens))
    strings = ["新华社北京"]
    labels = [[0, 1, 1, 0, 1]]
    tokenizer = tf_text.SplitMergeTokenizer()
    tokens = tokenizer.tokenize(strings, labels)
    print(decode_utf8_tensor(tokens))
    strings = [["新华社北京"]]
    labels = [[[5.0, -3.2], [0.2, 12.0], [0.0, 11.0], [2.2, -1.0], [-3.0, 3.0]]]
    tokenizer = tf_text.SplitMergeFromLogitsTokenizer()
    tokenizer.tokenize(strings, labels)
    print(decode_utf8_tensor(tokens))
    splitter = tf_text.RegexSplitter("\s")
    tokens = splitter.split(["What you know you can't explain, but you feel it."], )
    print(tokens.to_list())
    tokenizer = tf_text.UnicodeScriptTokenizer()
    (tokens, start_offsets, end_offsets) = tokenizer.tokenize_with_offsets(['Everything not saved will be lost.'])
    print(tokens.to_list())
    print(start_offsets.to_list())
    print(end_offsets.to_list())
    tokenizer = tf_text.UnicodeCharTokenizer()
    tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
    print(tokens.to_list())
    strings = tokenizer.detokenize(tokens)
    print(strings.numpy())
    docs = tf.data.Dataset.from_tensor_slices([['Never tell me the odds.'], ["It's a trap!"]])
    tokenizer = tf_text.WhitespaceTokenizer()
    tokenized_docs = docs.map(lambda x: tokenizer.tokenize(x))
    iterator = iter(tokenized_docs)
    print(next(iterator).to_list())
    print(next(iterator).to_list())
