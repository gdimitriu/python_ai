import tensorflow as tf

if __name__ == "__main__":
    # dtype: string; shape: [num_sentences]
    #
    # The sentences to process.  Edit this line to try out different inputs!
    sentence_texts = [u'Hello, world.', u'世界こんにちは']
    # dtype: int32; shape: [num_sentences, (num_chars_per_sentence)]
    #
    # sentence_char_codepoint[i, j] is the codepoint for the j'th character in
    # the i'th sentence.
    sentence_char_codepoint = tf.strings.unicode_decode(sentence_texts, 'UTF-8')
    print(sentence_char_codepoint)

    # dtype: int32; shape: [num_sentences, (num_chars_per_sentence)]
    #
    # sentence_char_scripts[i, j] is the Unicode script of the j'th character in
    # the i'th sentence.
    sentence_char_script = tf.strings.unicode_script(sentence_char_codepoint)
    print(sentence_char_script)
    # dtype: bool; shape: [num_sentences, (num_chars_per_sentence)]
    #
    # sentence_char_starts_word[i, j] is True if the j'th character in the i'th
    # sentence is the start of a word.
    sentence_char_starts_word = tf.concat(
        [tf.fill([sentence_char_script.nrows(), 1], True),
         tf.not_equal(sentence_char_script[:, 1:], sentence_char_script[:, :-1])],
        axis=1)

    # dtype: int64; shape: [num_words]
    #
    # word_starts[i] is the index of the character that starts the i'th word (in
    # the flattened list of characters from all sentences).
    word_starts = tf.squeeze(tf.where(sentence_char_starts_word.values), axis=1)
    print(word_starts)
    # dtype: int32; shape: [num_words, (num_chars_per_word)]
    #
    # word_char_codepoint[i, j] is the codepoint for the j'th character in the
    # i'th word.
    word_char_codepoint = tf.RaggedTensor.from_row_starts(
        values=sentence_char_codepoint.values,
        row_starts=word_starts)
    print(word_char_codepoint)
    # dtype: int64; shape: [num_sentences]
    #
    # sentence_num_words[i] is the number of words in the i'th sentence.
    sentence_num_words = tf.reduce_sum(
        tf.cast(sentence_char_starts_word, tf.int64),
        axis=1)

    # dtype: int32; shape: [num_sentences, (num_words_per_sentence), (num_chars_per_word)]
    #
    # sentence_word_char_codepoint[i, j, k] is the codepoint for the k'th character
    # in the j'th word in the i'th sentence.
    sentence_word_char_codepoint = tf.RaggedTensor.from_row_lengths(
        values=word_char_codepoint,
        row_lengths=sentence_num_words)
    print(sentence_word_char_codepoint)

    tf.strings.unicode_encode(sentence_word_char_codepoint, 'UTF-8').to_list()

    # dtype: int64; shape: [num_sentences]
    #
    # sentence_num_words[i] is the number of words in the i'th sentence.
    sentence_num_words = tf.reduce_sum(
        tf.cast(sentence_char_starts_word, tf.int64),
        axis=1)

    # dtype: int32; shape: [num_sentences, (num_words_per_sentence), (num_chars_per_word)]
    #
    # sentence_word_char_codepoint[i, j, k] is the codepoint for the k'th character
    # in the j'th word in the i'th sentence.
    sentence_word_char_codepoint = tf.RaggedTensor.from_row_lengths(
        values=word_char_codepoint,
        row_lengths=sentence_num_words)
    print(sentence_word_char_codepoint)

    tf.strings.unicode_encode(sentence_word_char_codepoint, 'UTF-8').to_list()
