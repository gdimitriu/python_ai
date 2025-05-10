# https://www.tensorflow.org/text/guide/unicode
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    print(tf.constant(u"Thanks 😊"))
    print(tf.constant([u"You're", u"welcome!"]).shape)
    # Unicode string, represented as a UTF-8 encoded string scalar.
    text_utf8 = tf.constant(u"语言处理")
    print(text_utf8)
    # Unicode string, represented as a UTF-16-BE encoded string scalar.
    text_utf16be = tf.constant(u"语言处理".encode("UTF-16-BE"))
    print(text_utf16be)
    # Unicode string, represented as a vector of Unicode code points.
    text_chars = tf.constant([ord(char) for char in u"语言处理"])
    print(text_chars)
    print(tf.strings.unicode_decode(text_utf8, input_encoding='UTF-8'))
    print(tf.strings.unicode_encode(text_chars, output_encoding='UTF-8'))
    print(tf.strings.unicode_transcode(text_utf8,
                             input_encoding='UTF8',
                             output_encoding='UTF-16-BE'))
    # A batch of Unicode strings, each represented as a UTF8-encoded string.
    batch_utf8 = [s.encode('UTF-8') for s in
                  [u'hÃllo', u'What is the weather tomorrow', u'Göödnight', u'😊']]
    batch_chars_ragged = tf.strings.unicode_decode(batch_utf8,
                                                   input_encoding='UTF-8')
    for sentence_chars in batch_chars_ragged.to_list():
        print(sentence_chars)
    batch_chars_padded = batch_chars_ragged.to_tensor(default_value=-1)
    print(batch_chars_padded.numpy())
    batch_chars_sparse = batch_chars_ragged.to_sparse()

    nrows, ncols = batch_chars_sparse.dense_shape.numpy()
    elements = [['_' for i in range(ncols)] for j in range(nrows)]
    for (row, col), value in zip(batch_chars_sparse.indices.numpy(), batch_chars_sparse.values.numpy()):
        elements[row][col] = str(value)
    # max_width = max(len(value) for row in elements for value in row)
    value_lengths = []
    for row in elements:
        for value in row:
            value_lengths.append(len(value))
    max_width = max(value_lengths)
    print('[%s]' % '\n '.join(
        '[%s]' % ', '.join(value.rjust(max_width) for value in row)
        for row in elements))
    print(tf.strings.unicode_encode([[99, 97, 116], [100, 111, 103], [99, 111, 119]], output_encoding='UTF-8'))
    print(tf.strings.unicode_encode(batch_chars_ragged, output_encoding='UTF-8'))
    print(tf.strings.unicode_encode(tf.RaggedTensor.from_sparse(batch_chars_sparse),output_encoding='UTF-8'))
    print(tf.strings.unicode_encode(tf.RaggedTensor.from_tensor(batch_chars_padded, padding=-1), output_encoding='UTF-8'))
    # Note that the final character takes up 4 bytes in UTF8.
    thanks = u'Thanks 😊'.encode('UTF-8')
    num_bytes = tf.strings.length(thanks).numpy()
    num_chars = tf.strings.length(thanks, unit='UTF8_CHAR').numpy()
    print('{} bytes; {} UTF-8 characters'.format(num_bytes, num_chars))

    inputs = [
        "This is a fantastic movie.",
        "This is a bad movie.",
        "This movie was so bad that it was good.",
        "I will never say yes to watching this movie.",
    ]
    batch_utf8 = [s.encode('UTF-8') for s in inputs]
    batch_chars_ragged = tf.strings.unicode_decode(batch_utf8[0], input_encoding='UTF-8')
    #batch_chars_padded = batch_chars_ragged.to_tensor(default_value=-1)
    print(batch_chars_ragged)
    # Note that the final character takes up 4 bytes in UTF8.
    thanks = u'Thanks 😊'.encode('UTF-8')
    num_bytes = tf.strings.length(thanks).numpy()
    num_chars = tf.strings.length(thanks, unit='UTF8_CHAR').numpy()
    print('{} bytes; {} UTF-8 characters'.format(num_bytes, num_chars))
    # Here, unit='BYTE' (default). Returns a single byte with len=1
    print(tf.strings.substr(thanks, pos=7, len=1).numpy())
    # Specifying unit='UTF8_CHAR', returns a single 4 byte character in this case
    print(tf.strings.substr(thanks, pos=7, len=1, unit='UTF8_CHAR').numpy())
    print(tf.strings.unicode_split(thanks, 'UTF-8').numpy())
    codepoints, offsets = tf.strings.unicode_decode_with_offsets(u'🎈🎉🎊', 'UTF-8')

    for (codepoint, offset) in zip(codepoints.numpy(), offsets.numpy()):
        print('At byte offset {}: codepoint {}'.format(offset, codepoint))
    uscript = tf.strings.unicode_script([33464, 1041])  # ['芸', 'Б']

    print(uscript.numpy())  # [17, 8] == [USCRIPT_HAN, USCRIPT_CYRILLIC]
    print(tf.strings.unicode_script(batch_chars_ragged))

