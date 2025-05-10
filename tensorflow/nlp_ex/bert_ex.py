import numpy as np
import tensorflow as tf

from tensorflow_models import nlp

if __name__ == "__main__":
    # Build a small transformer network.
    vocab_size = 100
    network = nlp.networks.BertEncoder(
        vocab_size=vocab_size,
        # The number of TransformerEncoderBlock layers
        num_layers=3)
    # tf.keras.utils.plot_model(network, show_shapes=True, expand_nested=True, dpi=48)
    # Create a BERT pretrainer with the created network.
    num_token_predictions = 8
    bert_pretrainer = nlp.models.BertPretrainer(
        network, num_classes=2, num_token_predictions=num_token_predictions, output='predictions')
    # tf.keras.utils.plot_model(bert_pretrainer, show_shapes=True, expand_nested=True, dpi=48)

    # We can feed some dummy data to get masked language model and sentence output.
    sequence_length = 16
    batch_size = 2

    word_id_data = np.random.randint(vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(2, size=(batch_size, sequence_length))
    masked_lm_positions_data = np.random.randint(2, size=(batch_size, num_token_predictions))

    outputs = bert_pretrainer(
        [word_id_data, mask_data, type_id_data, masked_lm_positions_data])
    lm_output = outputs["masked_lm"]
    sentence_output = outputs["classification"]
    print(f'lm_output: shape={lm_output.shape}, dtype={lm_output.dtype!r}')
    print(f'sentence_output: shape={sentence_output.shape}, dtype={sentence_output.dtype!r}')

    masked_lm_ids_data = np.random.randint(vocab_size, size=(batch_size, num_token_predictions))
    masked_lm_weights_data = np.random.randint(2, size=(batch_size, num_token_predictions))
    next_sentence_labels_data = np.random.randint(2, size=(batch_size))

    mlm_loss = nlp.losses.weighted_sparse_categorical_crossentropy_loss(
        labels=masked_lm_ids_data,
        predictions=lm_output,
        weights=masked_lm_weights_data)
    sentence_loss = nlp.losses.weighted_sparse_categorical_crossentropy_loss(
        labels=next_sentence_labels_data,
        predictions=sentence_output)
    loss = mlm_loss + sentence_loss

    print(loss)

    network = nlp.networks.BertEncoder(vocab_size=vocab_size, num_layers=2)

    # Create a BERT trainer with the created network.
    bert_span_labeler = nlp.models.BertSpanLabeler(network)

    # tf.keras.utils.plot_model(bert_span_labeler, show_shapes=True, expand_nested=True, dpi=48)

    # Create a set of 2-dimensional data tensors to feed into the model.
    word_id_data = np.random.randint(vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(2, size=(batch_size, sequence_length))

    # Feed the data to the model.
    start_logits, end_logits = bert_span_labeler([word_id_data, mask_data, type_id_data])

    print(f'start_logits: shape={start_logits.shape}, dtype={start_logits.dtype!r}')
    print(f'end_logits: shape={end_logits.shape}, dtype={end_logits.dtype!r}')

    start_positions = np.random.randint(sequence_length, size=(batch_size))
    end_positions = np.random.randint(sequence_length, size=(batch_size))

    start_loss = tf.keras.losses.sparse_categorical_crossentropy(
        start_positions, start_logits, from_logits=True)
    end_loss = tf.keras.losses.sparse_categorical_crossentropy(
        end_positions, end_logits, from_logits=True)

    total_loss = (tf.reduce_mean(start_loss) + tf.reduce_mean(end_loss)) / 2
    print(total_loss)

    network = nlp.networks.BertEncoder(vocab_size=vocab_size, num_layers=2)

    # Create a BERT trainer with the created network.
    num_classes = 2
    bert_classifier = nlp.models.BertClassifier(network, num_classes=num_classes)

    # tf.keras.utils.plot_model(bert_classifier, show_shapes=True, expand_nested=True, dpi=48)
    # Create a set of 2-dimensional data tensors to feed into the model.
    word_id_data = np.random.randint(vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(2, size=(batch_size, sequence_length))

    # Feed the data to the model.
    logits = bert_classifier([word_id_data, mask_data, type_id_data])
    print(f'logits: shape={logits.shape}, dtype={logits.dtype!r}')
    labels = np.random.randint(num_classes, size=(batch_size))

    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True)
    print(loss)