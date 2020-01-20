import tensorflow_datasets as tfds
import tensorflow as tf

import os

import json

def datsets():

        train_context = []
        train_response = []

        train_examples_list = []

        val_context = []
        val_response = []

        val_examples_list = []

        path_to_json = 'json_small/'
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

        for index, js in enumerate(json_files):
            with open(os.path.join(path_to_json, js)) as json_file:

                lines = json_file.readlines()

                for line in lines:
                    json_line = json.loads(line)
                    train_examples_list.append(json_line)
                    train_context.append(json_line['context'])
                    train_response.append(json_line['response'])

        train_examples = tf.data.Dataset.from_tensor_slices((train_context, train_response))

        with open(os.path.join(path_to_json, js)) as json_file:
            lines = json_file.readlines()
            for line in lines:
                json_line = json.loads(line)
                val_examples_list.append(json_line)
                val_context.append(json_line['context'])
                val_response.append(json_line['response'])

        val_examples = tf.data.Dataset.from_tensor_slices((val_context, val_response))

        tokenizer_context = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (line['context'] for line in train_examples_list), target_vocab_size=2 ** 13)

        tokenizer_response = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (line['response'] for line in train_examples_list), target_vocab_size=2 ** 13)

        BUFFER_SIZE = 20000
        BATCH_SIZE = 64
        MAX_LENGTH = 40

        def encode(context, response):
            context = [tokenizer_context.vocab_size] + tokenizer_context.encode(
                context.numpy()) + [tokenizer_context.vocab_size + 1]

            response = [tokenizer_response.vocab_size] + tokenizer_response.encode(
                response.numpy()) + [tokenizer_response.vocab_size + 1]

            return context, response

        def filter_max_length(x, y, max_length=MAX_LENGTH):
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)

        def tf_encode(pt, en):
            return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])

        train_dataset = train_examples.map(tf_encode)
        train_dataset = train_dataset.filter(filter_max_length)
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
            BATCH_SIZE, padded_shapes=([-1], [-1]))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = val_examples.map(tf_encode)
        val_dataset = val_dataset.filter(filter_max_length).padded_batch(
            BATCH_SIZE, padded_shapes=([-1], [-1]))

        return train_dataset, val_dataset, tokenizer_context, tokenizer_response
