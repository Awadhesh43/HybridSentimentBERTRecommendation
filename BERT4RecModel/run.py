# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
import tensorflow as tf
import numpy as np
import sys
import pickle
import tensorflow.compat.v1 as tfv1

flags = tfv1.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "train_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "test_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "checkpointDir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("signature", 'default', "signature_name")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence. "
                     "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

#flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("batch_size", 32, "Total batch size for training.")

#flags.DEFINE_integer("eval_batch_size", 1, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 1000, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("use_pop_random", True, "use pop random negative samples")
flags.DEFINE_string("vocab_filename", None, "vocab filename")
flags.DEFINE_string("user_history_filename", None, "user history filename")

flags.DEFINE_string(
    "mode", "item-based",
    "mode name for user-based or item-based BERT model.")

flags.DEFINE_bool("layer_metrics", False, "Whether to calculate CF/CBF layer metrics.")

class EvalHooks(tfv1.train.SessionRunHook):
    def __init__(self):
        tfv1.logging.info('run init')

    def begin(self):
        self.valid_user = 0.0

        self.ndcg_1 = 0.0
        self.hit_1 = 0.0
        self.ndcg_5 = 0.0
        self.hit_5 = 0.0
        self.ndcg_10 = 0.0
        self.hit_10 = 0.0
        self.ndcg_20 = 0.0
        self.hit_20 = 0.0
        self.ndcg_30 = 0.0
        self.hit_30 = 0.0
        self.ndcg_40 = 0.0
        self.hit_40 = 0.0
        self.ndcg_50 = 0.0
        self.hit_50 = 0.0
        self.ap = 0.0

        self.true_item = {}
        self.predictions = {}
        self.user = []
        self.mask_item = []
        self.mask_item_predict = []

        np.random.seed(12345)

        self.vocab = None

        if FLAGS.user_history_filename is not None:
            print('load user history from :' + FLAGS.user_history_filename)
            with open(FLAGS.user_history_filename, 'rb') as input_file:
                self.user_history = pickle.load(input_file)

        if FLAGS.vocab_filename is not None:
            print('load vocab from :' + FLAGS.vocab_filename)
            with open(FLAGS.vocab_filename, 'rb') as input_file:
                self.vocab = pickle.load(input_file)

            keys = self.vocab.counter.keys()
            values = self.vocab.counter.values()
            self.ids = self.vocab.convert_tokens_to_ids(keys)
            # normalize
            # print(values)
            sum_value = np.sum([x for x in values])
            # print(sum_value)
            self.probability = [value / sum_value for value in values]

    def end(self, session):
        if FLAGS.layer_metrics:
            print("ndcg@1:{}, hit@1:{}, ndcg@5:{}, hit@5:{}, ndcg@10:{}, hit@10:{}, ndcg@20:{}, hit@20:{}, ndcg@30:{}, hit@30:{}, ndcg@40:{}, hit@40:{}, ndcg@50:{}, hit@50:{}, ap:{}, valid_user:{}".
                format(self.ndcg_1 / self.valid_user, self.hit_1 / self.valid_user,
                    self.ndcg_5 / self.valid_user, self.hit_5 / self.valid_user,
                    self.ndcg_10 / self.valid_user, self.hit_10 / self.valid_user,
                    self.ndcg_20 / self.valid_user, self.hit_20 / self.valid_user,
                    self.ndcg_30 / self.valid_user, self.hit_30 / self.valid_user,
                    self.ndcg_40 / self.valid_user, self.hit_40 / self.valid_user,
                    self.ndcg_50 / self.valid_user, self.hit_50 / self.valid_user,
                    self.ap / self.valid_user, self.valid_user))

        print("mode:", FLAGS.mode)
        if FLAGS.mode  == "item-based":
            # for u in self.true_item.keys():
            #     for i in self.true_item[u]:
            #         j = i.strip('item_')
            #         self.true_item[u].append(movies[int(j)])
            #         break
            output_true_file = os.path.join(FLAGS.checkpointDir,
                                            "true_results.txt")
            with open(output_true_file, 'w') as f:
                for key in self.true_item.keys():
                    f.write("%s = %s\n" % (key, str(self.true_item[key])))
                f.close()

            # for u in self.predictions.keys():
            #     for i in self.predictions[u]:
            #         j = i.strip('item_')
            #         self.predictions[u].append(movies[int(j)])
            #         break

            output_predict_file = os.path.join(FLAGS.checkpointDir,
                                               "predictions_results.txt")
            with open(output_predict_file, 'w') as f:
                for key in self.predictions.keys():
                    f.write("%s = %s\n" % (key, str(self.predictions[key])))
                f.close()

        if FLAGS.mode == "user-based":
            # for u in self.true_item.keys():
            #     for i in self.true_item[u]:
            #         j = i.strip('item_')
            #         self.true_item[u].append(movies[int(j)])
            #         break

            output_true_file = os.path.join(FLAGS.checkpointDir,
                                            "true_results.txt")
            with open(output_true_file, 'w') as f:
                for key in self.true_item.keys():
                    f.write("%s = %s\n" % (key, str(self.true_item[key])))
                f.close()

            # for u in self.predictions.keys():
            #     for i in self.predictions[u]:
            #         j = i.strip('item_')
            #         self.predictions[u].append(movies[int(j)])
            #         break

            output_predict_file = os.path.join(FLAGS.checkpointDir,
                                               "predictions_results.txt")
            with open(output_predict_file, 'w') as f:
                for key in self.predictions.keys():
                    f.write("%s = %s\n" % (key, str(self.predictions[key])))
                f.close()

    def before_run(self, run_context):
        #tf.logging.info('run before run')
        #print('run before_run')
        variables = tfv1.get_collection('eval_sp')
        return tfv1.train.SessionRunArgs(variables)

    def after_run(self, run_context, run_values):
        #tf.logging.info('run after run')
        #print('run after run')
        masked_lm_log_probs, input_ids, masked_lm_ids, info, masked_lm_predictions = run_values.results
        masked_lm_log_probs = masked_lm_log_probs.reshape(
            (-1, FLAGS.max_predictions_per_seq, masked_lm_log_probs.shape[1]))
        #print("masked_lm_log_probs shape:", masked_lm_log_probs.shape, "\n input_ids shape:", input_ids.shape,
         #      "\n masked_lm_ids shape:", masked_lm_ids.shape, "\n info shape:", info.shape,
          #     "\n masked_lm_predictions shape:", masked_lm_predictions.shape)

        for user in info:
            for j in user:
                self.user.append(j)

        for mask in masked_lm_ids:
            for m in range(len(mask)):
                if m == 0:
                    self.mask_item.append(self.vocab.convert_ids_to_tokens([mask[m]]))

        for pred in masked_lm_predictions:
            for index in range(len(pred)):
                if index == 0:
                    self.mask_item_predict.append(self.vocab.convert_ids_to_tokens([pred[index]]))

        for i in range(len(self.user)):
            self.true_item[self.user[i]] = self.mask_item[i]

        for i in range(len(self.user)):
            self.predictions[self.user[i]] = self.mask_item_predict[i]

        if FLAGS.layer_metrics:
            for idx in range(len(input_ids)):
                rated = set(input_ids[idx])
                rated.add(0)
                rated.add(masked_lm_ids[idx][0])
                if FLAGS.mode  == "item-based":
                    map(lambda x: rated.add(x),
                        self.user_history["user_" + str(info[idx][0])][0])
                elif FLAGS.mode  == "user-based":
                    map(lambda x: rated.add(x),
                        self.user_history["item_" + str(info[idx][0])][0])
                item_idx = [masked_lm_ids[idx][0]]
                # here we need more consideration
                masked_lm_log_probs_elem = masked_lm_log_probs[idx, 0]  
                size_of_prob = len(self.ids) + 1  # len(masked_lm_log_probs_elem)
                if FLAGS.use_pop_random:
                    if self.vocab is not None:
                        while len(item_idx) < 101:
                            sampled_ids = np.random.choice(self.ids, 101, replace=False, p=self.probability)
                            sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
                            item_idx.extend(sampled_ids[:])
                        item_idx = item_idx[:101]
                else:
                    # print("evaluation random -> ")
                    for _ in range(100):
                        t = np.random.randint(1, size_of_prob)
                        while t in rated:
                            t = np.random.randint(1, size_of_prob)
                        item_idx.append(t)

                predictions = -masked_lm_log_probs_elem[item_idx]
                rank = predictions.argsort().argsort()[0]

                self.valid_user += 1

                if self.valid_user % 100 == 0:
                    print('.', end='')
                    sys.stdout.flush()

                if rank < 1:
                    self.ndcg_1 += 1
                    self.hit_1 += 1
                if rank < 5:
                    self.ndcg_5 += 1 / np.log2(rank + 2)
                    self.hit_5 += 1
                if rank < 10:
                    self.ndcg_10 += 1 / np.log2(rank + 2)
                    self.hit_10 += 1

                self.ap += 1.0 / (rank + 1)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, item_size):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tfv1.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tfv1.logging.info("  name = %s, shape = %s" % (name,
                                                         features[name].shape))

        info = features["info"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=use_one_hot_embeddings)
        
#         all_user_and_item = model.get_embedding_table()
#         item_ids = [i for i in range(0, item_size + 1)]
#         softmax_output_embedding = tf.nn.embedding_lookup(all_user_and_item, item_ids)

        (masked_lm_loss, masked_lm_example_loss,
         masked_lm_log_probs, masked_lm_predictions) = get_masked_lm_output(
             bert_config,
             model.get_sequence_output(),
             model.get_embedding_table(), masked_lm_positions, masked_lm_ids,
             masked_lm_weights)

        total_loss = masked_lm_loss

        tvars = tfv1.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(
                 tvars, init_checkpoint)

            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint,
                                                  assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tfv1.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tfv1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            tfv1.logging.info("**** Training Optimization parameters **** learning_rate %s", learning_rate)
            train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(masked_lm_example_loss, masked_lm_log_probs,
                          masked_lm_ids, masked_lm_weights):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.reshape(
                    masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]])
                masked_lm_predictions = tf.argmax(
                    masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss,
                                                    [-1])
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tfv1.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights)
                masked_lm_mean_loss = tfv1.metrics.mean(
                    values=masked_lm_example_loss, weights=masked_lm_weights)

                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss,
                }

            tfv1.add_to_collection('eval_sp', masked_lm_log_probs)
            tfv1.add_to_collection('eval_sp', input_ids)
            tfv1.add_to_collection('eval_sp', masked_lm_ids)
            tfv1.add_to_collection('eval_sp', info)
            tfv1.add_to_collection('eval_sp', masked_lm_predictions)            

            eval_metrics = metric_fn(masked_lm_example_loss,
                                     masked_lm_log_probs, masked_lm_ids,
                                     masked_lm_weights)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)

        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" %(mode))

        return output_spec

    return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    # [batch_size*label_size, dim]
    input_tensor = gather_indexes(input_tensor, positions)

    with tfv1.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tfv1.variable_scope("transform"):
            input_tensor = tfv1.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tfv1.get_variable(
            "output_bias",
            shape=[output_weights.shape[0]],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        # logits, (bs*label_size, vocab_size)
        log_probs = tf.nn.log_softmax(logits, -1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=output_weights.shape[0], dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(
            log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

        predictions = tf.reshape(log_probs, (-1, FLAGS.max_predictions_per_seq, log_probs.shape[1]))
        predictions = tf.argmax(
            predictions, axis=-1, output_type=tf.int32)

    return (loss, per_example_loss, log_probs, predictions)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "info":
            tfv1.FixedLenFeature([1], tf.int64),  #[user]
            "input_ids":
            tfv1.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
            tfv1.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
            tfv1.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
            tfv1.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
            tfv1.FixedLenFeature([max_predictions_per_seq], tf.float32)
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.TFRecordDataset(input_files)
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

            # `cycle_length` is the number of parallel files that get read.
            #cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            #d = d.apply(
            #    tf.contrib.data.parallel_interleave(
            #        tf.data.TFRecordDataset,
            #        sloppy=is_training,
            #        cycle_length=cycle_length))
            #d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)


        d = d.map(
            lambda record: _decode_record(record, name_to_features),
            num_parallel_calls=num_cpu_threads)
        d = d.batch(batch_size=batch_size)
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tfv1.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tfv1.int64:
            t = tfv1.to_int32(t)
        example[name] = t

    return example


def load_model(filename):
    ML_MODELS_DIRECTORY = "saved_model_files/"
    saved_file_prefix = FLAGS.checkpointDir[:-len(FLAGS.signature) + 1]
    filePath = ML_MODELS_DIRECTORY + saved_file_prefix + filename
    print(filePath)
    try:
        return pickle.load(open(filePath, "rb"))
    except:
        return None


def main(_):
    tfv1.logging.set_verbosity(tfv1.logging.INFO)

    FLAGS.checkpointDir = FLAGS.checkpointDir + FLAGS.signature
    print('checkpointDir:', FLAGS.checkpointDir)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tfv1.gfile.MakeDirs(FLAGS.checkpointDir)

    train_input_files = []
    for input_pattern in FLAGS.train_input_file.split(","):
        train_input_files.extend(tfv1.gfile.Glob(input_pattern))

    test_input_files = []
    if FLAGS.test_input_file is None:
        test_input_files = train_input_files
    else:
        for input_pattern in FLAGS.test_input_file.split(","):
            test_input_files.extend(tfv1.gfile.Glob(input_pattern))

    tfv1.logging.info("*** Train Input Files ***")
    for input_file in train_input_files:
        tfv1.logging.info("  %s" % input_file)

    tfv1.logging.info("*** Test Input Files ***")
    for input_file in test_input_files:
        tfv1.logging.info("  %s" % input_file)

    tpu_resolver = None
    num_accelerators = 0
    try:
        tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        num_accelerators = tpu_resolver.num_accelerators()["TPU"]
        print(f'Running on a TPU w/{num_accelerators} cores')
    except ValueError:
        print('WARN: Not connected to a TPU runtime')

    if tpu_resolver is not None:
        tf.config.experimental_connect_to_cluster(tpu_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_resolver)

        tpu_strategy = tf.distribute.TPUStrategy(tpu_resolver)
    
    #is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.checkpointDir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    
    if FLAGS.vocab_filename is not None:
        with open(FLAGS.vocab_filename, 'rb') as input_file:
            vocab = pickle.load(input_file)
    item_size = len(vocab.counter)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        item_size=item_size)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={
            "batch_size": FLAGS.batch_size
        })

    if num_accelerators == 0:
        num_cpu_threads = 4
    else:
        num_cpu_threads = num_accelerators

    if FLAGS.do_train:
        tfv1.logging.info("***** Running training *****")
        tfv1.logging.info("  Batch size = %d", FLAGS.batch_size)
        train_input_fn = input_fn_builder(
            input_files=train_input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=True, num_cpu_threads=num_cpu_threads)
        estimator.train(
            input_fn=train_input_fn,
            max_steps=FLAGS.num_train_steps)

    if FLAGS.do_eval:
        tfv1.logging.info("***** Running evaluation *****")
        tfv1.logging.info("  Batch size = %d", FLAGS.batch_size)

        eval_input_fn = input_fn_builder(
            input_files=test_input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False, num_cpu_threads=num_cpu_threads)

        #tfv1.logging.info('special eval ops:', special_eval_ops)
        result = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=None,
            hooks=[EvalHooks()])

        output_eval_file = os.path.join(FLAGS.checkpointDir, "eval_results.txt")
        with tfv1.gfile.GFile(output_eval_file, "w") as writer:
            tfv1.logging.info("***** Eval results *****")
            tfv1.logging.info(bert_config.to_json_string())
            writer.write(bert_config.to_json_string()+'\n')
            for key in sorted(result.keys()):
                tfv1.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        print("***** BERT4Rec Model finished *****")


if __name__ == "__main__":
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("checkpointDir")
    flags.mark_flag_as_required("user_history_filename")
    print("Start checking for GPU")
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU found and enabled for memory growth")
    print("GPU check finished")
    tfv1.app.run()
