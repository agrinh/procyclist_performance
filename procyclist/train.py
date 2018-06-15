import collections
import datetime
import os
import sys
import time

import numpy as np
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.client import timeline

import procyclist.dataset

from procyclist.batch_generator import tbatch_generator
from procyclist.config import DefaultConfig
from procyclist.model import ContinuousSequenceModel
from procyclist.sessions import Sessions
from procyclist.utilities import clear_lstm_state, simple_summary, time_string


def evaluate(session, model, sessions, config, predict=True):
    n_epochs = config.n_epochs
    batch_size = config.batch_size
    max_time = config.max_time

    embeddings = np.zeros((len(sessions.data), model.config.n_hidden))
    loss = 0
    loss_count = 0
    if predict:
        predictions = [np.empty(0)] * len(sessions.data)
        prediction_op = model.predictions
    else:
        predictions = None
        prediction_op = tf.no_op()

    # Evaluate over batches
    index = None
    state = None
    for batch in tbatch_generator(sessions, config, batch_size, max_time):
        previous_index = index
        drop_mask, lengths, inputs, targets, index = batch
        feed_dict = {
            model.sequence_length: lengths,
            model.inputs: inputs,
            model.targets: targets,
        }
        if state is not None:
            # Store data for all completed sessions
            batch_embeddings = state[-1].c[drop_mask]
            embeddings[previous_index[drop_mask], :] = batch_embeddings
            # Clear lstm states and remove last unused slot
            remove = len(drop_mask) - len(inputs)
            new_state = clear_lstm_state(state, drop_mask, remove)
            feed_dict[model.initial_state] = new_state

        batch_predictions, batch_loss, state = session.run(
            fetches=[prediction_op, model.loss, model.state],
            feed_dict=feed_dict
        )

        # Maintain sum and count for average loss
        loss += batch_loss
        loss_count += 1

        # Append predictions if any are predicted
        if predict:
            for i, j in enumerate(index):
                predictions[j] = np.append(
                    predictions[j], batch_predictions[i, :lengths[i]])
    # Store last embeddings
    embeddings[index, :] = state[-1].c
    return np.array(predictions), loss / loss_count, embeddings


def train_model(model, sessions, config):
    n_epochs = config.n_epochs
    batch_size = config.batch_size
    max_time = config.max_time

    # Set up logging and checkpoint paths
    time_str = time_string()
    log_path = os.path.join('log', time_str)
    checkpoint_path = os.path.join(log_path, 'model.ckpt')
    print('Logging to: %s' % log_path)

    # Separate into training and validation sets
    training, validation = sessions.split(0.7, 'cyclist')

    # Create summary for cost func.
    with tf.variable_scope('training'):
        loss_op = tf.scalar_summary('Loss (MSE)', model.loss)
        grad_norm_op = tf.scalar_summary('Gradient norm', model.grad_norm)
        summary_op = tf.summary.merge_all()

    with tf.variable_scope('embeddings'):
        with tf.device('/cpu:0'):
            embeddings_shape = [len(sessions.data), model.config.n_hidden]
            embeddings_var = tf.get_variable(
                'embeddings',
                embeddings_shape,
                initializer=tf.constant_initializer(0),
                dtype=tf.float32,
                trainable=False
            )

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)
    with tf.Session() as session:
        summary_writer = tf.train.SummaryWriter(log_path, session.graph)
        session.run(tf.initialize_all_variables())

        # Set up embedding storage
        projector_config = projector.ProjectorConfig()
        embedding = projector_config.embeddings.add()
        embedding.tensor_name = embeddings_var.name
        embedding.metadata_path = os.path.join(log_path, 'metadata.tsv')
        sessions.meta.to_csv(embedding.metadata_path, sep='\t')

        # Store config
        config.save(os.path.join(log_path, 'configuration'))

        # Saves a configuration file that TensorBoard will read during startup
        projector.visualize_embeddings(summary_writer, projector_config)

        best_loss = float('inf')
        batch_counter = 0
        try:
            for epoch in range(n_epochs):
                # Train over batches
                state = None
                max_time_fraction = 1 / (1 + np.exp(- epoch / 2))**2
                epoch_time = int(max_time * max_time_fraction)
                epoch_size = int(batch_size / max_time_fraction)
                batches = tbatch_generator(
                    sessions,
                    config,
                    epoch_size,
                    epoch_time
                )
                for batch in batches:
                    drop_mask, lengths, inputs, targets, index = batch

                    # Stop when batch size below threshold
                    if len(inputs) < epoch_size * 0.8:
                        break

                    feed_dict = {
                        model.dropout: config.dropout,
                        model.inputs: inputs,
                        model.sequence_length: lengths,
                        model.targets: targets,
                    }
                    if state is not None:
                        # Clear lstm states and remove last unused slot
                        remove = len(drop_mask) - len(inputs)
                        new_state = clear_lstm_state(state, drop_mask, remove)
                        feed_dict[model.initial_state] = new_state

                    print('Sequence length in each buffer slot:')
                    print(lengths)

                    start = time.time()
                    _, loss, summary, state = session.run(
                        fetches=[
                            model.train_op,
                            model.loss,
                            summary_op,
                            model.state
                        ],
                        feed_dict=feed_dict
                    )
                    end = time.time()
                    batch_counter += 1

                    # Write summaries
                    summary_writer.add_summary(summary, batch_counter)
                    metric_summary = simple_summary({
                        'Epoch': epoch,
                        'Batch size': len(inputs),
                        'Max time': epoch_time,
                        'Runtime': end - start
                    })
                    summary_writer.add_summary(metric_summary, batch_counter)

                if epoch % 2 == 0:
                    # Evaluate on entire dataset
                    _, loss, _ = evaluate(
                        session, model, validation, config, predict=False)

                    # Save checkpoint and summaries
                    saver.save(session, checkpoint_path, epoch)
                    validation_summary = simple_summary({
                        'Validation loss (MSE)': loss
                    })
                    summary_writer.add_summary(validation_summary, epoch)

        except (KeyboardInterrupt, SystemExit):
            print('Training stopped. Please wait while checkpointing and' +
                  ' evaluating model.')

        # Return predictions and loss for full dataset
        print('Evaluating on all data')
        predictions, loss, embeddings = evaluate(
            session, model, sessions, config)

        # Update embeddings tensor
        session.run(embeddings_var.assign(embeddings))

        # Save checkpoint and summaries
        saver.save(session, checkpoint_path, epoch + 1)
        end_summary = simple_summary({"Average loss (MSE)": loss})
        summary_writer.add_summary(end_summary, epoch)
        return predictions, loss, embeddings, validation.meta.index


def main():
    # Load data
    sessions = procyclist.dataset.load()

    # Initialize config and model
    i_out = [sessions.parameters.index('Heart Rate')]
    i_in = [i for i in range(len(sessions.parameters)) if i not in i_out]
    config = DefaultConfig(inputs=i_in, outputs=i_out)
    model = ContinuousSequenceModel(config)

    # Train model.
    predictions, loss, embeddings, validation_ix = train_model(
        model=model,
        sessions=sessions,
        config=config
    )

    # Print loss
    print('Average batch MSE: %.8f' % loss)

    # Store sequences and predictions for later use
    np.save('predictions', predictions)
    np.save('embeddings', embeddings)
    np.save('validation_ix', validation_ix)


if __name__ == '__main__':
    main()
