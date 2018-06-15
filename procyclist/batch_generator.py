from itertools import repeat

import numpy as np

from procyclist.utilities import SequenceBuffer


def batch_generator(sessions, config, batch_size, max_length=None):
    """
    Chunks up raw_data into shuffled batches

    Yields sequence lengths, inputs and targets for each batch.
    """
    if max_length is None:
        max_length = sessions.meta.length.max()

    # Extract inputs and targets from shuffled sessions.
    n_sessions = len(sessions.data)
    indices = np.arange(n_sessions)
    np.random.shuffle(indices)

    for sample in np.split(indices, range(batch_size, n_sessions, batch_size)):
        lengths = sessions.meta.length[sample]
        batch_max_length = min(max_length, lengths.max())
        data_dim = config.input_dim + config.output_dim
        container = np.zeros((len(lengths), batch_max_length, data_dim))
        for i, session in enumerate(sessions.data[sample]):
            session = session[:batch_max_length]
            container[i, :session.shape[0], :session.shape[1]] = session

        inputs = container[:, :, config.inputs]
        targets = container[:, :, config.outputs]
        yield lengths.values, inputs, targets


def tbatch_generator(sessions, config, batch_size, max_length=None):
    """
    Produce time truncated batches

    Yields mask, sequence lengths, inputs and targets for each batch. Please
    note that the same matrix is yielded each time but with content changed.
    The mask returned is inteded for syncronizing cell states in RNNs. It will
    have the value True on entries that are no longer part of the previous
    timeseries.
    """
    if max_length is None:
        max_length = sessions.meta.length.max()

    # Extract inputs and targets from shuffled sessions.
    n_sessions = len(sessions.data)
    indices = np.arange(n_sessions)
    np.random.shuffle(indices)

    # Create sequence buffers
    get = lambda arr: (arr[i] for i in indices)
    data = SequenceBuffer(batch_size, get(sessions.data))
    length = SequenceBuffer(batch_size, get(sessions.meta.length), np.int32)
    start = SequenceBuffer(batch_size, repeat(0, n_sessions), np.int32)
    index = SequenceBuffer(batch_size, iter(indices), np.int32)
    tss = SequenceBuffer(batch_size, get(sessions.meta['TSS']), np.int32)

    # Define container for data
    data_dim = config.input_dim + config.output_dim
    container_shape = (len(data), max_length, data_dim)
    container = np.zeros(container_shape, dtype=np.float32)

    # Set limits before first iteration
    remaining = None
    drop_mask = np.zeros(len(data))
    inserted = 0
    while len(data):
        if remaining is not None:
            # Evict where the complete sequence has been depleted
            drop_mask = remaining <= max_length
            inserted += drop_mask.sum()
            index.evict(drop_mask)
            length.evict(drop_mask)
            start.evict(drop_mask)
            tss.evict(drop_mask)
            sync_index = data.evict(drop_mask)

            # Drop rows if slots in buffers not completely filled anymore
            container = container[sync_index]

        # Determine largest limit from start in sequnce under max_length
        start_idx = start.content
        remaining = length.content - start_idx
        limits = np.minimum(remaining, max_length)

        # Fill container with sequences from buffers
        for i, session in enumerate(data.content):
            session_slice = session[start_idx[i]:(start_idx[i] + limits[i])]
            container[i, :limits[i], :session.shape[1]] = session_slice

        start_idx += max_length
        inputs = container[:, :, config.inputs]
        targets = container[:, :, config.outputs]
        #targets = tss.content.reshape((-1, 1))
        if len(container):  # Do not emit empty data
            yield drop_mask, limits, inputs, targets, index.content
