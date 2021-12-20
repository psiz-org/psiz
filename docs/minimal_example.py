import psiz
import tensorflow as tf

# Load ordinal similarity observations (and corresponding stimuli catalog) for
# a previously colelcted dataset.
(obs, catalog) = psiz.datasets.load_dataset('birds-16')
n_stimuli = catalog.n_stimuli

# Create a 2-dimensional embedding layer that will represent the psychological
# space of the stimuli.
# NOTE: In general, we assume masking (i.e., `mask_zero=True`), so we
# increment `n_stimuli` by one.
n_dim = 2
stimuli = tf.keras.layers.Embedding(
    catalog.n_stimuli + 1, n_dim, mask_zero=True
)

# Use a default similarity kernel (a fully trainable exponential similarity
# function that uses a weighted Minkowski distance function).
kernel = psiz.keras.layers.DistanceBased()

# Create a `Rank` model by supplying the above sub-components.
model = psiz.keras.models.Rank(stimuli=stimuli, kernel=kernel)

# Compile the model. We use categorical crossentropy because a `Rank` model
# will output probabilities associated with each distinct outcome.
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
)

# Fit the psychological embedding using all observations and a few epochs.
model.fit(obs.as_dataset(), epochs=10)

# The model can be saved and loaded.
model.save('my_embedding')
reconstructed_model = tf.keras.models.load_model('my_embedding')
