import psiz
import tensorflow as tf

# Load observations (and corresponding catalog) for a predefined dataset.
(obs, catalog) = psiz.datasets.load('birds-16')
n_stimuli = catalog.n_stimuli

# Create a 2-dimensional embedding layer for the stimuli.
# NOTE: Since we will use masking, we increment n_stimuli by one.
n_dim = 2
stimuli = tf.keras.layers.Embedding(
    catalog.n_stimuli+1, n_dim, mask_zero=True
)

# Use a default similarity kernel (a fully trainable exponential with
# weighted Minkowski distance).
kernel = psiz.keras.layers.DistanceBased()

# Create a `Rank` model.
model = psiz.keras.models.Rank(stimuli=stimuli, kernel=kernel)

# Compile the model.
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
)

# Fit the psychological embedding using all observations.
model.fit(obs.as_dataset(), epochs=10)

# Save and load the fitted model.
model.save('my_embedding')
reconstructed_model = tf.keras.models.load_model('my_embedding')
