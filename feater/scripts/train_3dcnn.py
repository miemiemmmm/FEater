import time

import numpy as np
import jax
import optax
import jax.numpy as jnp
from jax import jit, value_and_grad
import dask.array as da

from feater import models

# Use CPU platform instead of GPU
# from jax.config import config
# config.update("jax_platform_name", "cpu")


def gen_test_data(sample_nr = 1000):
  st = time.perf_counter()
  chunks = (sample_nr, 3, 32, 32, 32)  # Adjust this to a suitable size according to your memory capacity
  rand_td = da.random.normal(size=(sample_nr, 3, 32, 32, 32), chunks=chunks)
  rand_td.compute()
  rand_td = np.asarray(rand_td, dtype=np.float32)
  rand_ld = np.random.normal(size=(sample_nr, 1))

  print("Test input data preparation: ", rand_td.shape, f"Time elapsed: {time.perf_counter() - st:.3f} s")
  return rand_td, rand_ld


def load_test_data():
  pass


@jit
def train_step(params, opt_state, inputs, targets):
  loss, grads = value_and_grad(loss_fn)(params, inputs, targets)
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss


def train(training_data, label_data, params, opt_state):
  st = time.perf_counter()
  num_batches = training_data.shape[0] // BATCH_SIZE
  for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}\n------------------------------------------------")
    running_loss = 0.0
    for i in range(num_batches):
      start = i * BATCH_SIZE
      end = start + BATCH_SIZE
      inputs = training_data[start:end]
      targets = label_data[start:end]
      params, opt_state, batch_loss = train_step(params, opt_state, inputs, targets)
      running_loss += batch_loss
      if ((i+1) % 50) == 0:
        print(f"Epoch{epoch+1}, batch{i+1:5d}: running loss: {running_loss:.3f}")
        running_loss = 0.0
  print(f'Finished Training, Total time elapsed: {time.perf_counter()-st:.3f} seconds')

def count_parameters(params):
  for param in jax.tree_util.tree_leaves(params):
    print(param.size)
  return sum(param.size for param in jax.tree_util.tree_leaves(params))
  # return sum(jnp.prod(param.shape) for param in jax.tree_util.tree_leaves(params))

if __name__ == "__main__":
  EPOCHS = 15
  BATCH_SIZE = 32

  # Read the data
  rf_training_data, label_training_data = gen_test_data()
  print(f"Raw input data shape: {rf_training_data.shape}")
  training_data = models.bchw_to_bhwc(rf_training_data)
  print(f"Training input data shape: {training_data.shape}")

  def loss_fn(params, inputs, targets):
    preds = model.apply(params, inputs)
    return jnp.mean((preds - targets) ** 2)  # mean squared error

  # Initialize the model
  model = models.CNN3D_JAX()
  params = model.init(jax.random.PRNGKey(100), jnp.ones((1, 32, 32, 32, 3)))
  # Create optimizer
  optimizer = optax.adam(0.001)
  opt_state = optimizer.init(params)
  # print(jax.tree_util.tree_leaves(params))
  print("Number of parameters in the model:", count_parameters(params))

  # Convert the data to jax array
  # Assume training_data and labels are numpy array np.float32
  training_data = jnp.asarray(training_data, dtype=np.float32)
  label_training_data = jnp.asarray(label_training_data, dtype=np.float32).reshape(-1, 1)

  train(training_data, label_training_data, params, opt_state)




