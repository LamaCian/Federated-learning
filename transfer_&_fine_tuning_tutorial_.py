#!/usr/bin/env python
# coding: utf-8

# In[1]:




import nest_asyncio
nest_asyncio.apply()


# In[2]:


import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

np.random.seed(0)

tff.federated_computation(lambda: 'Hello, World!')()


# In[3]:


cifar100_train, cifar100_test = tff.simulation.datasets.cifar100.load_data()


# In[4]:


len(cifar100_train.client_ids)


# In[5]:


cifar100_train.element_type_structure


# In[6]:


example_dataset = cifar100_train.create_tf_dataset_for_client(
cifar100_train.client_ids[0])
example_element = next(iter(example_dataset))


# In[7]:


from matplotlib import pyplot as plt
plt.imshow(example_element['image'].numpy(), cmap='gray', aspect='equal')
plt.grid(False)
_ = plt.show()


# ### Exploring heterogeneity in federated data
# 
# 

# In[8]:


## Example MNIST digits for one client
figure = plt.figure(figsize=(20, 4))
j = 0

for example in example_dataset.take(40):
  plt.subplot(4, 10, j+1)
  plt.imshow(example['image'].numpy(), cmap='gray', aspect='equal')
  plt.axis('off')
  j += 1


# ### Preprocessing the input data

# In[ ]:





# In[9]:


NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

def preprocess(dataset):

  def batch_format_fn(element):
    """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
    return collections.OrderedDict(
#         x=tf.reshape(element['image'], [-1, 1024]),
        x = element['image'],
        y = element['label'])

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


# In[10]:


preprocessed_example_dataset = preprocess(example_dataset)

sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                     next(iter(preprocessed_example_dataset)))

sample_batch


# In[11]:


def make_federated_data(client_data, client_ids):
  return [
      preprocess(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]


# In[12]:


sample_clients = cifar100_train.client_ids[0:NUM_CLIENTS]

federated_train_data = make_federated_data(cifar100_train, sample_clients)

print(f'Number of client datasets: {len(federated_train_data)}')
print(f'First dataset: {federated_train_data[0]}')


# ## Creating a model with Keras
# 
# 

# In[13]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications


# In[14]:


def create_keras_model():
    model = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', 
                                                input_tensor=tf.keras.layers.Input(shape=(32,32,3)),
                                                pooling=None)
    return model


# In[15]:


def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


# In[16]:


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))


# In[17]:


print(iterative_process.initialize.type_signature.formatted_representation())


# **Note:** we do not compile the model yet. The loss, metrics, and optimizers are introduced later.
# 
# In order to use any model with TFF, it needs to be wrapped in an instance of the
# `tff.learning.Model` interface, which exposes methods to stamp the model's
# forward pass, metadata properties, etc., similarly to Keras, but also introduces
# additional elements, such as ways to control the process of computing federated
# metrics. Let's not worry about this for now; if you have a Keras model like the
# one we've just defined above, you can have TFF wrap it for you by invoking
# `tff.learning.from_keras_model`, passing the model and a sample data batch as
# arguments, as shown below.

# In[ ]:





# ## Training the model on federated data
# 
# Now that we have a model wrapped as `tff.learning.Model` for use with TFF, we
# can let TFF construct a Federated Averaging algorithm by invoking the helper
# function `tff.learning.build_federated_averaging_process`, as follows.
# 
# Keep in mind that the argument needs to be a constructor (such as `model_fn`
# above), not an already-constructed instance, so that the construction of your
# model can happen in a context controlled by TFF (if you're curious about the
# reasons for this, we encourage you to read the follow-up tutorial on
# [custom algorithms](custom_federated_algorithms_1.ipynb)).
# 
# One critical note on the Federated Averaging algorithm below, there are **2**
# optimizers: a _client_optimizer_ and a _server_optimizer_. The
# _client_optimizer_ is only used to compute local model updates on each client.
# The _server_optimizer_ applies the averaged update to the global model at the
# server. In particular, this means that the choice of optimizer and learning rate
# used may need to be different than the ones you have used to train the model on
# a standard i.i.d. dataset. We recommend starting with regular SGD, possibly with
# a smaller learning rate than usual. The learning rate we use has not been
# carefully tuned, feel free to experiment.

# In[18]:


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))


# What just happened? TFF has constructed a pair of *federated computations* and
# packaged them into a `tff.templates.IterativeProcess` in which these computations
# are available as a pair of properties `initialize` and `next`.
# 
# In a nutshell, *federated computations* are programs in TFF's internal language
# that can express various federated algorithms (you can find more about this in
# the [custom algorithms](custom_federated_algorithms_1.ipynb) tutorial). In this
# case, the two computations generated and packed into `iterative_process`
# implement [Federated Averaging](https://arxiv.org/abs/1602.05629).
# 
# It is a goal of TFF to define computations in a way that they could be executed
# in real federated learning settings, but currently only local execution
# simulation runtime is implemented. To execute a computation in a simulator, you
# simply invoke it like a Python function. This default interpreted environment is
# not designed for high performance, but it will suffice for this tutorial; we
# expect to provide higher-performance simulation runtimes to facilitate
# larger-scale research in future releases.
# 
# Let's start with the `initialize` computation. As is the case for all federated
# computations, you can think of it as a function. The computation takes no
# arguments, and returns one result - the representation of the state of the
# Federated Averaging process on the server. While we don't want to dive into the
# details of TFF, it may be instructive to see what this state looks like. You can
# visualize it as follows.

# In[19]:


print(iterative_process.initialize.type_signature.formatted_representation())


# While the above type signature may at first seem a bit cryptic, you can
# recognize that the server state consists of a `model` (the initial model
# parameters for MNIST that will be distributed to all devices), and
# `optimizer_state` (additional information maintained by the server, such as the
# number of rounds to use for hyperparameter schedules, etc.).
# 
# Let's invoke the `initialize` computation to construct the server state.

# In[20]:


state = iterative_process.initialize()


# The second of the pair of federated computations, `next`, represents a single
# round of Federated Averaging, which consists of pushing the server state
# (including the model parameters) to the clients, on-device training on their
# local data, collecting and averaging model updates, and producing a new updated
# model at the server.
# 
# Conceptually, you can think of `next` as having a functional type signature that
# looks as follows.
# 
# ```
# SERVER_STATE, FEDERATED_DATA -> SERVER_STATE, TRAINING_METRICS
# ```
# 
# In particular, one should think about `next()` not as being a function that runs on a server, but rather being a declarative functional representation of the entire decentralized computation - some of the inputs are provided by the server (`SERVER_STATE`), but each participating device contributes its own local dataset.
# 
# Let's run a single round of training and visualize the results. We can use the
# federated data we've already generated above for a sample of users.

# In[23]:


import time
start = time.time()
state, metrics = iterative_process.next(state, federated_train_data)
end = time.time()
total_time = end-start
print('round  1, metrics:{0}, time: {1} '.format(metrics,total_time))


# Let's run a few more rounds. As noted earlier, typically at this point you would
# pick a subset of your simulation data from a new randomly selected sample of
# users for each round in order to simulate a realistic deployment in which users
# continuously come and go, but in this interactive notebook, for the sake of
# demonstration we'll just reuse the same users, so that the system converges
# quickly.

# In[ ]:


# NUM_ROUNDS = 11
# for round_num in range(2, NUM_ROUNDS):
#   state, metrics = iterative_process.next(state, federated_train_data)
#   print('round {:2d}, metrics={}'.format(round_num, metrics))


# Training loss is decreasing after each round of federated training, indicating
# the model is converging. There are some important caveats with these training
# metrics, however, see the section on *Evaluation* later in this tutorial.

# ## Fine tunning

# Typical transfer learning workflow can be implemented in Keras:
# 
# 1. Instantiate a base model and load pre-trained weights into it.
# 2. Freeze all layers in the base model by setting trainable = False.
# 3. Create a new model on top of the output of one (or several) layers from the base model.
# 4. Train your new model on your new dataset.

# In[26]:


def create_transfer_learning_keras():
    base_model = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', 
                                                input_tensor=tf.keras.layers.Input(shape=(32,32,3)),
                                                pooling=None)
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=(32,32,3))
    x = base_model(inputs, training = False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
# A Dense classifier with a single unit (binary classification)
    outputs = tf.keras.layers.Dense(100)(x)
    model = tf.keras.Model(inputs, outputs)
    
#     tf.keras.Model.save_weights(model,'/Users/admin/Desktop/AML_intro/', save_format = 'tf')
    
    return model


# In[ ]:


#


# In[27]:


def create_FL_model():

  keras_model = create_transfer_learning_keras()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


# In[28]:


transfer_learning_iterative_process = tff.learning.build_federated_averaging_process(
    create_FL_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))


# In[30]:


state = transfer_learning_iterative_process.initialize()


# In[32]:


state, metrics = transfer_learning_iterative_process.next(state, federated_train_data)


# In[ ]:





# In[33]:


def fine_tuning_keras_model():
    model = create_transfer_learning_keras()
    #load model keras fn #path to weights file
    
    state.model.assign_weights_to(model)

#     tf.keras.Model.load_weights(model,filepath = '/Users/admin/Desktop/AML_intro/')
   #load_weights('/Users/admin/Desktop/AML_intro/checkpoint')#     model.trainable = true
    #go through all layer and set trainable to true
    model.trainable = True
    return model

def fine_tuning_FL_model():
    model = fine_tuning_keras_model()
    return tff.learning.from_keras_model(
    model,
    input_spec=preprocessed_example_dataset.element_spec,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


# In[34]:



fine_tuning_iterative_process = tff.learning.build_federated_averaging_process(
    fine_tuning_FL_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.00002),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.0001))


# In[35]:


fine_tuning_state = fine_tuning_iterative_process.initialize()


# In[36]:


fine_tuning_state, metrics = fine_tuning_iterative_process.next(fine_tuning_state, federated_train_data)


# Once your model has converged on the new data, you can try to unfreeze all or part of the base model and retrain the whole model end-to-end with a very low learning rate.
# 
# It is critical to only do this step after the model with frozen layers has been trained to convergence. If you mix randomly-initialized trainable layers with trainable layers that hold pre-trained features, the randomly-initialized layers will cause very large gradient updates during training, which will destroy your pre-trained features.
# 
# It's also critical to use a very low learning rate at this stage, because you are training a much larger model than in the first round of training, on a dataset that is typically very small. As a result, you are at risk of overfitting very quickly if you apply large weight updates. Here, you only want to readapt the pretrained weights in an incremental way.
# 
# 

# In[ ]:




