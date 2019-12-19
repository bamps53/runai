# Run:AI Gradient Accumulation

A generic gradient accumulation wrapper for Keras optimizers

## Getting Started

Follow the next instructions to use gradient accumulation in your own Keras models, or with one of our examples of basic Keras models.

### Installing

Install the `runai` Python library using `pip` using the following command:

```
pip install runai
```

> Make sure to use the correct `pip` installer (you might need to use `pip3` for Python3)

### Usage

First you need to add the following import command to your code:

```
import runai.ga
```

Then, you need to choose one of the two possible ways to use gradient accumulation:

#### Wrap an existing optimizer with gradient accumulation

In case you have an instance of a `Keras.Optimizer`, especially if you have implemented an optimizer by yourself, you should wrap your `optimizer` using the following line:

```
optimizer = runai.ga.keras.optimizers.Optimizer(optimizer, steps=STEPS)
```

#### Create a gradient-accumulated common optimizer

You can create an instance of any common Keras optimizer, already wrapped with gradient accumulation, out of the box. Just create an instance of the your selected optimizer from `runai.ga.keras.optimizers`. For example, to create an `Adam` optimizer, wrapped with gradient accumulation, you can use the following command:

```
optimizer = runai.ga.keras.optimizers.Adam(steps=STEPS)
```

Both ways require an argument `steps` to indicate the number of steps to accumulate gradients over (in the code above we are accumulating the gradients of `STEPS` steps).

> *NOTE:* It is not mandatory to pass `steps` as a keyword argument to the creation of the instance, and the `steps=` prefix may be removed

## Examples

Examples of wrappring Keras optimizers with gradient accumulation exist under the [examples](../../examples/ga/keras) directory:

* [VGG16](../../examples/ga/keras/vgg16.py) - **Recommended** - A very lean and simple example using the Keras builtin implementation of VGG16 on the CIFAR10 dataset
* [MLP](../../examples/ga/keras/mlp.py) - A very simple implementation of an MLP network on the MNIST dataset

> *NOTE:* The examples were tested using Python 3.6, Keras 2.2.4, TensorFlow 1.15.0
