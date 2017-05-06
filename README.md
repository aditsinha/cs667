# Differentially Private Training of Logistic Regression Models

This repository contains multiple tools for training and simulating
the training of differentially private logistic regression models.

## Compilation

### Dependencies

* [OblivC](https://oblivc.org/)
* [OpenMP](http://www.openmp.org/)
* [Eigen](http://eigen.tuxfamily.org)

### Building

Edit the Makefile so that the `CILPATH` variable points to the root
directory of the OblivC installation.

Run `make all` in order to compile all of the executables

## Running

This project contains multiple executable, each with a different
purpose.

### Configuration

All executables use a configuration file of the same format.  The
configuration file contains key-value pairs, each on a separate line,
where the key and value are separated by an "=".  See
"example_config.txt" for an example of the format.  The following is a
list of all possible configuration options along with the data type
and descriptions.

* `num_parties` integer. The number of parties that participate in training the
  model
* `num_data_rows` integer. The number of training examples for each party.  We
  assume that all parties have the same number of training examples
* `num_validation_rows` integer. The number of validation examples.  There is
  one common validation set; each party does not have its own
  validation set.
* `num_dimensions` integer. The number of dimensions in the training data
* `gradient_clip` float. Per example gradient clipping threshold.
* `batch_size` integer.  The batch size for each party in one training batch
* `epochs` integer.  The number of adjusted epochs.  For a single
  party, this is the number of training epochs.  For multiple parties,
  the number of training epochs is this number divided by the number
  of parties.
* `fractional_bits` integer.  The number of bits of precision to use
  after the radix point when converting real numbers to integers
* `privacy` float,float. First is the value of epsilon and the second
  is the value of delta for differential privacy.
* `initial_learning_rate` float.  Learning rate for the training epoch
* `learning_rate_decay` float.  Decay for time-based learning rate
  schedule
* `normalization` float OR comma separated list of float with
  `num_dimensions` entries.  Cannot appear before `num_dimensions`.
  Applies a scaling to each column of the training and validation
  features.

### Data

The program expects all data in the CSV file format.  The first column
should be a 0 or 1 corresponding to the label of the data.  The
subsequent columns should contain the features for each entry.  There
must be a comma at the end of each row.  `./gen <dimensions> <rows>
<noise>` can be used to generate a dataset inside the unit ball with
zero bias.  The dataset will be linearly separable if `<noise>` is 0.

The training and test datasets from MNIST in the proper format can be
found in mnist\_training.csv and mnist\_test.csv respectively.
Numbers 0-4 are assigned the label 0 and 5-9 are assignet the label 1.

### Executables

The following executables that can be used for training models.  Every
executable reads the following configuration variables

* `./train <config> <data> (<validation>)` Simulates training of a
  model with the provided configuration and data.  If `num_parties` is
  0, then we train using a single party without adding any
  differential privacy.  Otherwise, add noise to guarantee the
  differential privacy specified by `privacy`.  Reads every party's
  data from the same data file, with the first `num_data_rows` rows
  going to the first party, the second block going to the second
  party, etc.  If `<validation>` is provided, read validation examples
  from a different file, otherwise read the validation rows from the
  file specified in `<data>` after all training rows have been read.
* `./gradient_yao <port> <host>|-- <config> <data> (<validation>)`
  Trains a model using a garbled circuit by calculating gradients
  locally and updating the model within the circuit.  Ignores the
  `num_parties` option.  The first party must run the programm using
  "--" and the second party must provide the hostname of the first
  party.  The parties should provide different data files.  Only the
  first party will evaluate the model, so `<validation>` is ignored
  for the second party.  If `<validation>` is not provided by the
  first party, then read validation examples after the training data.
* `./full_yao <port> <host>|-- <config> <data> (<validation>)` Trains
  a model entirely within a garbled circuit.  Ignores the
  `num_parties`, `gradient_clip`, and `fractional_bits` options.  In
  order to change the number of bits used after the radix point in
  fixed point arithmetic, the user must change the `PRECISION`
  constant in obliv\_math\_def.h.  The same convention of how to run
  each party as described with the `./gradient_yao` applies to this
  program.
* `./full_yao_simulator <config> <data1> (<data2> (<validation>))`
  Simulates all of the fixed point arithmetic used in the `./full_yao`
  implementation.  Ignores the same configuration options as
  `./full_yao`.  If `<data2>` is not provided, then read all training
  and validation data from `<data1>`.  If `<data2>` is provided, but
  `<validation>` is not provided, then reads validation examples after
  the training examples in `<data1>`.
