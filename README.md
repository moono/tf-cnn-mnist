# Tensorflow: CNN on MINST with examples
* Tensorflow coding reference document

## Tested environments
* Tensorflow version: 1.8

## Prerequisite
* Must make MNIST tfrecords with **2_create_mnist_tfrecords.py** for some examples

## Examples

### Dataset API example
* **1_dataset_api.py**
    * take a look at 'case-*.txt' files with ```test_tfrecords()```
    * note: there is **file name shuffling** as well as **dataset element shuffling** which may confusing at first
* **2_create_mnist_tfrecords.py**
    * download mnist train & test data and converts to *.tfrecord files

### Low level API example
* **3_low_level_api.py**
    * use ```tf.get_variable()```, ```tf.nn.*```
    * ```tf.Session()``` and ```tf.placeholder()```

### high level API example
* **4_1_high_level_api.py**
    * ```tf.estimator``` and ```tf.layers.*```
    * prepare trained model for tensorflow serving
* **4_2_export_trained_estimator.py**
    * Prerequisite: 4_1_high_level_api.py
    * prepare existing estimator's model_fn() for serving to use in another estimator
* **4_3_estimator_within_estimator.py**
    * Prerequisite: 4_1_high_level_api.py, 4_2_export_trained_estimator.py
    * use trained custum estimator inside another estimator
    
### eager execution mode example
* **5_eager_execution.py**
    * saving eager model with ```tfe.Saver```
    * load model and evaluate
    * includes use case of converting eager trained model to servable model

### multi-gpu example
* **6_multi_gpu.py** and **6_multi_gpu_run.sh**
    * muti-tower fashion(using same network on each GPU - data parallelism)

## References
* [MNIST-tutorial](https://www.tensorflow.org/tutorials/layers)
* [Tensorflow dataset api](https://www.tensorflow.org/programmers_guide/datasets)
* [Tensorflow estimator](https://www.tensorflow.org/programmers_guide/estimators)
* [Tensorflow eager execution](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager)
