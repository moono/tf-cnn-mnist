# Learn Tensorflow by example - just give me the code!!
* For someone who wants to learn how Tensorflow's high level api works
* Currently I'm learning too

## Build cnn on mnist with Tensorflow
* Tensorflow version: 1.6
* Reference
    * [MNIST-tutorial](https://www.tensorflow.org/tutorials/layers)
    * [Tensorflow dataset api](https://www.tensorflow.org/programmers_guide/datasets)
    * [Tensorflow estimator](https://www.tensorflow.org/programmers_guide/estimators)
    * [Tensorflow eager execution](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager)

### tf.Session and tf.layers
* tf-official-tutorial.py

### multi-gpu, muti-tower fashion(using same network on each GPU)
* multi_gpu.py, test_run.sh

### high level APIs - Estimator(include serving), dataset api and tfrecord
* higher_api.py
* create_tfrecords.py
