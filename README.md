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

### high level APIs - Estimator(include serving), dataset api and tfrecord
* higher_api.py
* create_tfrecords.py

### tensorflow eager execution mode
* eage_mode.py
* eage_mode_with_saving.py
    * includes use case of converting eager trained model to servable model
    * Please *use Tensorflow version >= 1.7*
    * tfe.Saver() -> working
    * tfe.Checkpoint() -> Not working?
    * tfe.save_network_checkpoint(), tfe.restore_network_checkpoint() -> Error occurs?
        * Leave the ```with tf.device(device)``` context for saving and loading...

### multi-gpu, muti-tower fashion(using same network on each GPU)
* multi_gpu.py, test_run.sh

## detailed tensorflow dataset api examples
* how_dataset_api_works.py
    * take a look at 'case-*.txt' files with ```test_tfrecords()```
    * note: there is **file name shuffling** as well as **dataset element shuffling** which made me confusing at first 