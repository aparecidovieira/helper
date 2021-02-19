File for converting keras weights from retinanet to tensorflow serving format.
The notebook converts a model with backbone resnet101, while script converts a vgg19 backbone
#In case of error while loading weights, downgrade h5py version
#Should do the trick
pip install 'h5py<3.0.0'

