import keras.backend as K


# from keras import backend as K
import tensorflow as tf
from tensorflow.python import saved_model
from tensorflow.python.saved_model.signature_def_utils_impl import (
    build_signature_def, predict_signature_def
)
from keras_retinanet import models
import shutil
import os


K.set_learning_phase(0)
export_path = 'new_abies_tree/export/vgg_up_20_75'
model = models.convert_model(
    model=models.backbone(backbone_name='vgg19').retinanet(num_classes=1),
    nms=True,
    class_specific_filter=True,
    anchor_params=None
)
model.load_weights('/media/hdd/workings-space/keras-retinanet-master/new_abies_tree/vgg19_up_train_20.csv_csv_75.h5')

print('Output layers', [o.name[:-2] for o in model.outputs])
print('Input layer', model.inputs[0].name[:-2])
if os.path.isdir(export_path):
    shutil.rmtree(export_path)
builder = saved_model.builder.SavedModelBuilder(export_path)

signature = predict_signature_def(
    inputs={'input': model.input},
    outputs={
        'output1': model.outputs[0],
        'output2': model.outputs[1],
        'output3': model.outputs[2]
    }
)

sess = K.get_session()
# sess.run(tf.global_variables_initializer()) 

builder.add_meta_graph_and_variables(sess=sess,
                                     tags=[saved_model.tag_constants.SERVING],
                                     signature_def_map={'predict': signature})
builder.save()
