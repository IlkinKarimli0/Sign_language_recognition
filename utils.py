import tensorflow as tf
from keras.models import Model
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import BatchNormalization
from keras.layers import Conv3D
from keras import backend as K

def top_trainable(model):
    for layer in model.layers:
        if layer.name.split('.')[0] == 'top':
            layer.trainable = True
        else:
            layer.trainable = False


def get_submodel(model, until_layer_name):
    # Get the index of the until_layer_name
    until_layer_index = -1
    for i, layer in enumerate(model.layers):
        if layer.name == until_layer_name:
            until_layer_index = i
            break

    # Check if the until_layer_name exists in the model
    if until_layer_index == -1:
        raise ValueError("Layer '{}' not found in the model.".format(until_layer_name))

    # Create a new model with layers up to the until_layer_name
    inputs = model.inputs
    outputs = model.layers[until_layer_index].output
    submodel = tf.keras.Model(inputs=inputs, outputs=outputs)

    return submodel



def conv3d_bn(x,
              filters,
              num_frames,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1, 1),
              use_bias = False,
              use_activation_fn = True,
              use_bn = True,
              name=None):
    """Utility function to apply conv3d + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv3D`.
        num_frames: frames (time depth) of the convolution kernel.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv3D`.
        strides: strides in `Conv3D`.
        use_bias: use bias or not  
        use_activation_fn: use an activation function or not.
        use_bn: use batch normalization or not.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv3D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv3D(
        filters, (num_frames, num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        name=conv_name)(x)

    if use_bn:
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 4
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    if use_activation_fn:
        x = Activation('relu', name=name)(x)

    return x

def add_top_layers(model,num_classes = 241):

    # Add top layers of the model
    # Recreating the top layer of the inception i3d model
    x = conv3d_bn(model.output, num_classes, 1, 1, 1, padding='same', use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

    num_frames_remaining = int(x.shape[1])
    x = Reshape((num_frames_remaining, num_classes))(x)

    # logits (raw scores for each class)
    x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                output_shape=lambda s: (s[0], s[2]))(x)

    x = Activation('softmax', name='prediction')(x)

    modelo = Model(inputs=model.input, outputs=x, name='i3d_inception_train')

    return modelo

def change_top_size(model,num_classes):
    model = get_submodel(model, 'top.global_avg_pool')

    # Add top layers of the model
    # Recreating the top layer of the inception i3d model
    x = conv3d_bn(model.output, num_classes, 1, 1, 1, padding='same', use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

    num_frames_remaining = int(x.shape[1])
    x = Reshape((num_frames_remaining, num_classes))(x)

    # logits (raw scores for each class)
    x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                output_shape=lambda s: (s[0], s[2]))(x)

    x = Activation('softmax', name='prediction')(x)

    modelo = Model(inputs=model.input, outputs=x, name='i3d_inception_train')

    print('New model summary!')
    modelo.summary()
    
    return modelo

