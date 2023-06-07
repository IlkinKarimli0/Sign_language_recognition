from keras_i3d import I3D
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

rgb_input_shape  = (20, 224, 224, 3)
flow_input_shape = (20, 224, 224, 2)

num_classes = 241

rgb_weights_path = '/home/nigar.alishzada/SLR/keras-kinetics-i3d/weights_rgb/weights-9-0.37.h5'
flow_weights_path = '/home/nigar.alishzada/SLR/keras-kinetics-i3d/weights_tvl1/weights-2-1.19.h5'

def two_stream(rgb_weights=rgb_weights_path, flow_weights = flow_weights_path,
                                             rgb_input_shape = rgb_input_shape,
                                             flow_input_shape = flow_input_shape,
                                             num_classes = num_classes):

    input_rgb = tf.keras.layers.Input(rgb_input_shape)
    input_flow = tf.keras.layers.Input(flow_input_shape)
        
    #get output from RGB side
    RGB = I3D(input_size = rgb_input_shape).model(classes =num_classes,dropout_prob=0.0, name='rgb_i3d')
    RGB.trainable = False
    RGB.load_weights(rgb_weights)

    #get output from Flow side
    Flow = I3D(input_size = flow_input_shape).model(classes =num_classes,dropout_prob=0.0,name = 'flow_i3d')
    Flow.trainable = False
    Flow.load_weights(flow_weights)


    out_rgb = RGB(input_rgb)
    out_flow = Flow(input_flow)

    result = (out_rgb + out_flow)/2

    prediction = tf.keras.layers.Activation('softmax')(result)

    return tf.keras.Model(inputs = [input_rgb,input_flow], outputs = prediction)


