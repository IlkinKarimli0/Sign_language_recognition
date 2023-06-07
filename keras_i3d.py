import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
from keras.optimizers import SGD, Adam


class I3D:
    def __init__(self,input_size = (20,224,224,3)):
        
        #defining input size and declearing Input layer based on that
        self.input_size = input_size
        self.Input = tf.keras.layers.Input(input_size)
        
        # Defining layers that we will use within this model
        self.Conv3D = tf.keras.layers.Conv3D
        self.MaxPooling3D = tf.keras.layers.MaxPooling3D
        self.BatchNorm = tf.keras.layers.BatchNormalization
        
        self.activation = tf.keras.layers.Activation
        self.MaxP3D = tf.keras.layers.MaxPooling3D
        self.Avg3DPool = tf.keras.layers.AveragePooling3D
        self.Lambda = tf.keras.layers.Lambda
        
        self.Dropout = tf.keras.layers.Dropout
        self.Reshape = tf.keras.layers.Reshape
        self.Flatten = tf.keras.layers.Flatten
        self.Dense = tf.keras.layers.Dense

    def conv3d_bn(self,x,
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
        """
        Utility function to apply conv3d + BN.

        Arg:
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
        Returns
            Output tensor after applying `Conv3D` and `BatchNormalization`.
        """

        #Name declaration
        if name is not None:
            bn_name = name + '.bn'
            conv_name = name + '.conv3d'
        else:
            bn_name = None
            conv_name = None

        #Convolutional layer we want to pass through
        x = self.Conv3D(
            filters, (num_frames, num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            name=conv_name)(x)

        # Adding Batch Normalization after convolutional block
        if use_bn:
            if K.image_data_format() == 'channels_first':
                bn_axis = 1
            else:
                bn_axis = 4
            x = self.BatchNorm(axis=bn_axis, scale=False, name=bn_name)(x)

        # Relu Activation to prevent negative noise in case needed
        if use_activation_fn:
            x = self.activation('relu', name=name)(x)

        return x
    
    def Inception3d_module(self,inpt,filter_sizes:tuple, name:str,channel_axis = 4):
        """
        Inception block in model 

        Args:
            inpt: input to the inception block
            filter_sizes = tuple with six integers which define the filter sizes inside class
            name = Name of the blocks
            channels_axis = axis of the color channel for each frame

        Returns: 
            Processed output of the input through inception layers
        """
        
        f1,f2,f3,f4,f5,f6 = filter_sizes
        
        # First branch
        branch_0 = self.conv3d_bn(inpt, f1, 1, 1, 1, padding='same', name=name+'.b0')

        # Second branch
        branch_1 = self.conv3d_bn(inpt, f2, 1, 1, 1, padding='same', name=name+'.b1a')
        branch_1 = self.conv3d_bn(branch_1, f3, 3, 3, 3, padding='same', name=name+'.b1b')

        # Third branch
        branch_2 = self.conv3d_bn(inpt, f4, 1, 1, 1, padding='same', name=name+'.b2a')
        branch_2 = self.conv3d_bn(branch_2, f5, 3, 3, 3, padding='same', name=name+'.b2b')

        # Fourth branch
        branch_3 = self.MaxP3D((3, 3, 3), strides=(1, 1, 1), padding='same', name=name+'MaxPool2d')(inpt)
        branch_3 = self.conv3d_bn(branch_3, f6, 1, 1, 1, padding='same', name=name+'.b3b')

        # Concatenate all the outputs from all the branches
        x =  tf.keras.layers.concatenate(
            [branch_0, branch_1, branch_2, branch_3],
            axis=channel_axis,
            name='Mixed_'+name)
        
        return x
    
    def model(self,include_top = True,dropout_prob = 0.3,classes = 261,name = 'i3d_inception'):
        """
        Functional API function to create structure of the model 

        Args:
            Include_top: boolen whether to get top layers which are the layer after average pooling including average pooling
            Dropout_prob: percentage of the neurans that will turn of at layer that dropout added. Dropout layer randomly sets 
                          a fraction of the inputs to zero during training
            Classes: Number of the classes to classify

        Returns: 
            Keras Model 

        """
        x = self.conv3d_bn(self.Input, 64, 7, 7, 7, strides=(2, 2, 2), padding='same', name='Conv3d_1a_7x7')

        x = self.MaxP3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_2a_3x3')(x)
        x = self.conv3d_bn(x, 64, 1, 1, 1, strides=(1, 1, 1), padding='same', name='Conv3d_2b_1x1')
        x = self.conv3d_bn(x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same', name='Conv3d_2c_3x3')
        
        x = self.MaxP3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_3a_3x3')(x)
        
        x = self.Inception3d_module(x,(64,96,128,16,32,32),name = 'Mixed_3b')
        x = self.Inception3d_module(x,(128,128,192,32,96,64),name = 'Mixed_3c')
        
        x = self.MaxP3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='MaxPool2d_4a_3x3')(x)
            
        x = self.Inception3d_module(x,(192,96,208,16,48,64),name = 'Mixed_4b')
        x = self.Inception3d_module(x,(160,112,224,24,64,64),name = 'Mixed_4c')
        x = self.Inception3d_module(x,(128,128,256,24,64,64),name = 'Mixed_4d')
        x = self.Inception3d_module(x,(112,144,288,32,64,64),name = 'Mixed_4e')
        x = self.Inception3d_module(x,(256,160,320,32,128,128),name = 'Mixed_4f')
        
        x = self.MaxP3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='MaxPool2d_5a_2x2')(x)
        
        x = self.Inception3d_module(x,(256,160,320,32,128,128),name = 'Mixed_5b')
        x = self.Inception3d_module(x,(384,192,384,48,128,128),name = 'Mixed_5c')
        
        if include_top:
            x = self.Avg3DPool((2, 7, 7), strides=(1, 1, 1), padding='valid', name='top.global_avg_pool')(x)
            x = self.Dropout(dropout_prob,name = 'top.dropout')(x)
            x = self.conv3d_bn(x, classes, 1, 1, 1, padding='same', 
                    use_bias=True, use_activation_fn=False, use_bn=False, name='top')
            num_frames_remaining = int(x.shape[1])
            x = self.Reshape((num_frames_remaining, classes),name = "top.reshape")(x)
            
            # logits (raw scores for each class)
            x = self.Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[2]))(x)
            x = self.activation('softmax', name='top.prediction')(x)
        else : 
            h = int(x.shape[2])
            w = int(x.shape[3])
            x = self.Avg3DPool((2, h, w), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)

        model = tf.keras.Model(self.Input,x,name=name)
        return model
    

# modelo = I3D().model()

# modelo.summary()
        
    
    
    
    

        