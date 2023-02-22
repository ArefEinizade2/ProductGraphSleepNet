import sys
import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras import regularizers
from keras import models
from keras.layers import Layer, Bidirectional, LSTM, Reshape, GRU, Dropout, BatchNormalization
from keras.layers.core import Lambda
from graph_attention_layer import GraphWiseAttentionNetwork
#%%
# Model input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
# 
#     V: num_of_vertices
#     T: num_of_timesteps
#     F: num_of_features
#
# Model output: (*, 5)
# 
#     5: 5 sleep stages

#%%
class SpatialAttention(Layer):
    '''
    compute spatial attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.W_1 = self.add_weight(name='W_1',
                                      shape=(num_of_timesteps, 1),
                                      initializer='uniform',
                                      trainable=True)
        self.W_2 = self.add_weight(name='W_2',
                                      shape=(num_of_features, num_of_timesteps),
                                      initializer='uniform',
                                      trainable=True)
        self.W_3 = self.add_weight(name='W_3',
                                      shape=(num_of_features, ),
                                      initializer='uniform',
                                      trainable=True)
        self.b_s = self.add_weight(name='b_s',
                                      shape=(1, num_of_vertices, num_of_vertices),
                                      initializer='uniform',
                                      trainable=True)
        self.V_s = self.add_weight(name='V_s',
                                      shape=(num_of_vertices, num_of_vertices),
                                      initializer='uniform',
                                      trainable=True)
        super(SpatialAttention, self).build(input_shape)

    def call(self, x):
        _, num_of_timesteps, num_of_vertices, num_of_features = x.shape
        
        # shape of lhs is (batch_size, V, T)
        lhs=K.dot(tf.transpose(x,perm=[0,2,3,1]), self.W_1)
        lhs=tf.reshape(lhs,[tf.shape(x)[0],num_of_vertices,num_of_features])
        lhs = K.dot(lhs, self.W_2)
        
        # shape of rhs is (batch_size, T, V)
        rhs = K.dot(self.W_3, tf.transpose(x,perm=[1,0,3,2])) # K.dot((F),(T,batch_size,F,V))=(T,batch_size,V)
        rhs=tf.transpose(rhs,perm=[1,0,2]) # (batch_size, T, V)
        
        # shape of product is (batch_size, V, V)
        product = K.batch_dot(lhs, rhs)
        
        S = tf.transpose(K.dot(self.V_s, tf.transpose(K.sigmoid(product + self.b_s),perm=[1, 2, 0])),perm=[2, 0, 1])
        
        # normalization
        S = S - K.max(S, axis = 1, keepdims = True)
        exp = K.exp(S)
        S_normalized = exp / K.sum(exp, axis = 1, keepdims = True)
        return S_normalized

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2],input_shape[2])

#%%
def diff_loss(diff, S):
    '''
    compute the differential (smoothness term) loss for Spatial/Temporal graph learning
    '''
    diff_loss_value = 0
    F = diff.shape[1]
    for i in range(int(F)):
        diff_loss_value = diff_loss_value + K.sum(K.sum(diff[:,i]**2,axis=3)*S)
    
    return diff_loss_value

#%%
def F_norm_loss(S, Falpha):
    '''
    compute the Frobenious norm loss cheb_polynomials
    '''
    if len(S.shape)==3:
        # batch input
        return Falpha * K.sum(K.mean(S**2,axis=0))
    else:
        return Falpha * K.sum(S**2)

#%%
class Graph_Learn_Spatial(Layer):
    '''
    Spatial graph structure learning
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        self.S_Spatial = tf.convert_to_tensor([[[0.0]]])  # similar to placeholder
        self.diff_Spatial = tf.convert_to_tensor([[[[[0.0]]]]])  # similar to placeholder
        super(Graph_Learn_Spatial, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.a_Spatial = self.add_weight(name='a_Spatial',
                                 shape=(num_of_features, 1),
                                 initializer='uniform',
                                 trainable=True)
        super(Graph_Learn_Spatial, self).build(input_shape)

    def call(self, x):
        #Input:  [N, timesteps, vertices, features]
        _, T, V, F = x.shape
        N = tf.shape(x)[0]
                        
        # Spatial Graph Learning:
        for ff in range(int(F)):
            x_Spatial_ff = tf.transpose(x[:, :, :, ff], perm=[0, 2, 1]) #(N, V, T)

            diff_Spatial_temp = K.abs(tf.transpose(tf.transpose(tf.broadcast_to(x_Spatial_ff, [V,N,V,T]), perm=[2,1,0,3])
            - x_Spatial_ff, perm=[1,0,2,3])) #(N, V, V, T)
            
            diff_Spatial_temp = K.expand_dims(diff_Spatial_temp, axis=1) #(N, 1, V, V, T)

            if ff == 0:
                diff_Spatial = diff_Spatial_temp
            else:
                diff_Spatial = K.concatenate((diff_Spatial, diff_Spatial_temp), axis=1) #(N, F, V, V, T)
                
        tmpS = K.exp(K.relu(K.reshape(K.dot(tf.reduce_mean(tf.transpose(diff_Spatial, perm=[0, 4, 2, 3, 1]), axis=1), self.a_Spatial), [N,V,V]))) #(N, V, V)
        
        # normalization
        S_Spatial = tmpS / K.sum(tmpS, axis=1, keepdims=True)
        
        self.diff_Spatial = diff_Spatial
        self.S_Spatial = S_Spatial
        
        # add spatial graph learning loss in the layer
        self.add_loss(F_norm_loss(self.S_Spatial,self.alpha))
        self.add_loss(diff_loss(self.diff_Spatial,self.S_Spatial))

        return S_Spatial

    def compute_output_shape(self, input_shape):
        # shape: (N, V, V)
        return (input_shape[0], input_shape[2], input_shape[2])
#%%
class Graph_Learn_Temporal(Layer):
    '''
    Temporal graph structure learning 
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_timesteps, num_of_timesteps)
    '''
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        self.S_Temporal = tf.convert_to_tensor([[[0.0]]])  # similar to placeholder
        self.diff_Temporal = tf.convert_to_tensor([[[[0.0]]]])  # similar to placeholder
        super(Graph_Learn_Temporal, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.a_Temporal = self.add_weight(name='a_Temporal',
                                 shape=(num_of_features, 1),
                                 initializer='uniform',
                                 trainable=True)
        super(Graph_Learn_Temporal, self).build(input_shape)

    def call(self, x):
        #Input:  [N, timesteps, vertices, features]
        _, T, V, F = x.shape
        N = tf.shape(x)[0]
        
        # Temporal Graph Learning:
        for ff in range(int(F)):
            x_Temporal_ff = x[:, :, :, ff] #(N, T, V)

            diff_Temporal_temp = K.abs(tf.transpose(tf.transpose(tf.broadcast_to(x_Temporal_ff, [T,N,T,V]), perm=[2,1,0,3])
            - x_Temporal_ff, perm=[1,0,2,3])) #(N, T, T, V)
            
            diff_Temporal_temp = K.expand_dims(diff_Temporal_temp, axis=1) #(N, 1, T, T, V)

            if ff == 0:
                diff_Temporal = diff_Temporal_temp
            else:
                diff_Temporal = K.concatenate((diff_Temporal, diff_Temporal_temp), axis=1) #(N, F, T, T, V)
                
        tmpS = K.exp(K.relu(K.reshape(K.dot(tf.reduce_mean(tf.transpose(diff_Temporal, perm=[0, 4, 2, 3, 1]), axis=1), self.a_Temporal), [N,T,T]))) #(N, T, T)
        
        # normalization
        S_Temporal = tmpS / K.sum(tmpS, axis=1, keepdims=True)
        
        self.diff_Temporal = diff_Temporal
        self.S_Temporal = S_Temporal
        
        # add temporal graph learning loss in the layer
        self.add_loss(F_norm_loss(self.S_Temporal,self.alpha))
        self.add_loss(diff_loss(self.diff_Temporal, self.S_Temporal))

        return S_Temporal

    def compute_output_shape(self, input_shape):
        # shape: (N, T, T)
        return  ((input_shape[0], input_shape[1], input_shape[1]))
#%%
class cheb_conv_with_SAt_GL(Layer):
    '''
    K-order chebyshev graph convolution after Graph Learn
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             SAtt(batch_size, num_of_vertices, num_of_vertices),
             S   (batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    '''
    def __init__(self, num_of_filters, k, **kwargs):
        self.k = k
        self.num_of_filters = num_of_filters
        super(cheb_conv_with_SAt_GL, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        x_shape,SAtt_shape,S_shape=input_shape
        _, num_of_timesteps, num_of_vertices, num_of_features = x_shape
        self.Theta = self.add_weight(name='Theta',
                                     shape=(self.k, num_of_features, self.num_of_filters),
                                     initializer='uniform',
                                     trainable=True)
        super(cheb_conv_with_SAt_GL, self).build(input_shape)

    def call(self, x):
        #Input:  [x,SAtt,S]
        assert isinstance(x, list)
        assert len(x)==3,'cheb_conv_with_SAt_GL: number of input error'
        x, spatial_attention, W = x
        _, num_of_timesteps, num_of_vertices, num_of_features = x.shape
        #Calculating Chebyshev polynomials
        D = tf.matrix_diag(K.sum(W,axis=1))
        L = D - W
        '''
        Here we approximate λ_{max} to 2 to simplify the calculation.
        For more general calculations, please refer to here:
            lambda_max = K.max(tf.self_adjoint_eigvals(L),axis=1)
            L_t = (2 * L) / tf.reshape(lambda_max,[-1,1,1]) - [tf.eye(int(num_of_vertices))]
        '''
        lambda_max = 2.0
        L_t = (2 * L) / lambda_max - [tf.eye(int(num_of_vertices))]
        cheb_polynomials = [tf.eye(int(num_of_vertices)), L_t]
        for i in range(2, self.k):
            cheb_polynomials.append(2 * L_t * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
        
        #Graph Convolution
        outputs=[]
        for time_step in range(num_of_timesteps):
            # shape of x is (batch_size, V, F)
            graph_signal = x[:, time_step, :, :]
            # shape of x is (batch_size, V, F')
            output = K.zeros(shape = (tf.shape(x)[0], num_of_vertices, self.num_of_filters))
            
            for kk in range(self.k):
                # shape of T_k is (V, V)
                T_k = cheb_polynomials[kk]
                    
                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta[kk]

                # shape is (batch_size, V, F)
                rhs = K.batch_dot(tf.transpose(T_k_with_at,perm=[0, 2, 1]), graph_signal)

                output = output + K.dot(rhs, theta_k)
            outputs.append(tf.expand_dims(output,1))
            
        return K.relu(K.concatenate(outputs, axis = 1))

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        # shape: (n, num_of_timesteps, num_of_vertices, num_of_filters)
        return (input_shape[0][0],input_shape[0][1],input_shape[0][2],self.num_of_filters)

#%%
def ProductGraphSleepBlock(x, k, num_of_chev_filters, GLalpha):
    '''
    packaged Spatial-temporal convolution Block
    -------
    '''        
    # SpatialAttention
    # output shape is (batch_size, V, V)
    spatial_At = SpatialAttention()(x)
    
    # Graph Convolution with spatial attention
    # output shape is (batch_size, T, V, F)

    # use adaptive Graph Learn
    S_Spatial = Graph_Learn_Spatial(alpha=GLalpha)(x)
    spatial_gcn = cheb_conv_with_SAt_GL(num_of_filters=num_of_chev_filters, k=k)([x, spatial_At, S_Spatial])    
    S_Temporal = Graph_Learn_Temporal(alpha=GLalpha)(x)
        
    return spatial_gcn, S_Temporal

#%%
def build_ProductGraphSleepNet(k, num_of_chev_filters, 
                sample_shape, opt, GLalpha, regularizer, GRU_Cell, attn_heads, dropout):
    
    # Input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
    data_layer = layers.Input(shape=sample_shape, name='Input-Data')
    
    # ProductGraphSleepBlock
    block_out, S_Temporal = ProductGraphSleepBlock(data_layer,k, num_of_chev_filters,GLalpha)

    # BiLSTM
    block_out = Reshape((-1, int(block_out.shape[2]*block_out.shape[3])))(block_out)
    x_GRU = Bidirectional(GRU(GRU_Cell, dropout=dropout, recurrent_dropout=dropout, return_sequences=True))(block_out)    
            
    softmax = GraphWiseAttentionNetwork(5, attn_heads=attn_heads,
                                       attn_heads_reduction='average',
                                       dropout_rate=dropout,
                                       activation='softmax',
                                       kernel_regularizer=None,
                                       attn_kernel_regularizer=None)([x_GRU, S_Temporal])
    
    model = models.Model(inputs = data_layer, outputs = softmax)
    
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['acc'])
    
    return model
#%%

#def build_GraphSleepNet_test():
#    
#    # an example to test
#    cheb_k = 3
#    num_of_chev_filters = 10
#    num_of_time_filters = 10
#    time_conv_strides = 1
#    time_conv_kernel = 3
#    dense_size=np.array([64,32])
#    cheb_polynomials = [np.random.rand(21,21),np.random.rand(21,21),np.random.rand(21,21)]
#
#    opt='adam'
#
#    model=build_GraphSleepNet(cheb_k, num_of_chev_filters, num_of_time_filters,time_conv_strides, cheb_polynomials, time_conv_kernel, 
#                      sample_shape=(5,21,9),num_block=1, dense_size=dense_size, opt=opt, useGL=True, 
#                      GLalpha=0.0001, regularizer=None, dropout = 0.0)
#    model.summary()
#    model.save('GraphSleepNet_build_test.h5')
#    print("save ok")
#    return model
#
#
## build_GraphSleepNet_test()
#
