from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, UpSampling2D, Add, Multiply, BatchNormalization, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    LayerNormalization, Dense, Dropout, MultiHeadAttention, Flatten,
    Conv2D, Input, MaxPooling2D, UpSampling2D, concatenate, Reshape
)
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout, MultiHeadAttention, Flatten
from tensorflow.keras.layers import LayerNormalization, Dropout, Dense
from einops import rearrange
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Lambda, Add, MultiHeadAttention, LayerNormalization, Dense, Dropout, Conv2D, Layer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

def build_unet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(c5)

    # Decoder
    u6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(c9)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model



def attention_gate(x, g, inter_channels=None):
    if inter_channels is None:
        inter_channels = x.shape[-1]  # default to encoder skip channels

    theta_x = Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same')(x)
    phi_g = Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same')(g)

    add = Add()([theta_x, phi_g])
    relu = Activation('relu')(add)
    psi = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(relu)
    sigmoid = Activation('sigmoid')(psi)

    att = Multiply()([x, sigmoid])
    return att

def build_attention_unet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(c5)

    # Decoder with Attention
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    a4 = attention_gate(c4, u6, 256)
    u6 = concatenate([u6, a4])
    c6 = Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    a3 = attention_gate(c3, u7, 128)
    u7 = concatenate([u7, a3])
    c7 = Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    a2 = attention_gate(c2, u8, 64)
    u8 = concatenate([u8, a2])
    c8 = Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    a1 = attention_gate(c1, u9, 32)
    u9 = concatenate([u9, a1])
    c9 = Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(c9)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)

    return Model(inputs=[inputs], outputs=[outputs])



def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Adjust shortcut channels to match x if needed
    shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)

    x = Add()([shortcut, x])
    return x

def build_raunet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, 3, padding='same', activation='relu')(inputs)
    c1 = residual_block(c1, 64)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, 3, padding='same', activation='relu')(p1)
    c2 = residual_block(c2, 128)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, 3, padding='same', activation='relu')(p2)
    c3 = residual_block(c3, 256)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, 3, padding='same', activation='relu')(p3)
    c4 = residual_block(c4, 512)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, 3, padding='same', activation='relu')(p4)
    c5 = residual_block(c5, 1024)

    # Decoder with Residual + Attention
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    a4 = attention_gate(c4, u6, 256)
    u6 = concatenate([u6, a4])
    u6 = Conv2D(512, (1, 1), activation='relu', padding='same')(u6)
    c6 = residual_block(u6, 512)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    a3 = attention_gate(c3, u7, 128)
    u7 = concatenate([u7, a3])
    c7 = residual_block(u7, 256)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    a2 = attention_gate(c2, u8, 64)
    u8 = concatenate([u8, a2])
    c8 = residual_block(u8, 128)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    a1 = attention_gate(c1, u9, 32)
    u9 = concatenate([u9, a1])
    c9 = residual_block(u9, 64)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)

    return Model(inputs=[inputs], outputs=[outputs])




def transformer_encoder(inputs, num_heads=4, ff_dim=512, dropout=0.1):
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention = Dropout(dropout)(attention)
    x = LayerNormalization(epsilon=1e-6)(inputs + attention)

    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dense(inputs.shape[-1])(ff)
    ff = Dropout(dropout)(ff)
    x = LayerNormalization(epsilon=1e-6)(x + ff)
    return x

def patch_embedding(x, patch_size):
    patches = Conv2D(256, kernel_size=patch_size, strides=patch_size, padding='valid')(x)
    h, w, c = patches.shape[1], patches.shape[2], patches.shape[3]
    x = Reshape((h * w, c))(patches)
    return x, h, w
def build_transunet(input_shape=(256, 256, 1), patch_size=4, num_layers=4, ff_dim=512):
    inputs = Input(shape=input_shape)

    # CNN Encoder
    x = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    skip1 = x  # (256, 256, 64)
    x = MaxPooling2D()(x)  # (128, 128, 64)

    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    skip2 = x  # (128, 128, 128)
    x = MaxPooling2D()(x)  # (64, 64, 128)

    # Patch Embedding
    x = Conv2D(256, kernel_size=patch_size, strides=patch_size, padding='valid')(x)  # (16, 16, 256)
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x = Reshape((h * w, c))(x)  # (batch, 256, 256)

    # Transformer Encoder
    for _ in range(num_layers):
        x = transformer_encoder(x, num_heads=4, ff_dim=ff_dim)

    x = Reshape((h, w, 256))(x)  # (16, 16, 256)

    # Decoder
    x = UpSampling2D(size=(2, 2))(x)  # (32, 32)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)

    x = UpSampling2D(size=(2, 2))(x)  # (64, 64)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)

    # skip2 is (128, 128), upsample x to match
    x = UpSampling2D(size=(2, 2))(x)  # (128, 128)
    skip2 = Conv2D(128, 1, activation='relu', padding='same')(skip2)
    x = concatenate([x, skip2])
    x = Conv2D(128, 3, activation='relu', padding='same')(x)

    x = UpSampling2D(size=(2, 2))(x)  # (256, 256)
    skip1 = Conv2D(64, 1, activation='relu', padding='same')(skip1)
    x = concatenate([x, skip1])
    x = Conv2D(64, 3, activation='relu', padding='same')(x)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    return Model(inputs=inputs, outputs=outputs)


import tensorflow as tf

def window_partition_tf(x, window_size):
    
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, [-1, window_size, window_size, C])
    return windows



def window_reverse_tf(windows, window_size, H, W, C):
    B = tf.shape(windows)[0] // ((H // window_size) * (W // window_size))
    x = tf.reshape(windows, (B, H // window_size, W // window_size, window_size, window_size, C))
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, (B, H, W, C))
    return x


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation='gelu')(x)
        x = Dropout(dropout_rate)(x)
    return x

class RollLayer(Layer):
    def __init__(self, shift, axis, **kwargs):
        super(RollLayer, self).__init__(**kwargs)
        self.shift = shift
        self.axis = axis

    def call(self, inputs):
        return tf.roll(inputs, shift=self.shift, axis=self.axis)

def swin_block(x, window_size=8, shift_size=0, num_heads=4, embed_dim=128, mlp_dim=256):
    input_channels = x.shape[-1]
    H, W = x.shape[1], x.shape[2]

    shortcut = x
    x = LayerNormalization(epsilon=1e-6)(x)

    if shift_size > 0:
        # Use the custom RollLayer
        x = RollLayer(shift=[-shift_size, -shift_size], axis=[1, 2])(x)

    x_windows = Lambda(
        lambda t: window_partition_tf(t, window_size),
        output_shape=(window_size * window_size, input_channels)
    )(x)

    attn_windows = MultiHeadAttention(num_heads=num_heads, key_dim=input_channels)(x_windows, x_windows)

    x = Lambda(
        lambda t: window_reverse_tf(t, window_size, H, W, input_channels),
        output_shape=(H, W, input_channels)
    )(attn_windows)

    if shift_size > 0:
        # Use the custom RollLayer
        x = RollLayer(shift=[shift_size, shift_size], axis=[1, 2])(x)

    # Project shortcut if needed
    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = Conv2D(filters=int(x.shape[-1]), kernel_size=1, padding='same')(shortcut)

    # Project mlp output to match input channels
    mlp_output = mlp(LayerNormalization(epsilon=1e-6)(x), [mlp_dim], 0.1)
    mlp_output = Conv2D(filters=int(x.shape[-1]), kernel_size=1, padding='same')(mlp_output)  # Adjust channels

    x = Add()([shortcut, x])
    x = Add()([x, mlp_output])  # Use the projected mlp_output

    return x

from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, concatenate, Conv2D

def build_swin_unet(input_shape=(256, 256, 1), num_blocks=2):
    inputs = Input(input_shape)

    # Encoder
    x = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    skip1 = x
    x = MaxPooling2D()(x)

    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    skip2 = x
    x = MaxPooling2D()(x)

    # Swin Transformer Blocks
    for i in range(num_blocks):
        x = swin_block(x, window_size=8, shift_size=(i % 2) * 4, embed_dim=128)

    # Decoder
    x = UpSampling2D()(x)
    skip2 = Conv2D(128, 1, activation='relu', padding='same')(skip2)
    x = concatenate([x, skip2])
    x = Conv2D(128, 3, activation='relu', padding='same')(x)

    x = UpSampling2D()(x)
    skip1 = Conv2D(64, 1, activation='relu', padding='same')(skip1)
    x = concatenate([x, skip1])
    x = Conv2D(64, 3, activation='relu', padding='same')(x)

    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    return Model(inputs, outputs)


def build_metaensemble(input_shape=(5,)):
    meta_model = Sequential([
        Input(shape=input_shape),
        Dense(8, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return meta_model 