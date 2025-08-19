# model.py — TensorFlow 2.18 / Keras 3 compatible

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D, LayerNormalization, GlobalAveragePooling1D

def _sanitize(name: str) -> str:
    return name.replace("/", "_") if name else name

CFGS = {
    'swin_tiny_224':  dict(input_size=(224, 224), window_size=7,  embed_dim=96,  depths=[2, 2, 6, 2],  num_heads=[3, 6, 12, 24]),
    'swin_small_224': dict(input_size=(224, 224), window_size=7,  embed_dim=96,  depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
    'swin_base_224':  dict(input_size=(224, 224), window_size=7,  embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_base_384':  dict(input_size=(384, 384), window_size=12, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_large_224': dict(input_size=(224, 224), window_size=7,  embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48]),
    'swin_large_384': dict(input_size=(384, 384), window_size=12, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48])
}

class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., prefix=''):
        super().__init__()
        self.prefix = _sanitize(prefix)
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = Dense(hidden_features, name=f'{self.prefix}_mlp_fc1')
        self.fc2  = Dense(out_features,  name=f'{self.prefix}_mlp_fc2')
        self.drop = Dropout(drop)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = tf.keras.activations.gelu(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.get_shape().as_list()
    x = tf.reshape(x, shape=[-1, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, window_size, C])
    return windows

def window_reverse(windows, window_size, H, W, C):
    x = tf.reshape(windows, shape=[-1, H // window_size, W // window_size, window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, H, W, C])
    return x

class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., prefix=''):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = qk_scale or head_dim ** -0.5
        self.prefix      = _sanitize(prefix)

        self.qkv       = Dense(dim * 3, use_bias=qkv_bias, name=f'{self.prefix}_attn_qkv')
        self.attn_drop = Dropout(attn_drop)
        self.proj      = Dense(dim, name=f'{self.prefix}_attn_proj')
        self.proj_drop = Dropout(proj_drop)

        # buffers (non-trainable) — define here, init in build()
        self.relative_position_bias_table = None
        self.relative_position_index_var  = None  # ResourceVariable to avoid out-of-scope

    def build(self, input_shape):
        # bias table (trainable)
        self.relative_position_bias_table = self.add_weight(
            name=f'{self.prefix}_attn_relative_position_bias_table',
            shape=((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True
        )

        # precompute relative position index (numpy -> constant initializer -> ResourceVariable)
        Wh, Ww = self.window_size
        coords_h = np.arange(Wh)
        coords_w = np.arange(Ww)
        coords   = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten  = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= (2 * Ww - 1)
        relative_position_index = relative_coords.sum(-1).astype('int32')  # [N, N]

        self.relative_position_index_var = self.add_weight(
            name=f'{self.prefix}_attn_relative_position_index',
            shape=relative_position_index.shape,
            dtype=tf.int32,
            initializer=tf.constant_initializer(relative_position_index),
            trainable=False,
        )

        super().build(input_shape)

    def call(self, x, mask=None, training=None):
        B_, N, C = x.get_shape().as_list()
        qkv = tf.transpose(
            tf.reshape(self.qkv(x), shape=[-1, N, 3, self.num_heads, C // self.num_heads]),
            perm=[2, 0, 3, 1, 4]
        )
        q, k, v = tf.unstack(qkv, num=3, axis=0)

        q = q * self.scale
        attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))

        Wh, Ww = self.window_size
        rel_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(self.relative_position_index_var, [-1])  # int32 indices OK
        )
        rel_bias = tf.reshape(rel_bias, shape=[Wh * Ww, Wh * Ww, -1])  # [N, N, num_heads]
        rel_bias = tf.transpose(rel_bias, perm=[2, 0, 1])              # [num_heads, N, N]
        attn = attn + tf.expand_dims(rel_bias, axis=0)

        if mask is not None:
            nW   = tf.shape(mask)[0]
            attn = tf.reshape(attn, shape=[-1, nW, self.num_heads, N, N]) + tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), attn.dtype
            )
            attn = tf.reshape(attn, shape=[-1, self.num_heads, N, N])

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = tf.transpose((attn @ v), perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x

def drop_path(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs
    keep_prob = 1.0 - drop_prob
    input_shape = tf.shape(inputs)
    rank = tf.rank(inputs)
    ones = tf.ones([rank - 1], dtype=tf.int32)
    shape = tf.concat([input_shape[:1], ones], axis=0)
    random_tensor = keep_prob + tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output

class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, bool(training))

class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0.,
                 norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.prefix = _sanitize(prefix)

        self.norm1 = norm_layer(epsilon=1e-5, name=f'{self.prefix}_norm1')
        self.attn  = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                                     prefix=self.prefix)
        self.drop_path = DropPath(drop_path_prob if drop_path_prob > 0. else 0.)
        self.norm2 = norm_layer(epsilon=1e-5, name=f'{self.prefix}_norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp  = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, prefix=self.prefix)

        # store mask as non-trainable weight to avoid graph-scope issues
        self.attn_mask_var = None

    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = self.input_resolution
            Wh = Ww = self.window_size
            img_mask_np = np.zeros([1, H, W, 1], dtype=np.float32)
            h_slices = (slice(0, -Wh), slice(-Wh, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -Ww), slice(-Ww, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask_np[:, h, w, :] = cnt
                    cnt += 1

            img_mask = tf.convert_to_tensor(img_mask_np, dtype=tf.float32)
            mask_windows = window_partition(img_mask, self.window_size)  # [nW*B, Wh, Ww, 1]
            mask_windows = tf.reshape(mask_windows, shape=[-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0,   0.0,   attn_mask)

            nW = (H // self.window_size) * (W // self.window_size)
            # create as ResourceVariable via add_weight (non-trainable)
            self.attn_mask_var = self.add_weight(
                name=f'{self.prefix}_attn_mask',
                shape=(nW, self.window_size * self.window_size, self.window_size * self.window_size),
                dtype=tf.float32,
                initializer=tf.constant_initializer(attn_mask.numpy()),
                trainable=False,
            )
        super().build(input_shape)

    def call(self, x, training=None):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=[-1, H, W, C])

        shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]) if self.shift_size > 0 else x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, shape=[-1, self.window_size * self.window_size, C])

        attn_windows = self.attn(x_windows, mask=self.attn_mask_var, training=training)

        attn_windows = tf.reshape(attn_windows, shape=[-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]) if self.shift_size > 0 else shifted_x
        x = tf.reshape(x, shape=[-1, H * W, C])

        x = shortcut + self.drop_path(x, training=training)
        x = x + self.drop_path(self.mlp(self.norm2(x), training=training), training=training)
        return x

class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.prefix = _sanitize(prefix)
        self.reduction = Dense(2 * dim, use_bias=False, name=f'{self.prefix}_downsample_reduction')
        self.norm = norm_layer(epsilon=1e-5, name=f'{self.prefix}_downsample_norm')

    def call(self, x, training=None):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = tf.reshape(x, shape=[-1, H, W, C])

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x  = tf.concat([x0, x1, x2, x3], axis=-1)
        x  = tf.reshape(x, shape=[-1, (H // 2) * (W // 2), 4 * C])

        x  = self.norm(x)
        x  = self.reduction(x)
        return x

class BasicLayer(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_prob=0., norm_layer=LayerNormalization, downsample=None, use_checkpoint=False, prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.prefix = _sanitize(prefix)

        self.blocks = tf.keras.Sequential([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path_prob=drop_path_prob[i] if isinstance(drop_path_prob, list) else drop_path_prob,
                norm_layer=norm_layer,
                prefix=f'{self.prefix}_blocks{i}'
            ) for i in range(depth)
        ])
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, prefix=self.prefix) if downsample is not None else None

    def call(self, x, training=None):
        x = self.blocks(x, training=training)
        if self.downsample is not None:
            x = self.downsample(x, training=training)
        return x

class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, img_size=(224, 224), patch_size=(4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__(name='patch_embed')
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, name='proj')
        self.norm = norm_layer(epsilon=1e-5, name='norm') if norm_layer is not None else None

    def call(self, x, training=None):
        B, H, W, C = x.get_shape().as_list()
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = tf.reshape(x, shape=[-1, (H // self.patch_size[0]) * (W // self.patch_size[0]), self.embed_dim])
        if self.norm is not None:
            x = self.norm(x)
        return x

class SwinTransformerModel(tf.keras.Model):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', include_top=False,
                 img_size=(224, 224), patch_size=(4, 4), in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=LayerNormalization, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__(name=model_name)
        self.include_top = include_top
        self.num_classes = num_classes
        self.num_layers  = len(depths)
        self.embed_dim   = embed_dim
        self.ape         = ape
        self.patch_norm  = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio   = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = self.add_weight(
                name='absolute_pos_embed',
                shape=(1, num_patches, embed_dim),
                initializer=tf.initializers.Zeros(),
                trainable=True
            )

        self.pos_drop = Dropout(drop_rate)

        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

        self.basic_layers = tf.keras.Sequential([
            BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path_prob=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                prefix=f'layers{i_layer}'
            ) for i_layer in range(self.num_layers)
        ])
        self.norm = norm_layer(epsilon=1e-5, name='norm')
        self.avgpool = GlobalAveragePooling1D()
        self.head = Dense(num_classes, name='head') if self.include_top else None

    def forward_features(self, x, training=None):
        x = self.patch_embed(x, training=training)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x, training=training)
        x = self.basic_layers(x, training=training)
        x = self.norm(x)
        x = self.avgpool(x)
        return x

    def call(self, x, training=None):
        x = self.forward_features(x, training=training)
        if self.include_top:
            x = self.head(x)
        return x

def SwinTransformer(model_name='swin_tiny_224', num_classes=1000, include_top=True, pretrained=True, use_tpu=False, cfgs=CFGS):
    cfg = cfgs[model_name]
    net = SwinTransformerModel(
        model_name=model_name, include_top=include_top, num_classes=num_classes,
        img_size=cfg['input_size'], window_size=cfg['window_size'],
        embed_dim=cfg['embed_dim'], depths=cfg['depths'], num_heads=cfg['num_heads']
    )

    # Build graph
    _ = net(tf.keras.Input(shape=(cfg['input_size'][0], cfg['input_size'][1], 3)))

    if pretrained is True:
        url = f'https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/{model_name}.tgz'
        pretrained_ckpt = tf.keras.utils.get_file(model_name, url, untar=True)
    else:
        pretrained_ckpt = pretrained

    if pretrained_ckpt:
        if tf.io.gfile.isdir(pretrained_ckpt):
            pretrained_ckpt = f'{pretrained_ckpt}/{model_name}.ckpt'

        # Keras 3 no longer supports direct `.ckpt` in load_weights. Fallback to tf.train.Checkpoint.
        try:
            net.load_weights(pretrained_ckpt)
        except ValueError:
            ckpt = tf.train.Checkpoint(model=net)
            # expect_partial(): ckpt vars may not perfectly match Keras names
            ckpt.restore(pretrained_ckpt).expect_partial()

    return net
