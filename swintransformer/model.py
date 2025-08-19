import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D, LayerNormalization, GlobalAveragePooling1D

# -------------- Helpers --------------

def _safe_name(*parts):
    """Join parts into a Keras-safe name (no slashes)."""
    return "__".join([p for p in parts if p])

CFGS = {
    'swin_tiny_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]),
    'swin_small_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
    'swin_base_224': dict(input_size=(224, 224), window_size=7, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_base_384': dict(input_size=(384, 384), window_size=12, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_large_224': dict(input_size=(224, 224), window_size=7, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48]),
    'swin_large_384': dict(input_size=(384, 384), window_size=12, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48])
}

class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., prefix=''):
        super().__init__(name=_safe_name(prefix, 'mlp'))
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Dense(hidden_features, name=_safe_name(prefix, 'mlp', 'fc1'))
        self.fc2 = Dense(out_features,  name=_safe_name(prefix, 'mlp', 'fc2'))
        self.drop = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = tf.keras.activations.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    # x: [B, H, W, C]
    shape = tf.shape(x)
    B, H, W, C = shape[0], shape[1], shape[2], shape[3]
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, [-1, window_size, window_size, C])
    return windows

def window_reverse(windows, window_size, H, W, C):
    # windows: [B*nW, window, window, C]
    BnW = tf.shape(windows)[0]
    # Infer B from BnW = B * (H/window)*(W/window)
    B = BnW // ((H // window_size) * (W // window_size))
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H, W, C])
    return x

class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., prefix=''):
        super().__init__(name=_safe_name(prefix, 'attn'))
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.prefix = prefix

        self.qkv = Dense(dim * 3, use_bias=qkv_bias, name=_safe_name(prefix, 'attn', 'qkv'))
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name=_safe_name(prefix, 'attn', 'proj'))
        self.proj_drop = Dropout(proj_drop)

        self.relative_position_bias_table = None
        self.relative_position_index = None

    def build(self, input_shape):
        ws0, ws1 = self.window_size
        table_shape = ((2 * ws0 - 1) * (2 * ws1 - 1), self.num_heads)
        self.relative_position_bias_table = self.add_weight(
            name=_safe_name(self.prefix, 'attn', 'relative_position_bias_table'),
            shape=table_shape,
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

        coords_h = np.arange(ws0)
        coords_w = np.arange(ws1)
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += ws0 - 1
        relative_coords[:, :, 1] += ws1 - 1
        relative_coords[:, :, 0] *= 2 * ws1 - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int64)

        self.relative_position_index = self.add_weight(
            name=_safe_name(self.prefix, 'attn', 'relative_position_index'),
            shape=relative_position_index.shape,
            initializer=tf.keras.initializers.Constant(relative_position_index),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x, mask=None):
        # x: [B*nW, N, C]
        shape = x.shape
        B_, N, C = shape[0], shape[1], shape[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [-1, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2]))

        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, [-1]))
        relative_position_bias = tf.reshape(relative_position_bias, [self.window_size[0] * self.window_size[1],
                                                                     self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = tf.shape(mask)[0]
            attn = tf.reshape(attn, [-1, nW, self.num_heads, N, N]) + tf.cast(tf.expand_dims(tf.expand_dims(mask, 1), 0), attn.dtype)
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.transpose(tf.matmul(attn, v), [0, 2, 1, 3])
        x = tf.reshape(x, [-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def drop_path(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs
    keep_prob = 1.0 - drop_prob
    shape = (tf.shape(inputs)[0],) + (1,) * (len(inputs.shape) - 1)
    random_tensor = keep_prob + tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output

class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__(name=_safe_name('drop_path'))
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)

class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., norm_layer=LayerNormalization, prefix=''):
        super().__init__(name=_safe_name(prefix, 'swin_block'))
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
        self.prefix = prefix

        self.norm1 = norm_layer(epsilon=1e-5, name=_safe_name(prefix, 'norm1'))
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, prefix=_safe_name(prefix))
        self.drop_path = DropPath(drop_path_prob if drop_path_prob > 0. else 0.)
        self.norm2 = norm_layer(epsilon=1e-5, name=_safe_name(prefix, 'norm2'))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, prefix=_safe_name(prefix))

        self.attn_mask = None

    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1], dtype=np.float32)
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = self.add_weight(
                name=_safe_name(self.prefix, 'attn_mask'),
                shape=attn_mask.shape,
                initializer=tf.keras.initializers.Constant(attn_mask.numpy()),
                trainable=False,
            )
        else:
            self.attn_mask = None
        super().build(input_shape)

    def call(self, x):
        H, W = self.input_resolution
        # x: [B, L, C] where L = H*W
        shortcut = x
        x = self.norm1(x)
        B = tf.shape(x)[0]
        C = x.shape[-1]
        x = tf.reshape(x, [B, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [B*nW, window, window, C]
        x_windows = tf.reshape(x_windows, [-1, self.window_size * self.window_size, C])

        # attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
        x = tf.reshape(x, [B, H * W, C])

        # residual + MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization, prefix=''):
        super().__init__(name=_safe_name(prefix, 'downsample'))
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Dense(2 * dim, use_bias=False, name=_safe_name(prefix, 'downsample', 'reduction'))
        self.norm = norm_layer(epsilon=1e-5, name=_safe_name(prefix, 'downsample', 'norm'))

    def call(self, x):
        H, W = self.input_resolution
        B = tf.shape(x)[0]
        C = x.shape[-1]
        x = tf.reshape(x, [B, H, W, C])

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, [B, (H // 2) * (W // 2), 4 * C])

        x = self.norm(x)
        x = self.reduction(x)
        return x

class BasicLayer(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_prob=0., norm_layer=LayerNormalization, downsample=None, use_checkpoint=False, prefix=''):
        super().__init__(name=_safe_name(prefix, 'basic_layer'))
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        blocks = []
        for i in range(depth):
            blocks.append(
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
                    prefix=_safe_name(prefix, f'block{i}')
                )
            )
        self.blocks = tf.keras.Sequential(blocks, name=_safe_name(prefix, 'blocks_seq'))
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, prefix=_safe_name(prefix))
        else:
            self.downsample = None

    def call(self, x):
        x = self.blocks(x)
        if self.downsample is not None:
            x = self.downsample(x)
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

    def call(self, x):
        # Validate spatial size against configured img_size
        input_shape = x.shape
        H, W = input_shape[1], input_shape[2]
        assert (H == self.img_size[0]) and (W == self.img_size[1]), \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)  # [B, H/ps_h, W/ps_w, embed_dim]
        B = tf.shape(x)[0]
        Hp, Wp = tf.shape(x)[1], tf.shape(x)[2]
        x = tf.reshape(x, [B, Hp * Wp, self.embed_dim])
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
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

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
                initializer=tf.keras.initializers.Zeros(),
                trainable=True
            )

        self.pos_drop = Dropout(drop_rate)

        # stochastic depth schedule
        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

        # build basic layers
        layers_seq = []
        for i_layer in range(self.num_layers):
            layers_seq.append(
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
                    prefix=_safe_name(f'layers{i_layer}')
                )
            )
        self.basic_layers = tf.keras.Sequential(layers_seq, name='basic_layers')

        self.norm = norm_layer(epsilon=1e-5, name='norm')
        self.avgpool = GlobalAveragePooling1D()
        self.head = Dense(num_classes, name='head') if self.include_top else None

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x = self.basic_layers(x)
        x = self.norm(x)
        x = self.avgpool(x)
        return x

    def call(self, x):
        x = self.forward_features(x)
        if self.include_top:
            x = self.head(x)
        return x

def SwinTransformer(model_name='swin_tiny_224', num_classes=1000, include_top=True, pretrained=True, use_tpu=False, cfgs=CFGS):
    cfg = cfgs[model_name]
    net = SwinTransformerModel(
        model_name=model_name,
        include_top=include_top,
        num_classes=num_classes,
        img_size=cfg['input_size'],
        window_size=cfg['window_size'],
        embed_dim=cfg['embed_dim'],
        depths=cfg['depths'],
        num_heads=cfg['num_heads']
    )
    # Build the model by calling it once
    net(tf.keras.Input(shape=(cfg['input_size'][0], cfg['input_size'][1], 3)))

    if pretrained is True:
        url = f'https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/{model_name}.tgz'
        pretrained_ckpt = tf.keras.utils.get_file(model_name, url, untar=True)
    else:
        pretrained_ckpt = pretrained

    if pretrained_ckpt:
        if tf.io.gfile.isdir(pretrained_ckpt):
            pretrained_ckpt = f'{pretrained_ckpt}/{model_name}.ckpt'
        if use_tpu:
            load_locally = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
            net.load_weights(pretrained_ckpt, options=load_locally)
        else:
            net.load_weights(pretrained_ckpt)
    return net
