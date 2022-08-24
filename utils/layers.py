import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GRU, Concatenate, Lambda, Embedding, Dropout, Layer
from tensorflow.keras.layers import LayerNormalization, Add, MultiHeadAttention, Activation
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow_probability.python.layers import DistributionLambda
from tensorflow_probability.python.distributions import PowerSpherical, Distribution, SphericalUniform
from tensorflow.keras.optimizers import Adam

import numpy as np
from adabelief_tf import AdaBeliefOptimizer


# ----------------------------------------Helping classes---------------------------------------------------------------

class BatchWM(tf.keras.callbacks.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.step = 0

    def on_batch_begin(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.step += 1

        learning_rate = np.power(512, -0.5) * min(np.power(self.step, -0.5), self.step * np.power(8000, -1.5))
        K.set_value(self.model.optimizer.learning_rate, learning_rate)


class GLT(Layer):
    def __init__(self, N, wm, out_dim=None):
        super(GLT, self).__init__()
        self.N = N
        self.wm = wm
        self.out_dim = out_dim

    def build(self, input_shape):
        self.dim = input_shape[-1]

        if self.out_dim is None:
            self.out_dim = self.dim // 2

        d_max = self.wm * self.dim
        g_max = self.dim // 32

        grow = (np.linspace(self.dim, d_max, self.N // 2) + 0.5).astype(np.int32)
        reduction = (np.linspace(d_max, self.out_dim, self.N - self.N // 2) + 0.5).astype(np.int32)
        self.s = np.concatenate([grow, reduction])

        for i in range(len(self.s)):
            while (self.s[i] % g_max) != 0:
                self.s[i] += 1

        self.ff = []
        for l in range(1, self.N + 1):
            if l <= self.N // 2:
                g = min(2 ** (l - 1), g_max)
            else:
                g = min(2 ** (self.N - l + 1), g_max)

            while (self.s[l - 1] % g) != 0:
                self.s[l - 1] += 1

            gl = []
            for _ in range(g):
                gl.append(Dense(self.s[l - 1] // g))

            self.ff.append(gl)

        super(GLT, self).build(input_shape)

    def call(self, In, **kwargs):
        batch_size = K.shape(In)[0]
        len_size = K.int_shape(In)[1]

        G = self.ff[0][0](In)

        for l in range(1, len(self.ff)):
            x = Concatenate(axis=-1)([In, G])

            n = K.int_shape(x)[-1]

            x = K.reshape(x, [batch_size * len_size, n])

            x = K.reshape(x,
                          [batch_size * len_size, len(self.ff[l]), n // len(self.ff[l])])  # [B x N] --> [B x g x N/g]
            x = K.permute_dimensions(x, [1, 0, 2])  # [B x g x N/g] --> [g x B x N/g]

            g = []
            for i, f in enumerate(self.ff[l]):  # [g x B x N/g] x [g x N/g x M/g] --> [g x B x M/g]
                g.append(K.expand_dims(f(x[i]), axis=0))
            x = K.concatenate(g, axis=0)

            # shuffle
            x = K.permute_dimensions(x, [1, 2, 0])  # [g x B x M/g] --> [B x M/g x g]
            x = K.reshape(x, [batch_size * len_size, len(self.ff[l]), -1])  # [B x M/g x g] --> [B x g x M/g]

            G = K.reshape(x, [batch_size, len_size, self.s[l]])  # [B x g x M/g] --> [B, T, M]

        return G

    def get_config(self):
        config = super(GLT, self).get_config()
        config['N'] = self.N
        config['wm'] = self.wm
        config['out_dim'] = self.out_dim
        return config


class KL(Layer):
    def __init__(self, **kwargs):
        super(KL, self).__init__(**kwargs)
        # self.max_steps = max_steps
        # self.step = step

    def call(self, x, **kwargs):
        z_mean, z_log_var = x

        kl_loss = PowerSpherical(z_mean, K.flatten(z_log_var + 1)).kl_divergence(SphericalUniform(K.int_shape(z_mean)[-1]))

        return K.mean(kl_loss)  # * K.minimum(K.tanh(6 * self.step / self.max_steps - 3) + 1, 1.)


class LayerScale(Layer):
    def __init__(self):
        super(LayerScale, self).__init__()

    def build(self, input_shape):
        self.scaler = self.add_weight(shape=[input_shape[-1]],
                                      initializer=tf.keras.initializers.Constant(0.1),
                                      name='scaler',
                                      trainable=True)

    def call(self, x, **kwargs):
        return x * self.scaler


class SharedEmbeddingOutput(Layer):
    def __init__(self, w):
        self.w = w
        super(SharedEmbeddingOutput, self).__init__()

    def call(self, x, **kwargs):
        out = tf.linalg.matmul(x, self.w, transpose_b=True)
        return out

    def get_config(self):
        config = super(SharedEmbeddingOutput, self).get_config()
        config['w'] = self.w
        return config

# ----------------------------------------Helping functions-------------------------------------------------------------


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, :]


def FTSwishG(threshold=-0.2):
    def _FTSwishG(x):
        return K.relu(x) * K.sigmoid(x*1.702) + threshold
    return Lambda(_FTSwishG)


def get_centralized_gradients(optimizer, loss, params):
    """Compute a list of centralized gradients.

    Modified version of tf.keras.optimizers.Optimizer.get_gradients:
    https://github.com/keras-team/keras/blob/1931e2186843ad3ca2507a3b16cb09a7a3db5285/keras/optimizers.py#L88-L101
    Reference:
        https://arxiv.org/pdf/2004.01461.pdf
    """
    # The One Line: subtract means values of gradient matrix columns
    grads = []
    for grad in K.gradients(loss, params):
        rank = len(grad.shape)
        if rank > 1:
            grad -= tf.reduce_mean(grad, axis=list(range(rank - 1)), keep_dims=True)
        grads.append(grad)

    if None in grads:
        raise ValueError('An operation has `None` for gradient. '
                         'Please make sure that all of your ops have a '
                         'gradient defined (i.e. are differentiable). '
                         'Common ops without gradient: '
                         'K.argmax, K.round, K.eval.')
    if hasattr(optimizer, 'clipnorm') and optimizer.clipnorm > 0:
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        grads = [tf.keras.optimizers.clip_norm(g, optimizer.clipnorm, norm) for g in grads]
    if hasattr(optimizer, 'clipvalue') and optimizer.clipvalue > 0:
        grads = [K.clip(g, -optimizer.clipvalue, optimizer.clipvalue) for g in grads]
    return grads


def get_centralized_gradients_function(optimizer):
    """Produce a get_centralized_gradients function for a particular optimizer instance."""

    def get_centralized_gradients_for_instance(loss, params):
        return get_centralized_gradients(optimizer, loss, params)

    return get_centralized_gradients_for_instance


def get_embedding_layer(n_tokens: int, output_dim: int) -> Layer:
    """
    This function creates an Embedding layer that can be passed then to several places. This is used for weights
    sharing.
    :param n_tokens: Number of unique tokens.
    :param output_dim: Output dimension of the embedding vector.
    :return:
    """
    emb = Embedding(n_tokens+1, output_dim, embeddings_initializer='truncated_normal', name='embedding_layer',
                    mask_zero=True)
    return emb


def positional_encoding():
    def _pos_enc(x):
        _, max_len, d_emb = K.int_shape(x)

        pos_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0 else np.zeros(d_emb)
            for pos in range(max_len)])

        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        x += pos_enc
        return x

    return Lambda(_pos_enc)


def mask_acc(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    mask = tf.greater(y_true, 0.)

    y_pred = tf.boolean_mask(y_pred, mask)
    y_true = tf.boolean_mask(y_true, mask)

    return K.sum(tf.cast(tf.math.equal(y_true, tf.cast(y_pred, tf.float32)), tf.float32)) / tf.cast(K.shape(y_true)[0],
                                                                                                    tf.float32)


def rec_rate(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.float32)
    x = tf.cast(tf.equal(y_true, y_pred), tf.float32)
    mask = tf.cast(tf.greater(y_true, 0.), tf.float32)
    x = tf.multiply(x, mask)

    x = tf.reduce_sum(x, axis=1)
    sl = tf.reduce_sum(mask, axis=1)

    x = tf.cast(tf.equal(x, sl), tf.float32)

    return tf.reduce_mean(x)



# ----------------------------------------Model's parts-----------------------------------------------------------------


def features_layer(n_features: int, dim: int) -> Model:
    """
    This block is needed to pre-process input features that are given as a query vector. Pre-processing is done using a
    Group Linear Transformations (GLT) layer published in https://arxiv.org/pdf/2008.00623v2.pdf.
    :param n_features: Number of features.
    :param dim: Internal dimensionality.
    :return: Model
    """
    features = Input(shape=(1, n_features))
    x = Dense(dim, kernel_initializer='glorot_uniform')(features)
    x = GLT(N=4, wm=2, out_dim=dim)(x)
    x = Dropout(0.1)(x)

    return Model(inputs=features, outputs=x, name='features_layer')


def encoder(msl: int, latent_dim: int, dim: int, embedding_layer: Layer) -> Model:
    """
    This block takes the input SMILES given for the training, and transforms it into a latent representation defined by
    generated mean and log(variance) values, which describe a PowerSpherical distribution. Note that every sequence
    passed to this block should end by <EOS> symbol and be padded by <PAD> symbol.
    :parameter msl: Maximal SMILES length.
    :parameter latent_dim: Latent dimensionality.
    :parameter dim: Internal dimensionality of the model.
    :parameter embedding_layer: Embedding layer defined in advance.
    :return: Model
    """
    smiles_input = Input(shape=(msl,))
    x = embedding_layer(smiles_input)
    x = Lambda(lambda x: x * tf.math.sqrt(tf.cast(dim, tf.float32)))(x)
    x = positional_encoding()(x)
    x = Dropout(0.1)(x)
    x = GRU(dim)(x)
    z_mean = Dense(latent_dim, activation=lambda x: tf.nn.l2_normalize(x, axis=-1), name='z_mean')(x)
    z_log_var = Dense(1, activation='softplus', name='z_log_var')(x)

    return Model(inputs=smiles_input, outputs=[z_mean, z_log_var], name='encoder')


def decoder(msl: int, latent_dim: int, n_mha_layers: int, n_mha_heads: int, dim: int, embedding_layer: Layer) -> Model:
    """
    Decoder subnetwork.
    :param msl: Maximal SMILES length.
    :param latent_dim: Latent dimensionality.
    :param n_mha_layers: Number of Multi-Head Attention (MHA) layers.
    :param n_mha_heads: Number of MHA heads.
    :param dim: Internal model dimensionality.
    :param embedding_layer: Embedding layer.
    :return:
    """
    smiles_input = Input(shape=(msl,))
    latent_input = Input(shape=(latent_dim,))
    features_input = Input(shape=(1, dim))

    look_ahead_mask = create_look_ahead_mask(K.int_shape(smiles_input)[-1])
    dec_padding_mask = create_padding_mask(smiles_input)
    combined_mask = tf.maximum(dec_padding_mask, look_ahead_mask)
    combined_mask = tf.math.equal(combined_mask, 0)

    z = Dense(dim * 4, kernel_initializer='he_uniform')(latent_input)
    z = FTSwishG()(z)
    z = Dense(dim, kernel_initializer='glorot_uniform')(z)
    z = Dropout(0.1)(z)
    z = K.expand_dims(z, axis=-2)

    x = embedding_layer(smiles_input)
    x = Lambda(lambda x: x * tf.math.sqrt(tf.cast(dim, tf.float32)))(x)
    x = positional_encoding()(x)
    x = Dropout(0.1)(x)

    x = Concatenate(axis=-2)([z, x[:, :-1, :]])

    for l in range(n_mha_layers):
        att = LayerNormalization()(x)
        att = MultiHeadAttention(num_heads=n_mha_heads, key_dim=dim // n_mha_heads)(att, att, attention_mask=combined_mask)
        att = Dropout(0.1)(att)

        att = LayerScale()(att)

        x = Add()([att, x])

        att = LayerNormalization()(x)
        att = MultiHeadAttention(num_heads=n_mha_heads, key_dim=dim // n_mha_heads)(att, features_input)
        att = Dropout(0.1)(att)

        att = LayerScale()(att)

        x = Add()([att, x])

        ff = LayerNormalization()(x)
        ff = Dense(dim * 4, kernel_initializer='he_uniform')(ff)
        ff = FTSwishG()(ff)
        ff = Dense(dim, kernel_initializer='glorot_uniform')(ff)
        ff = Dropout(0.1)(ff)

        ff = LayerScale()(ff)

        x = Add()([ff, x])

    prob = Sequential([SharedEmbeddingOutput(embedding_layer.weights[0]), Activation('softmax', dtype='float32')])(x)

    return Model(inputs=[latent_input, features_input, smiles_input], outputs=prob, name='decoder')


# ----------------------------------------Training/Prediction-----------------------------------------------------------


def model(msl: int = 41, n_tokens: int = 25, latent_dim: int = 64, internal_dim: int = 256, n_features: int = 100,
          n_mha_layers: int = 4, n_mha_heads: int = 8, kld_coefficient: int = 20) -> Model:# , step: K.variable = None, max_steps: int = None) -> Model:
    """
    This block creates the model object.
    :param kld_coefficient: KLD loss coefficient.
    :parameter msl: Maximal SMILES length.
    :parameter n_tokens: Number of unique tokens.
    :parameter latent_dim: Latent dimensionality.
    :parameter internal_dim: Internal dimensionality of the model.
    :parameter n_features: Number of features in the query vector.
    :param n_mha_layers: Number of Multi-Head Attention (MHA) layers.
    :param n_mha_heads: Number of MHA heads.
    :param step: K.variable that tells the current step.
    :param max_steps: Number of batches per epoch multiplied by the number of epochs.
    :return: Model
    """
    smiles_input = Input(shape=(msl,))
    features_input = Input(shape=(n_features,))
    features = K.expand_dims(features_input, axis=1)
    features = features_layer(n_features=n_features, dim=internal_dim)(features)

    embedding_layer = get_embedding_layer(n_tokens, internal_dim)
    z_mean, z_log_var = encoder(msl=msl, latent_dim=latent_dim, dim=internal_dim,
                                embedding_layer=embedding_layer)(smiles_input)
    z_sampled = DistributionLambda(make_distribution_fn=lambda x: PowerSpherical(x[0], K.squeeze(x[1] + 1, axis=-1)),
                                   convert_to_tensor_fn=Distribution.sample)([z_mean, z_log_var])
    smiles_prob = decoder(msl=msl, latent_dim=latent_dim, n_mha_layers=n_mha_layers, n_mha_heads=n_mha_heads,
                          dim=internal_dim, embedding_layer=embedding_layer)([z_sampled, features, smiles_input])

    model_obj = Model(inputs=[smiles_input, features_input], outputs=smiles_prob)
    optimizer = AdaBeliefOptimizer(learning_rate=0.0005, weight_decay=0.0002, print_change_log=False)
    # optimizer = Adam(learning_rate=0.0005)
    optimizer.get_gradients = get_centralized_gradients_function(optimizer)

    kl_loss = KL(dtype=tf.float32)([z_mean, z_log_var])  # step=step, max_steps=max_steps, dtype=tf.float32)([z_mean, z_log_var])
    model_obj.add_loss(kl_loss * kld_coefficient)
    model_obj.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=[mask_acc, rec_rate])

    return model_obj


def sampler(msl: int, latent_dim: int, batch_size: int, model_obj: Model, features_query: np.ndarray) -> np.ndarray:
    """
    Samples one batch of SMILES out of random distribution according to the given query.
    :param msl: Maximal SMILES length.
    :param latent_dim: Latent dimensionality.
    :param batch_size: Batch size.
    :param model_obj: Model object.
    :param features_query: Query vector.
    :return: np.ndarray - Tokenized SMILES.
    """
    hpu = SphericalUniform(latent_dim)
    z_sampled = hpu.sample(batch_size).numpy()

    features = np.expand_dims(features_query, axis=1)
    features = model_obj.get_layer(name='features_layer').predict(features)
    smiles = np.zeros((batch_size, msl))

    for char_position in range(msl):
        smiles_prob = model_obj.get_layer(name='decoder').predict((z_sampled, features, smiles))
        smiles_pred = tf.random.categorical(tf.math.log(smiles_prob[:, char_position, :]), num_samples=1)
        # smiles_pred = np.argmax(smiles_prob, axis=-1)
        smiles[:, char_position] = smiles_pred[:, 0]

    return smiles
