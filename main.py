#############################################################################
# Molecular CVAE employing elements of a Transformer architecture
#############################################################################
# GNU GPL https://www.gnu.org/licenses/gpl-3.0.en.html
#############################################################################
# Corresponding Authors: Timur Madzhidov and Alexandre Varnek
# Corresponding Authors' emails: tmadzhidov@gmail.com and varnek@unistra.fr
# Main contributors: Arkadii Lin, Daniyar Mazitov, William Bort
# Copyright: Copyright 2021 !!!!!!!!TO BE DISCUSSED!!!!!!!!!
# Credits: Kazan Federal University, Russia
#          University of Strasbourg, France
# License: GNU GPL https://www.gnu.org/licenses/gpl-3.0.en.html
# Version: 00.01
#############################################################################
from io import TextIOWrapper
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras import callbacks as cb

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class KLAnnealing(cb.Callback):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def on_epoch_end(self, epoch, logs={}):
        self.step += 1.


def train(training_file: TextIOWrapper, model_name: str, model_parameters: dict, log_file: str,
          load_weights: bool = False, smiles_parser: str = None):
    from utils import model, SMILESParser, BatchWM

    smi_parser = SMILESParser(max_length=model_parameters['maximal_smiles_length'],
                              header=model_parameters['file_header'])

    if load_weights:
        smi_parser = pickle.load(open(smiles_parser, 'rb'))

    data, features = smi_parser.read_file(training_file)
    if not load_weights:
        pickle.dump(smi_parser, open(model_name + '_smi_parser.pkl', 'wb'))
    p = np.random.permutation(data.shape[0])
    data, features = data[p], features[p]
    data_train, data_val = data[:(data.shape[0]*8) // 10], data[(data.shape[0]*8) // 10:]
    features_train, features_val = features[:(data.shape[0]*8) // 10], features[(data.shape[0]*8) // 10:]
    del data
    del features

    # step_index = tf.keras.backend.variable(0.0, dtype=tf.float32)
    # max_steps = (data_train // model_parameters['batch_size']) * model_parameters['n_epochs']
    model_obj = model(msl=model_parameters['maximal_smiles_length']+1, n_tokens=smi_parser.n_tokens,
                      latent_dim=model_parameters['latent_dimensionality'], n_features=smi_parser.n_features,
                      n_mha_layers=model_parameters['n_mha_layers'], n_mha_heads=model_parameters['n_mha_heads'],
                      internal_dim=model_parameters['internal_dim'],
                      kld_coefficient=model_parameters['kld_coefficient'])#, step=step_index, max_steps=max_steps)
    if load_weights:
        model_obj.load_weights(model_name)
        model_name += '_retrained'
    lr_s = BatchWM()
    callbacks = [cb.CSVLogger(log_file, append=False), lr_s,
                 cb.ModelCheckpoint(model_name+'_{epoch:02d}_{val_mask_acc:.2f}', monitor='val_mask_acc',
                                    save_weights_only=True, mode='max', save_best_only=True)]
                 # KLAnnealing(step=step_index)]
    model_obj.fit((data_train, features_train), y=data_train, validation_data=((data_val, features_val), data_val),
                  batch_size=model_parameters['batch_size'], epochs=model_parameters['n_epochs'], callbacks=callbacks,
                  shuffle=True)


def sample(features_file: TextIOWrapper, model_name: str, model_parameters: dict, output_file: TextIOWrapper,
           smiles_parser: str, n_samples: int):
    from utils import model, sampler

    smi_parser = pickle.load(open(smiles_parser, 'rb'))
    smi_parser.generate_inverse_symbols_dict()
    data = smi_parser.parse_features(features_file)

    model_obj = model(msl=model_parameters['maximal_smiles_length']+1, n_tokens=smi_parser.n_tokens,
                      latent_dim=model_parameters['latent_dimensionality'], n_features=smi_parser.n_features,
                      n_mha_layers=model_parameters['n_mha_layers'], n_mha_heads=model_parameters['n_mha_heads'],
                      internal_dim=model_parameters['internal_dim'],
                      kld_coefficient=model_parameters['kld_coefficient'])
    model_obj.load_weights(model_name)
    output_file.write('SMILES\tQuery_ID\n')
    for query in data:
        query_id = query[0]
        features = np.repeat(np.expand_dims(query[1:], axis=0), model_parameters['batch_size'], axis=0)
        for i in range(n_samples):
            print(f'Query {query_id}: sampling {i}...')
            gen_smiles = sampler(msl=model_parameters['maximal_smiles_length']+1,
                                 latent_dim=model_parameters['latent_dimensionality'],
                                 batch_size=model_parameters['batch_size'], model_obj=model_obj,
                                 features_query=features)
            smi_parser.write_file(gen_smiles, output_file, query_id=int(query_id))
            output_file.flush()
    output_file.close()


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Molecular CVAE employing elements of a Transformer architecture.",
                                     epilog="Arkadii Lin, Daniyar Mazitov, Strasbourg (France) / Kazan (Russia) 2021",
                                     prog="TransCVAE")

    parser.add_argument('-i', '--input', type=argparse.FileType('r'), help='SVM file with SMILES in the first column '
                                                                           'and the corresponding features in others '
                                                                           'formatted as "feature_num:feature_value". '
                                                                           'Note that features with zero value are '
                                                                           'omitted, but the last feature is always '
                                                                           'mentioned even if it is zero. This file '
                                                                           'will be used for CVAE training.')
    parser.add_argument('-f', '--features', type=argparse.FileType('r'), help='Features file that is given explicitly '
                                                                              'only when sampling is required. The file'
                                                                              ' should be formatted as SVM, where the '
                                                                              'first column contains query ID (e.g., '
                                                                              'regular order numbering).')
    parser.add_argument('-o', '--output', type=argparse.FileType('w'), help='Output file with sampled and filtered '
                                                                            'SMILES strings. It contains two columns: '
                                                                            'a SMILES string and query ID. Note that '
                                                                            'the output file name is specified only '
                                                                            'when sampling is required.')
    parser.add_argument('-n', '--num_samples', type=int, default=1, help='Number of batches sampled per query.')
    parser.add_argument('-m', '--model', type=str, default='models/model', help='Model name with no extension.',
                        required=True)
    parser.add_argument('-lw', '--load_weights', action='store_true', help='To load the weights of the model.')
    parser.add_argument('-mp', '--model_parameters', type=str, help='YAML file containing model parameters.')
    parser.add_argument('-sp', '--smiles_parser', type=str, help='SMILES parser pickle object created during the '
                                                                 'network training. It is needed for sampling.')
    parser.add_argument('--log', type=str, default='model.log', help='CSV model log file.')

    args = parser.parse_args()

    if args.model_parameters:
        with open(args.model_parameters, 'r') as in_stream:
            model_parameters = yaml.safe_load(in_stream)
    else:
        model_parameters = dict(maximal_smiles_length=80, latent_dimensionality=64, n_mha_layers=4, n_mha_heads=8,
                                batch_size=512, n_epochs=100, file_header=False, internal_dim=256, kld_coefficient=20)
        with open('models/model.yaml', 'w') as out_stream:
            yaml.dump(model_parameters, out_stream, default_flow_style=False)

    if args.input:
        train(args.input, args.model, model_parameters, args.log, args.load_weights, args.smiles_parser)
    else:
        sample(args.features, args.model, model_parameters, args.output, args.smiles_parser, args.num_samples)
