# ACoVAE
Attention-Based Conditional Variational Autoencoder.

## Project overview
A new attention-based conditional variational autoencoder neural network architecture based on recent developments in attention-based methods.

## Prerequisites
- **Operating system** : Linux with CUDA support
- **Hardware** : GPU compatible with tensorflow
- **Python environment manager** : anaconda, miniconda...

## Project structure
```bash
.
└── ACoVAE/
    ├── doc/
    │   └── Pharms_transformer.pdf # A visual representation of the ACoVAE neural network 
    ├── utils/
    │   ├── __init__.py # Imports
    │   ├── data_preparation_umaps.sh # data preparation - only for GTM universal maps. 
    │   ├── layers.py # Network implementation
    │   └── utils.py # Utilities (SMILESParser class)
    ├── training_data/
    │   └── README.md
    ├── LICENSE # GNU General Public License v3.0 license
    ├── README.md
    ├── main.py
    ├── model_parameters_standard.yaml # Standard parameters for model training
    └── requirements.txt # Requirements to create the python environment
```

## Installation
### Clone the repository:
```bash
git clone https://github.com/Laboratoire-de-Chemoinformatique/ACoVAE.git
cd ACoVAE
mkdir model
```
### Install dependencies
```bash
conda create --name ACoVAE --file requirements.txt
pip install adabelief-tf CGRtools
```
### (Optional) Check installation
```bash
conda list | grep tensorflow
conda list | grep cudatoolkit
conda list | grep cudnn
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Usage 
### Input file
#### Input file description
The input file is a descriptor matrix file under the .svm format.

It can be generated using the ISIDA/Fragmentor software (Open access, request the software using the form : https://infochim.u-strasbg.fr/-DOWNLOADS-SOFTWARE-.html). 

Please refer to the ISIDA/Fragmentor documentation to generate the descriptor matrix.

#### Input file format
- **Column 1** : SMILES
- **Column 2-end** : Descriptor matrix in a .svm file following the libSVM format. These columns consist in a pair of values separated by a ":". The first value identifies the fragment's index in the header file (.hdr file created by ISIDA/Fragmentor), the second value is the fragment count.

#### Example dataset
An example dataset can be downloaded here : https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/ILWSLF

### Commands
## Model training
```bash 
conda activate ACoVAE
python3 main.py -i ./training_data/chembl23_umap1.svm -m ./model/model_name -mp model_parameters_standard.yaml --log log_19-07-2022
````
For extended functions, consult the help command:

```bash
python main.py --help
```

### Results and selected model
Check the log file for model quality scores : epoch, loss, mask_acc, rec_rate, val_loss, val_mask_acc, val_rec_rate

val_mask_acc (accuracy - character-specific reconstruction rate), val_rec_rate (reconstruction rate) are the values to follow.

Select the model which suits your needs in the /model folder.

## Generate compounds
First, generate the descriptor vector for a known compound, using ISIDA/Fragmentor.
**Note** : The first column in the output file must be the ID of the compound, not the SMILES. (see ./training_data/1.svm for an example).

Then, use the generated vector as seeds for new compounds generation.
```bash
mkdir sampled_smi
python main.py -f ./training_data/1.svm -n 1000 -m ./model/model_name_99_0.98 -sp ./model/model_name_smi_parser.pkl -mp model_parameters_standard.yaml -o ./sampled_smi/known_compound_vector_sampled.smi
```
- **sp** : SMILES parser pickle object created during the network training. It is needed for sampling.
- **m** : the model created during the network training.
- **mp** : the model yaml created during the network training.
- **n** : Number of batches sampled per query.

## Credits

* Arkadii Lin, Daniyar Mazitov, William Bort, Timur Madzhidov and Alexandre Varnek
* Kazan Federal University, Russia
* University of Strasbourg, France

## License

Distributed under the GNU GENERAL PUBLIC LICENSE Version 3. See the `LICENSE` link in the additional resources for more information.

## Additional Resources

* [GitHub](https://github.com/Laboratoire-de-Chemoinformatique/ACoVAE)
* [License](https://www.gnu.org/licenses/gpl-3.0.en.html)
* [Publication](https://pubmed.ncbi.nlm.nih.gov/36332178/)

