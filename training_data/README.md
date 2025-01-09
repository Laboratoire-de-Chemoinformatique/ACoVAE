# Input file description
The input file is a descriptor matrix file under the .svm format.

It can be generated using the ISIDA/Fragmentor software (Open access, request the software using the form : https://infochim.u-strasbg.fr/-DOWNLOADS-SOFTWARE-.html). 

Please refer to the ISIDA/Fragmentor documentation to generate the descriptor matrix.

# Input file format
- **Column 1** : SMILES
- **Column 2-end** : Descriptor matrix in a .svm file following the libSVM format. These columns consist in a pair of values separated by a ":". The first value identifies the fragment's index in the header file (.hdr file created by ISIDA/Fragmentor), the second value is the fragment count.
