#! /bin/bash

# This is the data preparation workflow that is required for GTM universal maps to run Transformer-based Conditional Variational AE.
# Inputs are row ISIDA descriptors in SVM format, corresponding canonical SMILES strings given in a separate file in the first column, mymap_projection.pri,
# mymap_projection.xml and chromo files.

# Note that these paths are examplary. Replace them with your paths.
row_ISIDA_desc='/data/william/cvae_transformer_data/umap1_data/IA-FF-FC-AP-2-3.svm'
row_can_SMILES='/data/william/cvae_transformer_data/umap1_data/chembl23_uniq.smi_chid'
pri_file='/data/william/cvae_transformer_data/umap1_data/mymap_projection.pri'
xml_file='/data/william/cvae_transformer_data/umap1_data/mymap_projection.xml'
chromo='/data/william/cvae_transformer_data/umap1_data/chromo'

output_dir='/data/william/cvae_transformer_data/umap1_data/model_input'
chunk_size=$((20000))
n_cores=$((45))


# Create the output directory if such does not exist
mkdir -p ${output_dir}


# Detect whether the descriptors must be scaled or stay original
desc_preproc_type=$(awk '{print $3}' $chromo)


# Transform the descriptors using GA facilities
if [ $desc_preproc_type = 'scaled' ]
then
    /home/opt/libsvm-GAconfig/svmPreProc.pl $row_ISIDA_desc selfile=$pri_file explicit=yes scale=yes output=${output_dir}/ga_preproc.svm
elif [ $desc_preproc_type = 'orig' ]
then
    /home/opt/libsvm-GAconfig/svmPreProc.pl $row_ISIDA_desc selfile=$pri_file explicit=yes output=${output_dir}/ga_preproc.svm
else
    echo 'Unrecognized scaling action in your chromo!'
fi


# Split the obtained file into chunks
mkdir -p ${output_dir}/chunks

split -d -l $chunk_size ${output_dir}/ga_preproc.svm ${output_dir}/chunks/chunk_


# Project the chunks onto our map
for f in $(ls ${output_dir}/chunks/chunk_* | sort)
do
    n_jobs=$(ps -e --no-headers | grep 'GTMapTool' | wc -l)
    while [ $n_jobs -eq $n_cores ]
    do
        sleep 10
        n_jobs=$(ps -e --no-headers | grep 'GTMapTool' | wc -l)
    done
    /home/opt/ISIDAGTM2016/lnx64/GTMapTool -j -y $f -x $xml_file -o ${f}_gtm --3D &
done

sleep 1

n_jobs=$(ps -e --no-headers | grep 'GTMapTool' | wc -l)
while [ $n_jobs -gt 0 ]
do
    sleep 1
    n_jobs=$(ps -e --no-headers | grep 'GTMapTool' | wc -l)
done

# Gather projected responsibilities together
touch ${output_dir}/responsibilities.rsvm

for f in $(ls ${output_dir}/chunks/chunk_*_gtmR.svm | sort)
do
    cat $f >> ${output_dir}/responsibilities.rsvm
done


# Gather descriptors preprocessed by GTMapTool
touch ${output_dir}/gtm_preprocessed.csv

for f in $(ls ${output_dir}/chunks/chunk_*_gtmZtest.mat | sort)
do
    head -n -1 $f | tail -n +2 >> ${output_dir}/gtm_preprocessed.csv
done


# Extraction of SMILES
awk '{print $1}' $row_can_SMILES > ${output_dir}/molecules.smi


# Transform CSV to SVM incorporating SMILES strings as the first column
export output_dir

python3 - << EOF
import os

smi = [line.strip() for line in open(os.environ['output_dir'] + '/molecules.smi') if line.strip()]

with open(os.environ['output_dir'] + '/gtm_preprocessed.svm', 'w') as out:
    tmp = []
    for i, line in enumerate(open(os.environ['output_dir'] + '/gtm_preprocessed.csv'), 1):
        if not line.strip():
            break

        d = [smi[i-1]]
        sline = line.strip().split(',')
        for j, item in enumerate(sline[:-1], 1):
            v = float(item)
            if v != 0.:
                d.append(f'{j}:{round(v, 6)}')
        d.append(f'{len(sline)}:{round(float(sline[-1]), 6)}\n')
        tmp.append(' '.join(d))
        del d

        if i % 50000 == 0:
            print(f'{i} lines passed..')
            out.writelines(tmp)
            out.flush()
            del tmp
            tmp = []
    if tmp:
        out.writelines(tmp)
        out.flush()
        del tmp
EOF


# Remove useless files
rm ${output_dir}/gtm_preprocessed.csv ${output_dir}/molecules.smi
rm -r ${output_dir}/chunks

# An example of how to train cvae_transformer
# python3 main.py -i ${output_dir}/gtm_preprocessed.svm -m $my_output_model_dir/my_model -mp my_model_params.yaml --log $my_output_model_dir/my_model.log

# An example of how to run the pre-trained cvae_transformer
# python3 main.py -f my_query_isida.svm -m $my_output_model_dir/my_model_${desirable_epoch}_${desirable_metric_val} -mp my_model_params.yaml -n 10 -sp $my_output_model_dir/my_model_smi_parser.pkl -o $my_output_model_dir/my_sampling.smi

# For cvae_transformer input options, see python3 main.py -h/--help.
