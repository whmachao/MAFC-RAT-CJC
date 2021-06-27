# MAFC-RAT-CJC
## Environment
### Create the Environment
Step 1: conda create --name env_name python=3.6.5;

Step 2: activate the environment 'rat' by running 'conda activate rat';

Step 3: install the following libraries in 'rat'.

        Numpy: conda install -c anaconda numpy
        Pandas: conda install -c anaconda pandas
        Scikit-learn: conda install -c anaconda scikit-learn
        Treelib: conda install -c conda-forge treelib
        Gensim: conda install -c anaconda gensim
        Tensorflow: conda install -c aaronzs tensorflow-gpu
        Matplotlib: conda install -c conda-forge matplotlib

### Export the Environment
Step 1: activate the environment 'rat' by running 'conda activate rat';

Step 2: conda env export --file rat_20210627.yml

Step 3: find and store the environment file 'rat_20210627.yml' as shown in the project directory.

### Import the Environment
Step 1: move 'rat_20210627.yml' to 'd:\' of the target computer;

Step 2: conda env create -f  d:\rat_20210627.yml.