# MAFC-RAT-CJC
## Environment
### Create the Environment
Step 1: conda create --name rat python=3.6.5;

Step 2: activate the environment 'rat' by running 'conda activate rat';

Step 3: install the following libraries in 'rat'.

        Numpy: conda install -c anaconda numpy=1.14.3
        Pandas: conda install -c anaconda pandas=0.23.0
        Scikit-learn: conda install -c anaconda scikit-learn=0.19.1
        Treelib: conda install -c conda-forge treelib=1.5.5
        Gensim: conda install -c anaconda gensim=3.4.0
        Tensorflow: conda install -c aaronzs tensorflow=1.10.0
        H5Py: conda install -c anaconda h5py=2.9.0
        Matplotlib: conda install -c conda-forge matplotlib=2.2.2

### Export the Environment
Step 1: activate the environment 'rat' by running 'conda activate rat';

Step 2: conda env export --file rat_20210627.yml

Step 3: find and store the environment file 'rat_20210627.yml' as shown in the project directory.

### Import the Environment
Step 1: move 'rat_20210627.yml' to 'd:\' of the target computer;

Step 2: conda env create -f  d:\rat_20210627.yml.