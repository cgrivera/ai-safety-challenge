# first argument is the zip file
# the second argument is the absolute path to the tanksworld executable
unzip -o -d test $1
source ~/anaconda3/etc/profile.d/conda.sh
conda env remove -y -n tanksworld_test
conda create -y -n tanksworld_test python=3.6
conda activate tanksworld_test
conda install -y mpi4py
pip --cert cert.crt install  -e ../
pip --cert cert.crt install  -e test/*/
python  test_submission.py --exe $2 --package_name $(perl -lane 'print $1 if /name\s*=\W*(\w+)\W*/' test/*/setup.py) 

