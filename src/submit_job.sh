#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J odc2024
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 10:00
# request 24GB of system-memory
#BSUB -R "rusage[mem=24GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
# # Load the cuda module
# module load cuda/11.6

# /appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source ~/miniconda3/etc/profile.d/conda.sh
conda activate odc2024

# # loads automatically also numpy and python3 and underlying dependencies for our python 3.11.7
# module load pandas/2.1.3-python-3.11.7

# # in case you have created a virtual environment,
# # activate it first:
# source foobar/bin/activate

# # use this for LSF to collect the stdout & stderr
# python3 helloworld.py

# # use this for unbuffered output, so that you can check in real-time
# # (with tail -f Output_.out Output_.err)
# # what your program was printing "on the screen"
# python3 -u helloworld.py

# # use this for just piping everything into a file, 
# # the program knows then, that it's outputting to a file
# # and not to a screen, and also combine stdout&stderr
# python3 helloworld.py > joboutput_$LSB_JOBID.out 2>&1
cd /zhome/46/2/189047/projects/biomed2024
# python src/train_xgboost_outlier_detection.py -c src/outlier-challenge-config_gbar_dtu.json -d train_files.txt # custom_train_list_100.txt
# python src/validate_xgboost_outlier_detection.py -c src/outlier-challenge-config_gbar_dtu.json -d custom_validation_list_100.txt
python src/test_xgboost_outlier_detection.py -c src/outlier-challenge-config_gbar_dtu.json -d test_files_200.txt
# python src/submit_outlier_detections.py -c src/outlier-challenge-config_gbar_dtu.json -d test_files_200.txt