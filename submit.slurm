#!/bin/bash
#SBATCH -N 1 #one node
#SBATCH -n 1 #one task
#SBATCH -t 00:05:00 #maximum wall time of 5 minutes
#SBATCH --reservation=vector

#load environment modules that correspond to the build environment
#and puts VTune in path

module load intel intel-advisor

#run command line Advisor, with survey analysis 

advixe-cl -c survey -project-dir nbody-advisor -- ./app-ICC 

#run command line Advisor, with Trip count analysis

#advixe-cl -c tripcounts -flop -project-dir nbody-advisor -- ./app-ICC


