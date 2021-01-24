#!/bin/bash
#SBATCH -N 1 #one node
#SBATCH -n 1 #one task
#SBATCH -t 00:05:00 #maximum wall time of 5 minutes
#SBATCH --reservation=perfvec

#load environment modules that correspond to the build environment
#and puts VTune in path

module load intel intel-advisor

#run command line Advisor, with survey analysis 

advisor --collect=survey --project-dir=./nbody-advisor -- ./app-ICC 

#run command line Advisor, with Roofline analysis (need to run Survey analysis first)

#advisor --collect=roofline --project-dir=./nbody-advisor -- ./app-ICC


