#!/bin/csh
#Runs download ensembles using parallel computing
#Runs twice per day: 0601 and 1201 UTC (for 00 and 12 UTC runs)

cd /home/disk/meso-home/jzagrod/Models/ensemble_tools
qsub -V -N ensdownload -cwd -o enslog.txt -e /dev/null ./process_ensembles.py

