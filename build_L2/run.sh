#!/bin/sh
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name="neonBij"
#SBATCH --output="log.txt"

#python base_h5.py
#mpiexec -n 25 python add_dp04.py
#mpiexec -n 25 python add_rad.py
#mpiexec -n 25 python add_2dws.py
#mpiexec -n 25 python add_chm.py
#mpiexec -n 25 python add_clim.py
#mpiexec -n 25 python add_era5.py
#mpiexec -n 25 python add_ghflx.py
#mpiexec -n 25 python add_hssf.py
#mpiexec -n 25 python add_lai.py
#mpiexec -n 25 python add_ndvi.py
#mpiexec -n 25 python add_nlcd.py
#mpiexec -n 25 python add_precp.py
#mpiexec -n 25 python add_rad.py
#mpiexec -n 25 python add_soilm.py
#mpiexec -n 25 python add_tssf.py
#mpiexec -n 25 python add_vcf.py
#mpiexec -n 25 python add_aniso.py
mpiexec -n 25 python add_grads.py

