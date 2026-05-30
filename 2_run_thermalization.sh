#!/bin/bash

for dir1 in _sys_*/; do
  cd $dir1

  if [ -d "adiabatic" ]; then
    cd adiabatic
    for dir2 in _fix_*/; do
      cd $dir2
      sed -i "s|python.*|python ../../../2_kcrpmd_thermalization.py |g" ../../../submit_template.slm
      sbatch ../../../submit_template.slm
      cd ../
    done
    cd ../
  fi

  if [ -d "kcrpmd_ori" ]; then
    cd kcrpmd_ori
    for dir2 in _fix_*/; do
      cd $dir2
      sed -i "s|python.*|python ../../../2_kcrpmd_thermalization.py |g" ../../../submit_template.slm
      sbatch ../../../submit_template.slm
      cd ../
    done
    cd ../
  fi

  if [ -d "kcrpmd_new" ]; then
    cd kcrpmd_new
    for dir2 in _fix_*/; do
      cd $dir2
      sed -i "s|python.*|python ../../../2_kcrpmd_thermalization.py |g" ../../../submit_template.slm
      sbatch ../../../submit_template.slm
      cd ../
    done
    cd ../
  fi

  cd ../
done

