#!/bin/bash 

# Artifacts of the build and testing process

sudo rm -rf wheelhouse
#sudo rm -rf multibuild
sudo rm -rf dist
sudo rm -rf __pycache__
sudo rm -rf pyhesaff.egg-info
sudo rm -rf pyhesaff/libhesaff*
sudo rm -rf pyhesaff/__pycache__
sudo rm -rf _skbuild

rm Dockerfile_*
rm opencv-docker-tag.txt


#PURGE="True"
PURGE="False"
if [ "$PURGE" = "True" ]; then 
    rm get-pip.py
fi
