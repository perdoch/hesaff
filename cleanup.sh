#!/bin/bash 

# Artifacts of the build and testing process

rm -rf wheelhouse
#sudo rm -rf multibuild
rm -rf dist
rm -rf __pycache__
rm -rf pyhesaff.egg-info
rm -rf pyhesaff/libhesaff*
rm -rf pyhesaff/__pycache__
rm -rf _skbuild

rm Dockerfile_*
rm opencv-docker-tag.txt


#PURGE="True"
PURGE="False"
if [ "$PURGE" = "True" ]; then 
    rm get-pip.py
fi
