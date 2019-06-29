#!/bin/bash 
sudo rm -rf wheelhouse
sudo rm -rf multibuild
sudo rm -rf __pycache__
sudo rm -rf pyhesaff.egg-info
sudo rm -rf pyhesaff/libhesaff*
sudo rm -rf pyhesaff/__pycache__
sudo rm -rf _skbuild

rm Dockerfile_*
rm opencv-docker-tag.txt
