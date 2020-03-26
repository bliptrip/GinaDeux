#!/bin/sh
#
# Script to setup the required packages for GinaDeux.  Right now this assumes
# a Linux OS with apt-type package installation.
sudo apt --yes install libopencv-dev nodejs npm python3 python3-pip

cd frontend
npm install #Install dependencies found in package.json
cd ../
cd backend
npm install #Install dependencies in package.json
sudo pip3 install virtualenv
mkdir -p ~/.envs
virtualenv ~/.envs/ginadeux/
source ~/.envs/ginadeux/bin/activate
pip3 install -r requirements.txt
