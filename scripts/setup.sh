#!/bin/sh
#
# Script to setup the required packages for GinaDeux.  Right now this assumes
# a Linux OS with apt-type package installation.
sudo apt --yes update
sudo apt --yes upgrade
sudo apt --yes install libopencv-dev mongodb nodejs npm python3 python3-pip

if [[ -n $SETUP_VIRGIN ]]; then #This is invoked when creating project from scratch -- if checking out from github, it's unecessary.
    sudo npm install npx -g
    sudo npm install create-react-app -g
    npx create-react-app frontend
    cd frontend
    npm install --save google-map-react
    cd ..
fi
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
