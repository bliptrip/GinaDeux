# GinaDeux
Feature-rich webapp for semi-automated image segmentation and analysis of pictures of plants, specifically designed for
cranberries but extensible to other crops.  Derived from https://doi.org/10.1371/journal.pone.0160439.

## Features

* Access pictures from cloud-based datastore services such as Google Drive, Box.com, Dropbox, and Cyverse.
* Upload pictures to server for server-based storage.
* Interactive tuning of classical segmentation parameters for immediate feedback of segmentation quality.
* Interactive generation of neural network images for advanced segmentation of images.
* Attach metadata to sets of images, such as dates (year/month), growth degree days, genotypes, location of samples,
  camera settings, color corrected tags, and lens distortion corrected tags.
* Dashboard for viewing statistics of dataset pools, with the ability to filter based on metadata tags.

## Architecture

## Requirements
This web application is best installed on a Linux or other \*Nix-based system with ample resources to adequately service the
expected amount of network traffic.

### Frontend (javascript)

* [Reactjs](https://reactjs.org/)
* [Redux](https://redux.js.org/)
* [Node Package Manager Packages](https://www.npmjs.com/)
    - [Paper.js](https://www.npmjs.com/package/paper)

### Backend (python)

* [Flask Microframework](https://flask.palletsprojects.com/en/1.1.x/)
* [MongoDB Python Driver](https://www.mongodb.com/blog/post/getting-started-with-python-and-mongodb)
* [OpenCV](https://opencv.org/links)
* [Keras Python Deep Learning Library](https://keras.io/)

## Installation

