# MegamanCausalSceneGeneration
Causal Modeling Approach to generate a two character scene between Megaman and Shademan


## Authors
Siddarth Sathyanarayanan, Pranita Deshpande, Dinny Mathew, Yashvin Jagarlamudi

## Background

This project implements causal scene generation using Megaman Sprites from the game Megaman 7. This project applies causal modeling with the use Directed Acyclic Graphs (DAGs), Bayesian Networks, 
conditioning, and interventions. In addition, the image generation is performed using deep learning architecture like Variational Auto Encoders (VAEs) and Generative Advesarial Networks (GANs).


## Introduction 

Initially we have two images with 3 attributes each: size, position, and action for the two characters Megaman and Shademan. Size can be tiny or magnificent and position can either be left or right.
Action is any action that either of the characters can perform. The left side of the image is attacking character and the right image is the defending character. Likewise,  the environment chosen will
affect the background of the fight. Given these attributes, a scence will be generated.

## Repository Structure

/mgan_dataset folder contains all the images used to train the VAE and GAN \
/Dag.Rmd is the r markdown file to make the DAG for the bayesian network \
/GAN.ipynb is a python notebook to create images from a GAN \
/VAE.ipynb is a python notebook to create images from a VAE \
/Causal scene generation/Causal scene generation/ folder contains the images for conditioning and interventions \
/Causal_Scene_Generation.ipynb is a python notebook run in Colab that contains the python UI to run the scene generation 

## Dependencies
Python libraries and packages needed for this project are

1)keras \
2) torch \
3) matplotlib \
4) Pillow \
5) numpy \
6) collections \
7) googlesearch \
8) ipywidgets \
9) IPython.display 
