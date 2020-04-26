# Agent Models for Causal Scene Generation

## Authors
[Harish Ramani](https://www.linkedin.com/in/harishramani1792/), [Nalin Gupta](https://www.linkedin.com/in/nalin-gupta/), [Shih gian lee](https://www.linkedin.com/in/shihgianlee/)

## Introduction

The causal scene generation project involves generating 2D procedural images from natural language where the causal model describes the relationship between the entities involved in the caption. Flickr8k dataset contains captions and the corresponding image to it. Using these captions, along with the causal model we can probabilistically reason on it.

**Example**:
A girl is going to a wooden building.

In the above example, if we have a causal model, we can ask the question *How would the picture look if the girl was replaced by a boy given that we have observed the action was walking and the environment is a wooden building.* 

To achieve this, we need to first probabilistically express the relationship between the different entities present. Object-oriented programming (OOP) has been around since the 70s[1] and has been used to model relationships between different entities. However, OOP is not prevalent in the probabilistic programming (PP) community. In this tutorial, we first see how to model relationships probabilistically using pyro.

[Video Abstract](https://youtu.be/bL-xlx5_KbQ)

## Causal ORM

The way we have created objects are directly from a sqlite database. This gives us the ability to do causal-inference on a knowledge base directly. 

## How to set up the environment

### **Using Docker**

If docker isn't installed, please refer this [documentation](https://docs.docker.com/get-docker/) on how to install docker for your respective OS.

If docker is installed, just use the runDocker script to build the image and run the jupyter notebook.
When the image is built for the first time, it will take some time to set up as it will download pytorch and torchvision and they are huge files.

``` bash
sh runDocker.sh
```

After the image is built, use the docker ps command to see the port in which the jupyter notebook is running. Usually it will take 32768 port.

``` bash
docker ps
```

### **Manual Setup**

If you dont wish to install docker, then use the requirements.txt to install the dependencies and run jupyter notebook by yourself.

``` bash
cd <code_folder>
pip install -r requirements.txt

jupyter notebook

```