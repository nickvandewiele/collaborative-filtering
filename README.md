# Collaborative Filtering

## Introduction


Here, I explore [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering), a technique used in recommender systems.

I focus on 2 types of collaborative filtering: user-based and item-based. I've created a memory-based implementation for both of them.

## Data

The [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/) is used for building the recommender systems.


Copy the [dataset](http://files.grouplens.org/datasets/movielens/ml-100k.zip), and unzip it into a folder.

## Implementation


The implementation of the collaborative filtering algorithms is done using Pandas. 

For the item-based collaborative filtering algorithm, I based my implementation on the excellent Udemy course [Taming Big Data with Apache Spark and Python - Hands On!](https://www.udemy.com/course/taming-big-data-with-apache-spark-hands-on/) by Frank Kane. The motivation to implement with Pandas the algorithm is to compare implementations with a library for distributed computing like Spark. For the user-based approached, I did not follow a specific recipe.

Jupyter notebooks ([here](notebooks/item-based-collaborative-filtering.ipynb) and [here](notebooks/user-based-collaborative-filtering.ipynb)) explain the methodology and the followed steps. I also develop a strategy to measure the quality of the recommendations [here](notebooks/measuring-quality-item-based-CF.ipynb) and [here](notebooks/measuring-quality-user-based-CF.ipynb).