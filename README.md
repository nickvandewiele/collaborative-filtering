# Item-based Collaborative Filtering

## Introduction


Here, I explore [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering), a technique used in recommender systems.

I focus on 2 types of collaborative filtering: 
- user-based 
- item-based


## Data

The [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/) is used for building the recommender systems.


Copy the [dataset](http://files.grouplens.org/datasets/movielens/ml-100k.zip), and unzip it into a folder.

## Implementation


The implementation of the collaborative filtering algorithms is done using Pandas. 

Nevertheless, for the item-based collaborative filtering algorithm, I based my implementation on the excellent Udemy course [Taming Big Data with Apache Spark and Python - Hands On!](https://www.udemy.com/course/taming-big-data-with-apache-spark-hands-on/) by Frank Kane. The motive to implement with Pandas the algorithm is to compare implementations with a library for distributed computing like Spark.


Jupyter notebooks ([here](item-based-collaborative-filtering.ipynb) and [here](user-based-collaborative-filtering.ipynb)) explain the methodology and the followed steps. 

More concise versions in python can be found [here](item-based-collaborative-filtering.py) and [here](user-based-collaborative-filtering.py), and can be run from the command line:

```python
python item-based-collaborative-filtering.py
python user_based_collaborative_filtering.py
```