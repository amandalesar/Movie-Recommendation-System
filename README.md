# Movie Recommendation System

The collaborative filtering learning algorithm will be implemented and applied to a dataset of movie ratings, to build a movie recommendation system. 

I will adapt my recommeder system from the second part of the eighth exercise from Andrew Ng’s Machine Learning Course on Coursera. The provided dataset consists of ratings on a scale of 1 to 5. The dataset has 943 users and 1682 movies. This dataset is from MovieLens 100k Dataset from GroupLens Research.

# Running the Project 

- Make sure you have MATLAB or Octave installed. 
- Clone the project to your local machine. 
- Run movierecommendations.m. For a guided implementation, you can instead run the live script RecommenderSystems.mlx. 

# Project Details

The collaborative filtering learning algorithm will be implemented and applied to a dataset of movie ratings, to build a movie recommendation system. 

The function cofiCostFunc.m computes the collaborative fitlering objective function and gradient. The function fmincg will be used to learn the parameters for collaborative filtering.

First we'll load in our data (ex_movies.mat). This will provide the variable Y, which will store movie ratings for each user (with dimensions num_movies x num_user).  The matrix R is an binary-valued indicator matrix, where R(i,j) = 1 if user j gave a rating to movie i, and R(i,j) = 0 otherwise.

The objective of collaborative filtering is to predict movie ratings for the movies that users have not yet rated, that is, the entries with R(i,j) = 0. This will allow us to recommend the movies with the highest predicted ratings to the user.

This data includes the following:
* Y is a 1682 x 943 matrix, containing ratings (1 - 5) of 1682 movies on 943 users
* R is a 1682 x 943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i

We will work with the matrices X and θ. The i-th row of X corresponds to the feature vector for the i-th movie. The j-th row of θ corresponds to a parameter vector for the j-th user. 

First the cost function will be implemented. The collaborative filtering algorithm in the context of movie recommendations considers a set of n-dimensional parameter vectors, where the model predicts the rating for movie *i* by user *j*. Given a dataset that consists of a set of ratings produced by some users on some movies, we wish to learn the parameter vectors that produce the best fit (minimizes the squared error).

We will use cofiCostFunc.m to return the cost function (with regularization) in the variable J, and the gradients in the variables X_grad and Theta_grad. The cost function and gradients are given by: 

![cost](https://github.com/amandalesar/movie_recommendation_system/blob/master/images/costfunction.png)
![grads](https://github.com/amandalesar/movie_recommendation_system/blob/master/images/grads.png)

Note that the cost should be accumulated for user *i* and movie *j* only if R(i,j) = 1. 

A gradient check (checkCostFunction.m) will be applied to numerically check the implementation of the gradients. If the implementation is correct, then the analytical and numerical gradients match up closely.

Now the algorithm can be trained to make movie recommendations. In the code, you can enter your own movie preferences, so that later when the algorithm runs, you can get your own movie recommendations! The list of all movies and their number in the dataset can be found listed in the file movie_idx.txt. After the additional ratings have been added to the dataset, the code will proceed to train the collaborative filtering model. This will learn the parameters X and Theta.

The top predicted movies are then output. By giving your ratings above to train the recommender system, you can see which movies you might like!

