% The collaborative filtering learning algorithm 
% will be implemented and applied to a dataset 
% of movie ratings. 
% I will adapt my recommeder system from the 
% second part of the eighth exercise from 
% Andrew Ng’s Machine Learning Course on Coursera.

% Load data
load('ex_movies.mat');

% Make movie recommendations

% Load movvie list
movieList = loadMovieList();

% Initialize my ratings
my_ratings = zeros(1682, 1);

% Check the file movie_idx.txt for id of each movie in our dataset
% Remember, 5 is best and 1 is worst!
num = input("How many movie ratings would you like to input? ");
for i = 1:num
    id = input("Enter movie id: ");
    rating = input("Enter rating: ");
    my_ratings(id) = rating;
end

% I have selected a few movies I liked / did not like and the ratings I gave are as follows:
% Use these if you would just like to test the code! 
% my_ratings(1) = 5;
% my_ratings(14) = 2;
% my_ratings(22) = 4;
% my_ratings(28) = 5;
% my_ratings(29) = 4;
% my_ratings(50) = 5;
% my_ratings(54) = 2;
% my_ratings(56) = 4;
% my_ratings(63) = 3;
% my_ratings(64) = 5;
% my_ratings(69) = 4;
% my_ratings(71) = 5;
% my_ratings(72) = 3;
% my_ratings(82) = 5;
% my_ratings(91) = 5;
% my_ratings(95) = 5;
% my_ratings(96) = 4;
% my_ratings(98) = 1;
% my_ratings(99) = 4;
% my_ratings(121) = 5;
% my_ratings(132) = 4;
% my_ratings(133) = 3;
% my_ratings(155) = 3;
% my_ratings(172) = 5;
% my_ratings(173) = 5;
% my_ratings(174) = 5;
% my_ratings(176) = 2;
% my_ratings(181) = 4;
% my_ratings(183) = 2;
% my_ratings(195) = 4;
% my_ratings(200) = 3;
% my_ratings(204) = 5;
% my_ratings(219) = 1;
% my_ratings(234) = 4;
% my_ratings(313) = 4;
% my_ratings(352) = 4;
% my_ratings(420) = 4;
% my_ratings(423) = 3;
% my_ratings(538) = 5;
% my_ratings(588) = 5;

fprintf('\n\nNew user ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), movieList{i});
    end
end

%  Load data
load('ex_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 943 users
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);
initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj','on','MaxIter',100);

% Set Regularization
lambda = 10;
theta = fmincg(@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, num_features,lambda)), initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), num_users, num_features);

% Predict ratings
p = X * Theta';
my_predictions = p(:,1) + Ymean;

movieList = loadMovieList();

[r, ix] = sort(my_predictions,'descend');
for i=1:25
    j = ix(i);
    if i == 1
        fprintf('\nTop recommendations for you:\n');
    end
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), movieList{j});
end


