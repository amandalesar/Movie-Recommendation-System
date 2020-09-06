function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


J = (1/2) * sum(sum(R .* (X * Theta' - Y).^2)) + (lambda/2)*sum(sum(Theta.^2)) + (lambda/2)*sum(sum(X.^2));

%x = feature, goes with num movies
%theta = parameters, goes with num users
%only need to do movies that were actually rated

for i = 1:num_movies
    idx = find(R(i,:)==1);
    Theta_temp = Theta(idx,:);
    Y_temp = Y(i,idx); 
    X_grad(i,:) = (X(i,:) * Theta_temp' - Y_temp) * Theta_temp + lambda*X(i,:);
end

for i = 1:num_users
    idx = find(R(:,i)==1);
    X_temp = X(idx,:); 
    Y_temp = Y(idx,i); 
    Theta_grad(i,:) = (X_temp * Theta(i,:)' - Y_temp)' * X_temp + lambda*Theta(i,:);
end


grad = [X_grad(:); Theta_grad(:)];

end
