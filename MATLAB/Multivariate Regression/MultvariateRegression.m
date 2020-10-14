data = load('ex1data2.txt');    %Getting data from the file
X = data(:, 1:2);   %Extracting features to predict price
y = data(:, 3); %Extracting lists of prices
m = length(y);  %Number of data elements
mu = mean(X);   %Mean of each column of X
sigma = std(X); %standard deviation of each column of X


X = featureNormalize(X); %All elements of the matrix are normalized

X = [ones(m, 1) X]; %adding bias term


alpha = 0.1;    %choosing alpha term
num_iters = 2000;    %choosing iterations for algorithm

theta = zeros(3, 1);    %setting value of theta
[theta, ~] = gradientDescentMulti(X, y, theta, alpha, num_iters);   %calling gradient descent function to set theta

% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f\n%f\n%f',theta(1),theta(2),theta(3)); %Value of theta displayed



X1 = ([1852 4] - mu) ./ sigma;  % Setting matrix with normalized data
X1 = [1 X1];    % Adding bias term
price = X1 * theta;
fprintf('\nPredicted price of house:\n%f',price);


function [X_norm, mu, sigma] = featureNormalize(X)
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
mu = mean(X);
sigma = std(X);

for i = 1:size(X,2)
    X_norm(:,i:i) = ((X(:,i:i) - mu(:,i:i))/sigma(:,i:i));
end

end

function J = computeCostMulti(X, y, theta)
m = length(y);

J = 0;

h = X * theta;
Jtemp= (h - y).^2;

J = sum(Jtemp) * (1/(2*m));

end

function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
computeCostMulti(X, y, theta);
for iter = 1:num_iters
    h = X * theta;
    hmy = (h-y);
    val = zeros(1,size(X,2));
    for i = 1:size(X,2)
        hmy(:,i:i) = (h-y) .* X(:,i:i);
    
        val(:,i:i) = sum(hmy(:,i)) * (1/m);
    end
    for j = 1:size(X,2)
        theta(j) = (theta(j) -(val(:,j:j) .* alpha)); 
    end

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
end