data = load('ex1data1.txt'); %loading data
X = data(:,1); %Setting X as 1st column (House Price)
Y = data(:,2); %Setting Y as 2nd column (Profit)
plot(X,Y,'bo', 'MarkerSize', 3); %plotting the datapoints
ylabel('Profit in $10,000s'); % Set the y-axis label
xlabel('Population of City in 10,000s'); % Set the x-axis label

m = length(X); %number of data element rows
X = [ones(m,1),data(:,1)]; %Setting the X matrix. 1'st column is ones and 2nd column is data (House Price)
theta = zeros(2, 1); %Setting the theta matrix
iterations = 2000; %Setting the number of iterations to run
alpha = 0.01; %Setting learning rate

computeCost(X, Y,[-1; 2]); %Calling cost function
theta = gradientDescent(X, Y, theta, alpha, iterations); %Calling the gradient descent function

hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure


% Visualizing J(theta_0, theta_1):
% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, Y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';

% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;

function J = computeCost(X, Y, theta)
    m = length(X);
    J = 0;
    h = X * theta;
    J = (1/(2*m)* sum((h-Y).^2)); %Equation for cost function
end

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1); %setting the history of cost function matrix to zero for all iterations

theta1 = theta(1); %Setting value of theta1 as value of 1st row of theta
theta2 = theta(2); %Setting value of theta2 as value of 2nd row of theta

for iter = 1:num_iters %For loop to desired iterations of the gradient descent function
    h = X * theta;  %Hypothesis
    temp0 = 0;
    temp1 = 0;
    for i=1:m
       error = (h(i) - y(i));
       temp0 = temp0 + (error * X(i, 1));
       temp1 = temp1 + (error * X(i, 2)); 
    end
    theta1 = theta1 - ((alpha/m) * temp0);
    theta2 = theta2 - ((alpha/m) * temp1);
    theta = [theta1;theta2];
    J_history(iter) = computeCost(X, y, theta);
end

end
