global Xtrain
global Ytrain
global Ntrain
global sigma_n

data = load('../../data2017/data2010.mat');
x = data.data2010.Score;
y = data.data2010.TargetScore1;

IDtest = (y==-1);
Xtest = x(IDtest, :);
Ytest = y(IDtest, :);
IDtrain = (y~=-1);
X = x(IDtrain, :);
Y = y(IDtrain, :);

N = size(Y, 1);
Nval = floor(N/10);
Ntrain = N-Nval;
Ntest = size(Ytest, 1);
sigma_n = 1;

for t = 0
    start = t*Nval+1;
    if N-start < Nval
        break;
    end
    Xtrain = [X(1:start-1,:); X(start+Nval:end,:)];
    Ytrain = [Y(1:start-1,:); Y(start+Nval:end,:)];
    Xval = X(start:start+Nval-1, :);
    Yval = Y(start:start+Nval-1, :);
    
    S = 0;
    L = 0;
    mse = 9999999;
    for i = 0:50
        theta0 = rand(2, 1)*5;
        result = fminunc(@Func, theta0);
        sigma = result(1);
        l = result(2);

        [K, dK1, dK2] = GetK(result, Xtrain, Ntrain, sigma_n);
        Kinv = inv(K);
        Ypred = zeros(Nval, 1);
        for m = 1:Nval
            xstar = Xval(m, :);
            Kstar = zeros(1, Ntrain);
            for n = 1:Ntrain
                xx = Xtrain(n, :);
                Kstar(n) = Kse(xx, xstar, sigma, l);
            end
            Kstar2 = Kse(xstar, xstar, sigma, l);
            Ypred(m) = normrnd(Kstar*Kinv*Ytrain, Kstar2-Kstar*Kinv*Kstar');
        end

        loss = sum((Ypred-Yval).^2)/Nval;
        
        if mse>loss
            S = sigma;
            L = l;
            mse = loss;
        end
        disp(mse)
    end
end


function output = Kse(x1, x2, sigma, l)
    output = sigma^2*exp(-sum((x1-x2).^2)/(2*l^2));
end

function output = dKse(x1, x2, sigma, l)
    output = zeros(2, 1);
    output(1) = 2*sigma*exp(-sum((x1-x2).^2)/(2*l^2));
    output(2) = sigma^2*exp(-sum((x1-x2).^2)/(2*l^2)) * sum((x1-x2).^2)/l^3;
end

function [K, dK1, dK2] = GetK(theta, X, N, sigma_n)
    sigma = theta(1);
    l = theta(2);
    K = zeros(N, N);
    dK1 = zeros(N, N);
    dK2 = zeros(N, N);
    for i = 1:N
        x1 = X(i, :);
        for j = 1:N
             x2 = X(j, :);
             K(i, j) = Kse(x1, x2, sigma, l);
             temp = dKse(x1, x2, sigma, l);
             dK1(i, j) = temp(1);
             dK2(i, j) = temp(2);
        end
    end
    K = K + sigma_n*ones(N, N);
end

function [MLE, dMLE] = Func(theta)
    global Xtrain
    global Ytrain
    global Ntrain
    global sigma_n

    [K, dK1, dK2] = GetK(theta, Xtrain, Ntrain, sigma_n);
    Kinv = inv(K);
    alpha = Kinv*Ytrain;

    MLE = 0.5*Ytrain'*Kinv*Ytrain+0.5*log(det(K))+Ntrain/2*log(2*pi);
    dMLE = zeros(2, 1);
    dMLE(1) = -0.5*trace((alpha*alpha'-Kinv)*dK1);
    dMLE(2) = -0.5*trace((alpha*alpha'-Kinv)*dK2);
end