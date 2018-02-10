function [output, grad] = F_SE(x, y, para)

sigma = para(1);
l = para(2);
sigma_n = para(3);
n = length(y);

K = zeros(n, n);
dS = zeros(n, n);
dL = zeros(n, n);
dSn = zeros(n, n);
grad = zeros(3, 1);

for i = 1:n
    x1 = x(i, :);
    for j = 1:n
        x2 = x(j, :);
        K(i, j) = sigma^2*exp(-(sum((x1-x2).^2))/(2*l^2));
        dS(i, j) = 2*sigma*exp(-(sum((x1-x2).^2))/(2*l^2));
        dL(i, j) = K(i, j)*sum((x1-x2).^2)/l^3;
        if i==j
            K(i, j) = K(i, j)+sigma_n^2;
            dSn(i, j) = 2*sigma_n;
        end
    end 
end

K_inv = eye(n)/K;
alpha = K_inv*y;
output = y'*alpha+log(det(K))+size(x, 2)*log(2*pi);
grad(1) = trace((alpha*alpha'-K_inv)*dS);  
grad(2) = trace((alpha*alpha'-K_inv)*dL);  
grad(3) = trace((alpha*alpha'-K_inv)*dSn);  
    
end

