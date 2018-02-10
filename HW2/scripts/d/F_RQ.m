function [output, grad] = F_RQ(x, y, para)

sigma = para(1);
l = para(2);
sigma_n = para(3);
a = para(4);
n = length(y);

K = zeros(n, n);
dS = zeros(n, n);
dL = zeros(n, n);
dSn = zeros(n, n);
dA = zeros(n, n);
grad = zeros(4, 1);

for i = 1:n
    x1 = x(i, :);
    for j = 1:n
        x2 = x(j, :);
        xr = sum((x1-x2).^2);
        K(i, j) = sigma^2*(1+xr/(2*a*l^2))^(-a);
        dS(i, j) = 2*sigma*(1+xr/(2*a*l^2))^(-a);
        dL(i, j) = sigma^2*(1+xr/(2*a*l^2))^(-a-1)*xr/(l^3);
        dA(i ,j) = K(i, j)*(xr/(2*a*l^2*(1+xr/(2*a*l^2)))-log(1+xr/(2*a*l^2)));
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
grad(4) = trace((alpha*alpha'-K_inv)*dA); 
    
end

