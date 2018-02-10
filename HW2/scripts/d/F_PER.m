function [output, grad] = F_PER(x, y, para)

sigma = para(1);
l = para(2);
sigma_n = para(3);
p = para(4);
n = length(y);

K = zeros(n, n);
dS = zeros(n, n);
dL = zeros(n, n);
dP = zeros(n, n);
dSn = zeros(n, n);
grad = zeros(4, 1);

for i = 1:n
    x1 = x(i, :);
    for j = 1:n
        x2 = x(j, :);
        K(i, j) = sigma^2*exp(-2*(sin(pi*sqrt(sum((x1-x2).^2))/p))^2/l^2);
        dS(i, j) = 2*sigma*exp(-2*(sin(pi*sqrt(sum((x1-x2).^2))/p))^2/l^2);
        dL(i, j) = 4*K(i, j)*(sin(pi*sqrt(sum((x1-x2).^2))/p))^2/l^3;
        dP(i, j) = K(i, j)*sin(2*pi*sqrt(sum((x1-x2).^2))/p)*pi*sqrt(sum((x1-x2).^2))/(l^2*p^2);
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
grad(4) = trace((alpha*alpha'-K_inv)*dP);  
    
end

