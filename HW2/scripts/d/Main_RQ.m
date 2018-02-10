close all;
clear all;
clc;

result = [];
LOSS = zeros(5,10);
year = [2010, 2011, 2012, 2013, 2014];
for order = 1:5
    year_cur = year(order);
    data_name = sprintf('data%d', year_cur);
    data_path = sprintf('../../data2017/%s.mat', data_name);
    data = load(data_path);
    
    switch year_cur
        case 2010
            x = data.data2010.Score/100;
            y = data.data2010.TargetScore1;
        case 2011
            x = data.data2011.Score/100;
            y = data.data2011.TargetScore1;
        case 2012
            x = data.data2012.Score/100;
            y = data.data2012.TargetScore1;
        case 2013
            x = data.data2013.Score/100;
            y = data.data2013.TargetScore1;
        otherwise
            x = data.data2014.Score/100;
            y = data.data2014.TargetScore1;
    end
    

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
    d = size(x, 2);

    para_best = [];
    loss_min = +inf;
    s_id = 0;
    K_inv_best = [];

    for s = 0:9
        start = s*Nval+1;
        Xtrain = [X(1:start-1, :); X(start+Nval:end, :)];
        Ytrain = [Y(1:start-1, :); Y(start+Nval:end, :)];
        Xval = X(start:start+Nval-1, :);
        Yval = Y(start:start+Nval-1, :);

        para_opt = [];
        err_min = +inf;
        K_inv = [];
        for t = 1:10
            para_init=(rand(4,1)-0.5)*2;
            para_init(4) = abs(para_init(4));

            options=optimset('Algorithm','trust-region','GradObj','on');
            try
                para_temp = fminunc(@(para)F_RQ(Xtrain,Ytrain,para),para_init,options);
            catch
                t = t-1;
                continue;
            end

            sigma = para_temp(1);
            l = para_temp(2);
            sigma_n = para_temp(3);
            a = para_temp(4);

            K = zeros(Ntrain);
            for i = 1:Ntrain
                x1 = Xtrain(i, :);
                for j = 1:Ntrain
                    x2 = Xtrain(j, :);
                    K(i, j) = sigma^2*(1+sum((x1-x2).^2)/(2*a*l^2))^(-a);
                    if i==j
                        K(i,j) = K(i,j)+sigma_n^2;
                    end
                end
            end
            K_inv_temp = eye(Ntrain)/K;
            Ypred_val = zeros(Nval, 1);
            for i = 1:Nval
                K_star = zeros(1, Ntrain);
                for j = 1:Ntrain
                    K_star(1, j) = sigma^2*(1+sum((Xval(i,:)-Xtrain(j,:)).^2)/(2*a*l^2))^(-a);
                end
                Ypred_val(i) = normrnd(K_star*K_inv_temp*Ytrain, sigma^2+sigma_n^2-K_star*K_inv_temp*K_star');
            end

            err = sum((Ypred_val-Yval).^2)/Nval;

            if err_min>err
                err_min = err;
                K_inv = K_inv_temp;
                para_opt = para_temp;
            end
        end

        LOSS(order, s+1) = err_min;

        if loss_min>err_min
            loss_min = err_min;
            s_id = s;
            para_best = para_opt;
            K_inv_best = K_inv;
        end
    end

    sigma = para_best(1);
    l = para_best(2);
    sigma_n = para_best(3);

    start = s_id*Nval+1;
    Xtrain = [X(1:start-1, :); X(start+Nval:end, :)];
    Ytrain = [Y(1:start-1, :); Y(start+Nval:end, :)];

    K_inv = K_inv_best;
    Ypred = zeros(Ntest, 1);
    for i = 1:Ntest
        K_star = zeros(1, Ntrain);
        for j = 1:Ntrain
            K_star(1, j) = sigma^2*(1+sum((Xtest(i,:)-Xtrain(j,:)).^2)/(2*a*l^2))^(-a);
        end
        Ypred(i) = normrnd(K_star*K_inv*Ytrain, sigma^2+sigma_n^2-K_star*K_inv*K_star');
    end
    result = [result; Ypred];
end

for i = 1:5
    fprintf('\n*** data%d ***\n', year(i))
    for j = 1:10
        fprintf('#%d\tvalloss = %.3f\n', j, LOSS(i, j));
    end
end

result(result>100)=100;
result(result<0)=0;
save('../../results/LOSS_RQ.mat', 'LOSS');
save('../../results/d_RQ.mat', 'result');