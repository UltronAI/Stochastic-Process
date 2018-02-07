function loss = mse(y, y_pred)

loss = sum((y-y_pred).^2)/n

end

