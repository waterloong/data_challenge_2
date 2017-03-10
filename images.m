for i = 1:size(X_train, 2),
    imwrite(reshape(X_train(:,i), 110, 110),strcat(num2str(i), '.png'));
end
for i = 1:size(X_test, 2),
    imwrite(reshape(X_train(:,i), 110, 110),strcat(num2str(i), '.png'));
end