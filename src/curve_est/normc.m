function x=normc(x)
for i=1:size(x,2)
    x(:,i)=x(:,i)/norm(x(:,i));
end
end