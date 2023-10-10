load mnist_all
train0 = double(train0);
train8 = double(train8);
test0 = double(test0);
test8 = double(test8);
train1 = double(train1);
train2 = double(train2);
test1 = double(test1);
test2 = double(test2);
csvwrite('train0.csv',train0);
csvwrite('test0.csv',test0);
csvwrite('train8.csv',train8);
csvwrite('test8.csv',test8);
csvwrite('train1.csv',train1);
csvwrite('test1.csv',test1);
csvwrite('train2.csv',train2);
csvwrite('test2.csv',test2);