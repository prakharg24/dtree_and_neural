
# coding: utf-8

# In[502]:


import numpy as np
import copy
import csv
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
import math
def plot_decision_boundary(model, X, y, filename):
    """
    Given a model(a function) and a set of points(X), corresponding labels(y), scatter the points in X with color coding
    according to y. Also use the model to predict the label at grid points to get the region for each label, and thus the 
    descion boundary.
    Example usage:
    say we have a function predict(x,other params) which makes 0/1 prediction for point x and we want to plot
    train set then call as:
    plot_decision_boundary(lambda x:predict(x,other params),X_train,Y_train)
    params(3): 
        model : a function which expectes the point to make 0/1 label prediction
        X : a (mx2) numpy array with the points
        y : a (mx1) numpy array with labels
    outputs(None)
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.savefig(filename)
    plt.close()


# In[527]:


def csv_reader(file_obj):
    reader = csv.reader(file_obj)
    final_ans = []
    for row in reader:
        temp_ele = []
        for ele in row:
            temp_ele.append(float(ele))
        final_ans.append(temp_ele)
    return final_ans

def mnist_reader(file_obj):
    reader = csv.reader(file_obj)
    final_ans = []
    final_y = []
    for row in reader:
#         print("Row num", row[-1])
        if(float(row[-1])==6):
#             print("yeah")
            final_y.append(0.0)
        else:
            final_y.append(1.0)
        temp_ele = []
        for ele in row:
            temp_ele.append(float(ele))
        final_ans.append(temp_ele[:-1])
    return final_ans, final_y

def csv_reader_y(file_obj):
    reader = csv.reader(file_obj)
    final_ans = []
    for row in reader:
        for ele in row:
            final_ans.append(float(ele))
    return final_ans


# In[556]:


def sigmoid(x):
#     print("hello")
#     print(x)
    return (1.0/(1.0 + np.exp(-x)))
#     print("bye")

def fprop(inp, lay_w, lay_b):
    outpts = []
    outpts.append(np.reshape(inp, [-1]))
    for (wei, bia) in zip(lay_w, lay_b):
        prop = np.reshape(outpts[-1], [1, -1])
        prop = np.add(np.dot(prop, wei),bia)
        outpts.append(sigmoid(np.reshape(prop, [-1])))
    return outpts

def bprop(y_exp, lay_w, lay_b, outpts):
    bia_up = []
    wei_up = []
    last_delta = [-1*(y_exp - outpts[-1][0])]
    for (wei, bia, out, out_prev) in zip(reversed(lay_w), reversed(lay_b), reversed(outpts), reversed(outpts[:-1])):
        bia_up.append(np.reshape(last_delta, [1, -1]))
        out_temp = np.reshape(out_prev, [-1, 1])
        total_deriv = np.dot(out_temp, np.reshape(last_delta, [1, -1]))
        wei_up.append(total_deriv)
        sig_deriv = np.dot(np.diag(out_prev), np.diag(1 - out_prev))
        no_oj = np.dot(wei, np.reshape(last_delta, [-1, 1]))
        last_delta = np.dot(sig_deriv, no_oj)
        last_delta = np.reshape(last_delta, [-1])
    return wei_up[::-1], bia_up[::-1]

def neural_train(inp, out_exp, hidden_lay, l_rate, batch, max_iter):
    wei_up = []
    bia_up = []
    lay_w = []
    lay_b = []
    fir = len(inp[0])
    for ele in hidden_lay:
#         lay_w.append(np.random.rand(fir, ele))
        lay_w.append((np.random.rand(fir, ele)-0.5)*10)
#         lay_b.append(np.random.rand(1, ele))
        lay_b.append((np.random.rand(1, ele)-0.5)*10)
        wei_up.append(np.zeros((fir, ele)))
        bia_up.append(np.zeros((1, ele)))
        fir = ele
#     lay_w.append(np.random.rand(fir, 1))
    lay_w.append((np.random.rand(fir, 1)-0.5)*10)
#     lay_b.append(np.random.rand(1, 1))
    lay_b.append((np.random.rand(1, 1)-0.5)*10)
    wei_up.append(np.zeros((fir, 1)))
    bia_up.append(np.zeros((1, 1)))
    err_sum = 0.0
    prev_err_sum = 0.0
    total_err_sum = 0.0
    epoch = 0
    while(True):
        if(epoch>max_iter):
            break
        l_rate = 0.001/(math.sqrt(epoch+1))
        print("Epoch", epoch)
        if(epoch%10==0):
            print(epoch, total_err_sum)
            total_err_sum = 0.0
        for i in range(0, len(inp)):
            if(i%1000==0):
                print(i)
            outputs = fprop(inp[i], lay_w, lay_b)
            error = (out_exp[i] - outputs[-1][0])**2
            err_sum += error
            wup_tmp, bup_tmp = bprop(out_exp[i], lay_w, lay_b, outputs)
            for (wei, bia, wup, bup) in zip(wei_up, bia_up, wup_tmp, bup_tmp):
                wei += wup
                bia += bup
            if((i+1)%batch==0):
#                 print(wei_up)
#                 print(bia_up)
                for j in range(len(lay_w)):
                    lay_w[j] -= l_rate*wei_up[j]
                    lay_b[j] -= l_rate*bia_up[j]
                    wei_up[j] = np.zeros(np.shape(wei_up[j]))
                    bia_up[j] = np.zeros(np.shape(bia_up[j]))
        
        epoch += 1
        diff = prev_err_sum - err_sum
        if(diff < 0.001 and diff >=0):
            break
        prev_err_sum = err_sum
        total_err_sum += err_sum
        err_sum = 0.0
    return lay_w, lay_b

def fprop_new(inp, lay_w, lay_b):
    outpts = []
    outpts.append(inp)
    for (wei, bia) in zip(lay_w, lay_b):
        prop = outpts[-1]
        prop = np.add(np.dot(prop, wei),bia)
        outpts.append(sigmoid(prop))
    return outpts


def neural_predict(inp):
    global lw, lb
    outputs = fprop_new(inp, lw, lb)
#     print(np.shape(outputs[-1]))
    arr = []
    for i in range(len(outputs[-1])):
        if(outputs[-1][i][0]>=0.5):
            arr.append([1])
        else:
            arr.append([0])
    return np.array(arr)


# In[450]:


with open("toy_data/toy_trainX.csv", "r") as f_obj:
    train_inp = csv_reader(f_obj)
with open("toy_data/toy_trainY.csv", "r") as f_obj:
    train_out = csv_reader_y(f_obj)


# In[486]:


hidden_lay = [5, 5]
lw, lb = neural_train(train_inp, train_out, hidden_lay, 0.001, len(inp))


# In[487]:


with open("toy_data/toy_testX.csv", "r") as f_obj:
    inp_test = csv_reader(f_obj)
with open("toy_data/toy_testY.csv", "r") as f_obj:
    out_exp_test = csv_reader_y(f_obj)


    
corr = 0
total = 0
for data_pt, corr_pred in zip(train_inp, train_out):
#     print(corr_pred[0], neural_predict(data_pt))
    if(corr_pred == neural_predict(data_pt)):
        corr += 1
    total += 1

print((corr+0.0)/total)

corr = 0
total = 0
for data_pt, corr_pred in zip(inp_test, out_exp_test):
#     print(corr_pred[0], neural_predict(data_pt))
    if(corr_pred == neural_predict(data_pt)):
        corr += 1
    total += 1

print((corr+0.0)/total)

# loglin = linear_model.LogisticRegression()

# loglin.fit(train_inp, train_out)

# total = 0
# corr = 0
# one = 0
# zero = 0
# ans = loglin.predict(inp_test)
# print(metrics.accuracy_score(ans, out_exp_test))
# ans = loglin.predict(train_inp)
# print(metrics.accuracy_score(ans, train_out))

# plot_decision_boundary(lambda x:loglin.predict(x), np.array(inp_test), np.array(out_exp_test), "lin_reg_test.png")
# plot_decision_boundary(lambda x:loglin.predict(x), np.array(train_inp), np.array(train_out), "lin_reg_train.png")

plot_decision_boundary(lambda x:neural_predict(x), np.array(inp_test), np.array(out_exp_test), "neural_test_5_5.png")
# plot_decision_boundary(lambda x:neural_predict(x), np.array(train_inp), np.array(train_out), "neural_train_5.png")


# In[547]:


with open("mnist_data/MNIST_train.csv", "r") as f_obj:
    train_inp, train_out = mnist_reader(f_obj)
with open("mnist_data/MNIST_test.csv", "r") as f_obj:
    inp_test, out_exp_test = mnist_reader(f_obj)
# print(train_out)


# In[557]:


hidden_lay = [100]
print(len(train_inp))
lw, lb = neural_train(train_inp[1750:1850], train_out[1750:1850], hidden_lay, 0.001, 100, 200)


# In[560]:


corr = 0
total = 0
for data_pt, corr_pred in zip(inp_test, out_exp_test):
#     print(neural_predict(data_pt))
    if(corr_pred == neural_predict(data_pt)):
        corr += 1
    total += 1

print(corr)
print((corr+0.0)/total)

