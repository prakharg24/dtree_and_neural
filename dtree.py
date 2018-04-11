
# coding: utf-8

# In[88]:


from __future__ import print_function
import time,sys,statistics,csv
import numpy as np
import math
import matplotlib.pyplot as plt
import copy


# In[92]:


## The possible attributes in the data with the prediction at index 0. Smaller names for brevity.
attributes = ["rich","age","wc","fnlwgt","edu","edun","mar","occ","rel","race","sex","capg","canpl","hpw","nc"]

## Get the encoding of the csv file by replacing each categorical attribute value by its index.
wc_l = "Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked".split(", ")
edu_l = "Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool".split(", ")
mar_l = "Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse".split(", ")
occ_l = "Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces".split(", ")
rel_l = "Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried".split(", ")
race_l = "White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black".split(", ")
sex_l = "Female, Male".split(", ")
nc_l = "United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands".split(", ")
encode = {
    "rich"   : {"0":0,"1":1},
    "wc"     : {wc_l[i]:i for i in range(len(wc_l))},
    "edu"    : {edu_l[i]:i for i in range(len(edu_l))},
    "mar"    : {mar_l[i]:i for i in range(len(mar_l))},
    "occ"    : {occ_l[i]:i for i in range(len(occ_l))},
    "rel"    : {rel_l[i]:i for i in range(len(rel_l))},
    "race"   : {race_l[i]:i for i in range(len(race_l))},
    "sex"    : {sex_l[i]:i for i in range(len(sex_l))},
    "nc"     : {nc_l[i]:i for i in range(len(nc_l))},
    }

attr_len = [2, 2, len(wc_l), 2, len(edu_l), 2, len(mar_l), len(occ_l), len(rel_l), len(race_l), len(sex_l), 2, 2, 2, len(nc_l)]

def medians(file):
    """
    Given a csv file, find the medians of the categorical attributes for the whole data.
    params(1): 
        file : string : the name of the file
    outputs(6):
        median values for the categorical columns
    """
    fin = open(file,"r")
    reader = csv.reader(fin)
    age, fnlwgt, edun, capg, capl, hpw = ([] for i in range(6))
    total = 0
    for row in reader:
        total+=1
        if(total==1):
            continue
        l = [x.lstrip().rstrip() for x in row]
        age.append(int(l[0]));
        fnlwgt.append(int(l[2]));
        edun.append(int(l[4]));
        capg.append(int(l[10]));
        capl.append(int(l[11]));
        hpw.append(int(l[12]));
    fin.close()
    return(statistics.median(age),statistics.median(fnlwgt),statistics.median(edun),statistics.median(capg),statistics.median(capl),statistics.median(hpw))

def preprocess(file, median):
    """
    Given a file, read its data by encoding categorical attributes and binarising continuos attributes based on median.
    params(1): 
        file : string : the name of the file
    outputs(6):
        2D numpy array with the data
    """
    # Calculate the medians
    agem,fnlwgtm,edunm,capgm,caplm,hpwm = medians(file)
    fin = open(file,"r")
    reader = csv.reader(fin)
    data = []
    total = 0
    for row in reader:
        total+=1
        # Skip line 0 in the file
        if(total==1):
            continue
        l = [x.lstrip().rstrip() for x in row]
        t = [0 for i in range(15)]
        
        # Encode the categorical attributes
        t[0] = encode["rich"][l[-1]]; t[2] = encode["wc"][l[1]]; t[4] = encode["edu"][l[3]]
        t[6] = encode["mar"][l[5]]; t[7] = encode["occ"][l[6]]; t[8] = encode["rel"][l[7]]
        t[9] = encode["race"][l[8]]; t[10] = encode["sex"][l[9]]; t[14] = encode["nc"][l[13]]
        
        # Binarize the numerical attributes based on median.
        # Modify this section to read the file in part c where you split the continuos attributes baed on dynamic median values.
        if(median):
            t[1] = float(l[0])>=agem; t[3] = float(l[2])>=fnlwgtm; t[5] = float(l[4])>=edunm;
            t[11] = float(l[10])>=capgm; t[12] = float(l[11])>=caplm; t[13] = float(l[12])>=hpwm;
        else:
            t[1] = l[0]; t[3] = l[2]; t[5] = l[4];
            t[11] = l[10]; t[12] = l[11]; t[13] = l[12];
        
        # Convert some of the booleans to ints
        data.append([int(x) for x in t])
    
    return np.array(data,dtype=np.int64)


## Read the data
train_data = preprocess("dtree_data/train.csv", False)
valid_data = preprocess("dtree_data/valid.csv", False)
test_data = preprocess("dtree_data/test.csv", False)

print("The sizes are ","Train:",len(train_data),", Validation:",len(valid_data),", Test:",len(test_data))


# In[101]:


def is_median(ind):
    if(ind==1 or ind==3 or ind==5 or ind==11 or ind==12 or ind==13):
        return True
    else:
        return False

def predict(inp_pt, dtree_ind):
    global dtree
    attr_ind = dtree[dtree_ind][0]
    if(dtree[dtree_ind][1][inp_pt[attr_ind]]==-1):
        return 1
    elif(dtree[dtree_ind][1][inp_pt[attr_ind]]==-2):
        return 0
    else:
        return predict(inp_pt, dtree[dtree_ind][1][inp_pt[attr_ind]])

def get_acc(test_data):
    corr = 0
    total = 0
    for data_pt in test_data:
        my_pred = predict(data_pt, 1)
        if(my_pred==data_pt[0]):
            corr += 1
        total += 1

    return (corr+0.0)/total

def calc_gain(new_inp, attr_ind):
    rich = [0 for i in range(attr_len[attr_ind])]
    poor = [0 for i in range(attr_len[attr_ind])]
    num = 0
    den = 0
    med_arr = []
    inp = copy.deepcopy(new_inp)
    median = 0.0
    if(is_median(attr_ind)):
        for data_pt in inp:
            med_arr.append(int(data_pt[attr_ind]))
        median = statistics.median(med_arr)
        new_data = []
        for data_pt in inp:
            new_data.append(int(float(data_pt[attr_ind])>=median))
        for data_pt, new_dp in zip(inp, new_data):
            data_pt[attr_ind] = new_dp
    for data_pt in inp:
        if(data_pt[0]==0):
            poor[data_pt[attr_ind]] += 1
        else:
            rich[data_pt[attr_ind]] += 1
    for i in range(attr_len[attr_ind]):
        if(rich[i]!=0 and poor[i]!=0):
            p_x = (rich[i] + 0.0)/(rich[i] + poor[i])
            num += (rich[i] + poor[i])*(p_x*math.log(p_x) + (1-p_x)*math.log(1-p_x))
            den += (rich[i] + poor[i])
    if(den==0):
        return 0, rich, poor, median
    else:
        return (num/den), rich, poor, median

def max_gain(new_inp):
    max_gn = float('-inf')
    max_ind = -1
    max_rich = []
    max_poor = []
    max_med = 0.0
    rich = 0
    poor = 0
    inp = copy.deepcopy(new_inp)
    for i in range(len(inp)):
        if(inp[i][0]==0):
            poor += 1
        else:
            rich += 1
    if(rich==0 or poor==0):
        return 0.0, max_ind, max_rich, max_poor, max_med
    p_x = (rich + 0.0)/(rich + poor)
    entropy = p_x*math.log(p_x) + (1-p_x)*math.log(1-p_x)
    for attr_ind in range(1, len(attributes)):
        attr_gain, attr_rich, attr_poor, attr_med = calc_gain(inp, attr_ind)
        if(attr_gain>max_gn):
            max_gn, max_ind, max_rich, max_poor, max_med = attr_gain, attr_ind, attr_rich, attr_poor, attr_med

    return (-1*entropy + max_gn), max_ind, max_rich, max_poor, max_med

def remove_node(ind, mq, iq, aq):
    global dtree
    ans = 1
    dtree[iq[ind-1]][1][aq[ind-1]] = dtree[ind][2]
    for i in range(len(dtree[ind][1])):
        if(dtree[ind][1][i]>0):
            ans += remove_node(dtree[ind][1][i], mq, iq, aq)
    dtree[ind] = (dtree[ind][0], dtree[ind][1], dtree[ind][2], 0)
    return ans

def calc_prof(i, inp):
    global dtree
    if(dtree[i][3]==0):
        return 0
    if(dtree[i][2]==-1):
        my_pred = 1
    else:
        my_pred = 0

    new_corr = 0
    old_corr = 0
    for data_pt in inp:
        if(data_pt[0]==my_pred):
            new_corr += 1
        if(data_pt[0]==predict(data_pt, i)):
            old_corr += 1
    return (new_corr - old_corr)

def plot_node_acc(data_set, filename):
    global dtree
    fnl_dec = []
    for data_pt in data_set:
        temp_arr = []
        ite = 1
        while(ite<len(dtree)):
            temp_arr.append((ite, dtree[ite][2]))
            if(is_median(dtree[ite][0])):
                temp_val = int(float(data_pt[dtree[ite][0]])>=dtree[ite][4])
            else:
                temp_val = data_pt[dtree[ite][0]]
            if(dtree[ite][1][temp_val]==-1):
                temp_arr.append((ite+1, -1))
                break
            elif(dtree[ite][1][temp_val]==-2):
                temp_arr.append((ite+1, -2))
                break
            else:
                ite = dtree[ite][1][temp_val]
        fnl_arr = []
        it = 0
        for i in range(0, len(dtree)-1):
            if(i<temp_arr[it][0]):
                fnl_arr.append(temp_arr[it][1])
            else:
                it += 1
                if(it==len(temp_arr)):
                    it -= 1
                fnl_arr.append(temp_arr[it][1])
        corr_pred = []
        for i in range(0, len(dtree) - 1):
            if(data_pt[0] - fnl_arr[i]==2):
                corr_pred.append(1)
            else:
                corr_pred.append(0)
        fnl_dec.append(corr_pred)

    fnl_dec = np.sum(fnl_dec, axis = 0)
    fnl_dec = fnl_dec/len(data_set)
    fnl_dec = fnl_dec*100
    print(fnl_dec[-1])
    plt.plot(fnl_dec)
    plt.savefig(filename)
    plt.close()

def fnl_fnc(train_data, valid_data, test_data, prune):
    global dtree
    ite = 0
    main_q = [train_data]
#     valid_q = [valid_data]
    ind_q = [0]
    attr_q = [0]
    curr_acc = 0
    while(ite<len(main_q)):
        inp = main_q[ite]
        prev_ind = ind_q[ite]
        attr_val = attr_q[ite]
        gn, ind, rich, poor, med = max_gain(inp)
        if(gn<=1e-15):
            main_q = main_q[:ite] + main_q[(ite+1):]
#             valid_q = valid_q[:ite] + valid_q[(ite+1):]
            ind_q = ind_q[:ite] + ind_q[(ite+1):]
            attr_q = attr_q[:ite] + attr_q[(ite+1):]
            continue
        my_ind = len(dtree)
        temp_val = dtree[prev_ind][1][attr_val]
        dtree[prev_ind][1][attr_val] = my_ind
        temp_lst = []
        for i in range(len(rich)):
            if(rich[i]>poor[i]):
                temp_lst.append(-1)
            else:
                temp_lst.append(-2)
        dtree.append((ind, temp_lst, temp_val, 1, med))

        new_inp = [[] for i in range(attr_len[ind])]
        for data_pt in inp:
            if(is_median(ind)):
                temp_val = int(float(data_pt[ind])>=med)
            else:
                temp_val = data_pt[ind]
            new_inp[temp_val].append(data_pt)

#         valid_inp = [[] for i in range(attr_len[ind])]
#         for data_pt in valid_q[ite]:
#             valid_inp[data_pt[ind]].append(data_pt)

        for i in range(attr_len[ind]):
            main_q.append(new_inp[i])
#             valid_q.append(valid_inp[i])
            ind_q.append(my_ind)
            attr_q.append(i)

        ite += 1

#     if(prune):
#         X = []
#         Y = []
#         num_nodes = len(dtree)-1
#         max_nodes = num_nodes
#         while(True):
#             max_prof = float('-inf')
#             max_ind = -1
#             for i in range(1, len(dtree)):
#                 this_prof = calc_prof(i, valid_q[i-1])
#                 if(this_prof>0 and this_prof>max_prof):
#                     max_prof = this_prof
#                     max_ind = i
#             if(max_ind==-1):
#                 break
#             print(max_prof)
#             X.append(num_nodes)
#             Y.append(get_acc(valid_data))
#             rem_node = remove_node(max_ind, valid_q, ind_q, attr_q)
#             num_nodes = num_nodes - rem_node
# #             print(X, Y)
#         X.append(num_nodes)
#         Y.append(get_acc(valid_data))
#         print(get_acc(valid_data))
#         plt.plot(X, Y)
#         plt.xlim(max_nodes+100, num_nodes-100)
#         plt.savefig("test_prune.png")
#         plt.close()
#         print(num_nodes)



#     plot_node_acc(valid_data, "valid_data.png")
#     print("valid done")
#     plot_node_acc(test_data, "test_data.png")
#     print("test done")
#     plot_node_acc(train_data, "train_data.png")
#     print("train done")


    plot_node_acc(valid_data, "valid_data_med.png")
    print("valid done")
    plot_node_acc(test_data, "test_data_med.png")
    print("test done")
    plot_node_acc(train_data, "train_data_med.png")
    print("train done")


# In[102]:


rich = 0
poor = 0
for data_pt in train_data:
    if(data_pt[0]==0):
        poor += 1
    else:
        rich += 1
dtree = []
if (rich>poor):
    dtree.append((-1, [-1], -1, 0))
else:
    dtree.append((-1, [-2], -2, 0))
fnl_fnc(train_data, valid_data, test_data, False)


# In[111]:


def traversal(dtree_ind, my_ind):
    global dtree
    attr_ind = dtree[dtree_ind][0]
    max_lst = []
    max_height = 0
    for i in range(len(dtree[dtree_ind][1])):
        if(dtree[dtree_ind][1][i]>0):
            temp_lst, height = traversal(dtree[dtree_ind][1][i], my_ind)
            if(height>max_height):
                max_lst, max_height = temp_lst, height
    if(attr_ind==my_ind):
        max_lst.append(dtree_ind)
    return max_lst, len(max_lst)

fnl_lst, leng = traversal(1, 13)
print(leng)
if(leng>1):
    for ele in reversed(fnl_lst):
        print(dtree[ele][4])


# In[79]:


def arr(c):
    a = c
    a.append(1)
    return a

a = [1, 2, 3]
b = arr(a)
print(a)


# In[113]:


a = [1, 2]
np.diag(a)


# In[298]:


from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=200, min_samples_leaf=1, max_depth=20)
X = []
Y = []
for data_pt in train_data:
    Y.append(data_pt[0])
    X.append(data_pt[1:])
clf = clf.fit(X, Y)
X = []
Y = []
for data_pt in train_data:
    Y.append(data_pt[0])
    X.append(data_pt[1:])
arr = clf.predict(X)

count = 0
total = 0
for ele, ele2 in zip(arr, Y):
    if(ele==ele2):
        count += 1
    total += 1
print((count+0.0)/total)


# In[295]:


from sklearn import ensemble
clf = ensemble.RandomForestClassifier(n_estimators = 100, max_features = 4, bootstrap=True)
X = []
Y = []
for data_pt in train_data:
    Y.append(data_pt[0])
    X.append(data_pt[1:])
clf = clf.fit(X, Y)

X = []
Y = []
for data_pt in train_data:
    Y.append(data_pt[0])
    X.append(data_pt[1:])
arr = clf.predict(X)

count = 0
total = 0
for ele, ele2 in zip(arr, Y):
    if(ele==ele2):
        count += 1
    total += 1

print((count+0.0)/total)

