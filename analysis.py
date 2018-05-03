# encoding:utf-8
# read data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import tree
# 参数初始化
from xgboost import XGBClassifier
import xgboost as xgb
train_filename = "data/train.csv"
test_filename = "data/test.csv"


# 处理训练数据
def del_data(filename):
    # 加载数据
    data = pd.read_csv(filename)
    # 删除空白值超过一半的列
    half_count = len(data)/2
    data = data.dropna(thresh=half_count, axis=1)

    # 删除数值完全相同的列
    data = data.drop(['policy_code', 'application_type'], axis=1)

    # 删除与业务相关性不大的列
    data = data.drop(['emp_title', 'issue_d', 'title', 'zip_code', 'addr_state', 'earliest_cr_line'],
                     axis=1)
    # loan_status-》pymnt_plan
    data = data.drop(['term', 'funded_amnt_inv', 'pymnt_plan', 'out_prncp', 'total_rec_late_fee',
                      'tot_coll_amt', 'sub_grade', 'collection_recovery_fee'], axis=1)
    data = data.drop(['pub_rec', 'initial_list_status', 'out_prncp_inv', 'recoveries', 'total_rec_prncp',
                        'collections_12_mths_ex_med', 'verification_status'], axis=1)
    # 对非数值列数据使用数值进行替换tot_coll_amt
    status_replace1 = {
        "grade": {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6
        }
    }
    data = data.replace(status_replace1)

    status_replace2 = {
        "emp_length": {
            "n/a": 0,
            "< 1 year": 0,
            "1 year": 1,
            "2 years": 2,
            "3 years": 3,
            "4 years": 4,
            "5 years": 5,
            "6 years": 6,
            "7 years": 7,
            "8 years": 8,
            "9 years": 9,
            "10+ years": 10,
        }
    }
    data = data.replace(status_replace2)

    status_replace3 = {
        "home_ownership": {
            "NONE": 0,
            "RENT": 1,
            "OWN": 2,
            "MORTGAGE": 3,
            "OTHER": 4,
            "ANY": 5,
        }
    }
    data = data.replace(status_replace3)

    status_replace4 = {
        "verification_status": {
            "Verified": 0,
            "Source Verified": 1,
            "Not Verified": 2
        }
    }
    data = data.replace(status_replace4)

    status_replace5 = {
        "pymnt_plan": {
            "y": 0,
            "n": 1
        }
    }
    data = data.replace(status_replace5)

    status_replace6 = {
        "initial_list_status": {
            "f": 0,
            "w": 1
        }
    }
    data = data.replace(status_replace6)

    status_replace7 = {
        "term": {
            " 36 months": 0,
            " 60 months": 1
        }
    }
    data = data.replace(status_replace7)
    status_replace8 = {
        "loan_status": {
            "Charged Off": 0,
            "Fully Paid": 1,
            "Current": 2,
            "In Grace Period": 3,
            "Late (31-120 days)": 4,
            'Issued': 5,
            'Does not meet the credit policy. Status:Charged Off': 6,
            'Default': 7,
            'Late (16-30 days)': 8,
            'Does not meet the credit policy. Status:Fully Paid': 9
        }
    }
    data = data.replace(status_replace8)
    status_replace8 = {
        "purpose": {
            "debt_consolidation": 0,
            "credit_card": 1,
            "major_purchase": 2,
            "home_improvement": 3,
            "other": 4,
            'small_business': 5,
            'renewable_energy': 6,
            'car': 7,
            'house': 8,
            'medical': 9,
            'vacation': 10,
            'moving': 11,
            'wedding': 12,
            'educational': 13
        }
    }
    data = data.replace(status_replace8)
    # 对某些列空白数值数据进行删除
    # data = data.dropna(axis=0)

    # 对某些空白列使用平均值进行填充
    column_len = len(data['member_id'])
    print(column_len)
    columns = data.columns
    for x in columns:
        data[x] = data[x].fillna(data[x].mean())
    # 输出数据处理结果
    # data.to_csv('data/5.csv', index=False)

    return data, data['member_id']


# 计算f2-score和正确率
# data1:预测 data2:真实
def f2_score(data1, data2):
    tp = 0
    fn = 0
    fp = 0
    acc = 0
    for i in range(len(data1)):
        if data1[i] == 1 and data2[i+len(data1)-1] == 1:
            tp += 1
            acc += 1
        elif data1[i] == 0 and data2[i+len(data1)-1] == 1:
            fn += 1
        elif data1[i] == 1 and data2[i+len(data1)-1] == 0:
            fp += 1
        elif data1[i] == 0 and data2[i+len(data1)-1] == 0:
            acc += 1
    if (tp + fn) != 0:
        r = tp / (tp + fn)
    else:
        r = 0
    if (tp + fp) != 0:
        p = tp / (tp + fp)
    else:
        p = 0

    if p == 0 or r == 0:
        score = 0
    else:
        score = 5 * p * r / (4 * p + r)
    ac = acc / len(data1)
    return score, ac

train_data, train_id = del_data(train_filename)  # 删除id属性
train_data = train_data.drop(['member_id'], axis=1)

X_neg = train_data[:(int)(len(train_data)/2)].loc[train_data['acc_now_delinq'] == 1]
x_X = train_data[:(int)(len(train_data)/2)].loc[train_data['acc_now_delinq'] == 0]
x_ppp, X_pos = train_test_split(x_X, test_size=0.04, random_state=1)
frames = [X_pos, X_neg]
X_ = shuffle(pd.concat(frames, axis=0))
x = train_data[(int)(len(train_data)/2):].drop(['acc_now_delinq'], axis=1)
y_ = train_data[(int)(len(train_data)/2):].acc_now_delinq
# 18个属性
y = X_.acc_now_delinq
X = X_.drop(['acc_now_delinq'], axis=1)
# 结果标签
print("0-{0},1-{1},total-{2},0-rio-{3},1-rpo-{4}".format(np.sum(y == 0), np.sum(y == 1), len(y),
                                                             np.sum(y == 0) / len(y), np.sum(y == 1) / len(y)))

# change categorical
# ============================================================
# xgboost try

clf = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=18,
    min_child_weight=1,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)

useTrainCV = True
cv_folds = 5
early_stopping_rounds = 1000
if useTrainCV:
    xgb_param = clf.get_xgb_params()
    xgtrain = xgb.DMatrix(X.values, label=y.values)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_folds,
                      metrics='auc', early_stopping_rounds=early_stopping_rounds)
    clf.set_params(n_estimators=cvresult.shape[0])

# Fit the algorithm on the data
clf.fit(X, y, eval_metric='auc')

# =============================================================
# clf = tree.DecisionTreeClassifier(max_depth=25)
# clf = clf.fit(X, y)
test_data, test_id = del_data(test_filename)

# x_test = test_data.drop(['member_id'], axis=1)
# x_test = train_data.drop(['acc_now_delinq'], axis=1)
x_test = x
for i in range(len(X.columns)):
    print('{0}:{1}'.format(X.columns[i], clf.feature_importances_[i]))
result = clf.predict(x_test)
f2_score, acc = f2_score(result, y_)
print('f2-score:{0}'.format(f2_score))
print('accuracy:{0}%'.format(acc * 100))
# r = pd.DataFrame({
#     'member_id': train_id,
#     'acc_now_delinq': result
# })
# cols = ['member_id', 'acc_now_delinq']
# r = r.ix[:, cols]
# r.to_csv('data/result.csv', index=False)
# 画饼状图
l = len(result)
one_counts = np.sum(result == 1)
labels = '0-{0}'.format(l - one_counts), '1-{0}'.format(one_counts)
fracs = [(l - one_counts) / l * 100, one_counts / l * 100]
explode = [0, 0.1]
plt.axes(aspect=1)
plt.pie(x=fracs, labels=labels, explode=explode, autopct='%3.1f %%',
        shadow=True, labeldistance=1.1, startangle=90, pctdistance=0.6

        )
plt.show()
