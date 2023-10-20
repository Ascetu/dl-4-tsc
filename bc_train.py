import sklearn
from tslearn.utils import save_time_series_txt, load_time_series_txt
import numpy as np
from tslearn.utils import to_sklearn_dataset
import model
import sys
import os

# 读取数据，tslearn格式(暂时如此，后面优化为直接读取sklearn格式数据)
# x_train
X_train = load_time_series_txt("BC_X.txt")
# y_train
data = np.loadtxt('BC_Y.txt')
Y_train = data.flatten()

# x_test
X_prd = load_time_series_txt("BC_X_PRD.txt")
# y_test
data2 = np.loadtxt('BC_Y_PRD.txt')
Y_prd = data2.flatten()

print("转换前tslearn格式数据，形状：", X_train.shape, Y_train.shape, X_prd.shape, Y_prd.shape)

# 转换为sklearn格式数据
x_train = to_sklearn_dataset(X_train[:5000])
y_train = to_sklearn_dataset(Y_train[:5000])
x_test = to_sklearn_dataset(X_prd[:5000])
y_test = to_sklearn_dataset(Y_prd[:5000])

print("转换后sklearn格式数据，形状：", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

# train、test标签转换为one-hot向量
enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

# test标签转回去(直接用原始的y_test会报错，后续探究原因)
y_true = np.argmax(y_test, axis=1)

print("标签转换后以及原始形状，y_train，y_test，y_true：", y_train.shape, y_test.shape, y_true.shape)

if len(x_train.shape) == 2:  # if univariate：如果是单变量的
    # add a dimension to make it multivariate with one dimension
    # 添加一个维度，使其成为具有一个维度的多元变量"。这可能是指在数据分析中，将单变量数据转化为多元变量数据，但只增加一个维度
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    input_shape = x_train.shape[1:]

    print("训练、测试数据增加维度后形状，x_train，x_test：", x_train.shape, x_test.shape)

    # 构建算法模型（后续根据指令选择不同算法）
    classifier_name = "fcn"
    classifier_name = sys.argv[1]
    output_directory = "results/" + classifier_name + "/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # classifier = Classifier_FCN("fcn", input_shape, nb_classes, "fcn")
    classifier = model.create_classifier(classifier_name, input_shape, nb_classes, output_directory)
    # 训练
    classifier.fit(x_train, y_train, x_test, y_test, y_true)

    # 评估
    res = classifier.predict(x_test, y_true)
    print(res)
