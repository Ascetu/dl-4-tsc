import sklearn
from tslearn.utils import save_time_series_txt, load_time_series_txt
import numpy as np
from tslearn.utils import to_sklearn_dataset
import tensorflow as tf
from utils.utils import calculate_metrics
import sys

def predict(x_test, y_true):
    model_path = output_directory + 'best_model.hdf5'
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    df_metrics = calculate_metrics(y_true, y_pred, 0.0)
    return df_metrics

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
x_train = to_sklearn_dataset(X_train[:400])
y_train = to_sklearn_dataset(Y_train[:400])
x_test = to_sklearn_dataset(X_prd)
y_test = to_sklearn_dataset(Y_prd)

print("转换为sklearn格式数据，形状：", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

nb_classes = len(np.unique(y_train))

# train标签转换为one-hot向量
enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(y_train.reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()

# test标签转换为one-hot向量
enc2 = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc2.fit(y_test.reshape(-1, 1))
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

# test标签转回去(直接用原始的y_test会报错，后续探究原因)
y_true = np.argmax(y_test, axis=1)

print("标签转换后以及原始形状，y_train，y_test，y_true：", y_train.shape, y_test.shape, y_true.shape)

# todo 加载模型，使用模型进行预测
classifier_name = "fcn"
classifier_name = sys.argv[1]
output_directory = "./results/" + classifier_name + "/"

# 预测测试数据，并计算准确率
res = predict(x_test, y_true)
print(res)


