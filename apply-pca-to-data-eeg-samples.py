import numpy
from imblearn.over_sampling import SMOTE
from numpy import savetxt
from numpy import loadtxt
from matplotlib import pyplot
from pandas import DataFrame
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from numpy import savetxt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.losses import sparse_categorical_crossentropy
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from keras.utils import to_categorical 
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder
import dill as pickle
from sklearn.pipeline import Pipeline
from numpy import asarray
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import learning_curve
from sklearn.ensemble import StackingClassifier
from keras.layers import Input
from keras.models import Model
from keras.losses import binary_crossentropy
from tensorflow.keras import regularizers
from numpy.random import seed
from tensorflow.random import set_seed
import tensorflow
from sklearn.decomposition import PCA


# setting the seed
seed(1)
set_seed(1)

# load data from spreadsheet
combined_eeg_handle_yes = loadtxt('D:\\eeg\\combined_eeg_handle_yes.csv', delimiter=',')
combined_eeg_handle_no = loadtxt('D:\\eeg\\combined_eeg_handle_no.csv', delimiter=',')

#Calculate average target vector
avg_data_vector_target = numpy.mean(combined_eeg_handle_yes[:, 0:960], axis=0)
print(avg_data_vector_target.shape)


#Calculate average non-target vector
avg_data_vector_nonTargetData = numpy.mean(combined_eeg_handle_no[:, 0:960], axis=0)
print(avg_data_vector_nonTargetData.shape)

avg_t_nt = numpy.mean(numpy.append(combined_eeg_handle_yes[:, 0:960], combined_eeg_handle_no[:, 0:960], axis=0), axis=0)

pca_target = PCA(n_components=15)
pca_target.fit(combined_eeg_handle_yes)
print(pca_target.components_.shape)
print(pca_target.explained_variance_ratio_)
eigen_vector_target = pca_target.components_[0, 0:960]
eigen_vector_target_1 = pca_target.components_[1, 0:960]
eigen_vector_target_2 = pca_target.components_[2, 0:960]

pca_non_target = PCA(n_components=15)
pca_non_target.fit(combined_eeg_handle_no)
print(pca_non_target.components_.shape)
print(pca_non_target.explained_variance_ratio_)
eigen_vector_nonTargetData = pca_non_target.components_[0, 0:960]
eigen_vector_nonTargetData_1 = pca_non_target.components_[1, 0:960]
eigen_vector_nonTargetData_2 = pca_non_target.components_[2, 0:960]

data_characteristics = numpy.empty([10, 960])

data_characteristics = numpy.append(numpy.asarray(avg_data_vector_target).reshape(-1, 960), numpy.asarray(avg_data_vector_nonTargetData).reshape(-1, 960), axis=0)
data_characteristics = numpy.append(data_characteristics, numpy.reshape(numpy.asarray(avg_t_nt), (-1, 960)), axis=0)
data_characteristics = numpy.append(data_characteristics, numpy.reshape(numpy.asarray(eigen_vector_target), (-1, 960)), axis=0)
data_characteristics = numpy.append(data_characteristics, numpy.reshape(numpy.asarray(eigen_vector_target_1), (-1, 960)), axis=0)
data_characteristics = numpy.append(data_characteristics, numpy.reshape(numpy.asarray(eigen_vector_target_2), (-1, 960)), axis=0)
data_characteristics = numpy.append(data_characteristics, numpy.reshape(numpy.asarray(eigen_vector_nonTargetData), (-1, 960)), axis=0)
data_characteristics = numpy.append(data_characteristics, numpy.reshape(numpy.asarray(eigen_vector_nonTargetData_1), (-1, 960)), axis=0)
data_characteristics = numpy.append(data_characteristics, numpy.reshape(numpy.asarray(eigen_vector_nonTargetData_2), (-1, 960)), axis=0)

savetxt('d:\\eeg\\data_characteristics_eeg.csv', data_characteristics, delimiter=',')

