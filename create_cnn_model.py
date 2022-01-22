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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,AveragePooling1D
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

def NormalizeData(data):
	#print(numpy.amin(data))	
	#print(numpy.amax(data))	
	return (data + abs(numpy.amin(data))) / (numpy.amax(data) - numpy.amin(data))


# setting the seed
seed(1)
set_seed(1)

# load array
X_train_whole = loadtxt('d:\\eeg\\combined_eeg_handle.csv', delimiter=',')

# augment data
choice = X_train_whole[:, -1] == 0.
X_total_1 = numpy.append(X_train_whole, X_train_whole[choice, :], axis=0)
X_total = numpy.append(X_total_1, X_train_whole[choice, :], axis=0)
print(X_total.shape)

# balancing
sm = SMOTE(random_state = 2)
X_train_res, Y_train_res = sm.fit_resample(X_total[:, 0:960], X_total[:, -1].ravel())
print("After OverSampling, counts of label '1': {}".format(sum(Y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_res == 0)))

tensorflow.compat.v1.reset_default_graph()
X_train, X_test, Y_train, Y_test = train_test_split(X_train_res, Y_train_res, random_state=1, test_size=0.3, shuffle = True)
print(X_train.shape)
print(X_test.shape)


#=======================================
 
# Model configuration

input = numpy.zeros((0, 960))
testinput = numpy.zeros((0, 960))

mean_of_train = mean(X_train[:, 0:960])
print(mean_of_train)
input = X_train[:, 0:960] - mean_of_train
#input = NormalizeData(input)
input = numpy.apply_along_axis(NormalizeData, 1, input)
input_output = numpy.append(input, Y_train.reshape(len(Y_train), 1), axis=1) 
savetxt('d:\\input_output.csv', input_output, delimiter=',')

mean_of_test = mean(X_test[:, 0:960])
print(mean_of_test)
testinput = X_test[:, 0:960] - mean_of_test
#testinput = NormalizeData(testinput)
testinput = numpy.apply_along_axis(NormalizeData, 1, testinput)
savetxt('d:\\testinput.csv', testinput, delimiter=',')


#=====================================

print(len(input))
print(len(testinput))

input = input.reshape(len(input), 15, 64)
input = input.transpose(0, 2, 1)
print (input.shape)

testinput = testinput.reshape(len(testinput), 15, 64)
testinput = testinput.transpose(0, 2, 1)
print (testinput.shape)


# Create the model
model=Sequential()
model.add(Conv1D(filters=20, kernel_size=6, padding='valid', activation='relu', strides=1, input_shape=(64, 15)))
model.add(Conv1D(filters=20, kernel_size=6, padding='valid', activation='relu', strides=1))
model.add(AveragePooling1D(pool_size=2))
model.add(Dropout(0.4))
model.add(Conv1D(filters=40, kernel_size=3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

# Compile the model
adam = Adam()       
model.compile(loss=sparse_categorical_crossentropy, optimizer=adam, metrics=['accuracy'])


hist = model.fit(input, Y_train, batch_size=32, epochs=500, verbose=1, validation_data=(testinput, Y_test), steps_per_epoch=None)		


# evaluate the model
Y_hat_classes = model.predict_classes(testinput)
matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)


# plot training history
pyplot.plot(hist.history['loss'], label='tr_loss')
pyplot.plot(hist.history['val_loss'], label='val_loss')
pyplot.plot(hist.history['accuracy'], label='tr_accuracy')
pyplot.plot(hist.history['val_accuracy'], label='val_accuracy')
pyplot.legend()

pyplot.show()

#==================================

model.save("D:\\eeg\\model_conv1d.h5")

#==================================

#Removed dropout and reduced momentum and reduced learning rate