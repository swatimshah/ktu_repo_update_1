from sklearn import preprocessing
from numpy import loadtxt
import numpy
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from numpy import savetxt
from imblearn.over_sampling import SMOTE
import tensorflow
from sklearn.model_selection import train_test_split
from numpy import mean


def NormalizeData(data):
	#print(numpy.amin(data))	
	#print(numpy.amax(data))	
	return (data + abs(numpy.amin(data))) / (numpy.amax(data) - numpy.amin(data))
	#return (data + (MY_CONST)) / (MY_CONST - (MY_CONST_NEG))

X = loadtxt('d:\\eeg\\eeg_test.csv', delimiter=',')

mean_of_test = mean(X[:, 0:960])
print(mean_of_test)
input = X[:, 0:960] - mean_of_test
#input = NormalizeData(input)
input = numpy.apply_along_axis(NormalizeData, 1, input)
savetxt('d:\\input-swati-online.csv', input, delimiter=',')

input = input.reshape(len(input), 15, 64)
input = input.transpose(0, 2, 1)

y_real = X[:, -1]

model = load_model('D:\\eeg\\model_conv1d.h5')
y_pred = model.predict_proba(input) 
print(y_pred.shape)

y_max = numpy.argmax(y_pred, axis=1)
matrix = confusion_matrix(y_real, y_max)
print(matrix)
