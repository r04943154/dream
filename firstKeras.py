# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import matplotlib.pyplot as plt 
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("mul2_add3.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0]
Y = dataset[:,1]
plt.scatter(X, Y)
plt.show()
Xtrain, Ytrain=X[:40], Y[:40]
Xtest, Ytest=X[40:], Y[40:]
# create model
model = Sequential()
model.add(Dense(units=1,input_dim=1,use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
#model.add(Dense(12, input_dim=8, activation='relu'))
#model.add(Dense(8, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='mse', optimizer='rmsprop')
for step in range(300):
 cost = model.train_on_batch(Xtrain, Ytrain)
 if step % 10 == 0:
  print "train cost: ",cost 
print "\nTesting ------------"
cost = model.evaluate(Xtest, Ytest, batch_size=10)
print "test cost:", cost
W, b = model.layers[0].get_weights()
print "Weights=", W, "\nbiases=", b
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
#model.fit(X, Y, epochs=2, batch_size=10)

# evaluate the model
#scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
