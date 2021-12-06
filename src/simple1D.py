
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution1D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import random_uniform


#hyperparameters
input_dimension = 88200
learning_rate = 0.0025
momentum = 0.85
SEED = 24
linear_init = random_uniform(seed=SEED)
dropout_rate = 0.2


def create_model(init_shape,  hidden_units=256, n_class =3, dropout_rate=0.3, loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy']):
    # create model
    model = Sequential()
    ksize =2* int(init_shape[0]//8)+1
    model.add(Convolution1D(filters=32, kernel_size=ksize , input_shape=init_shape, activation='relu'))
    model.add(Convolution1D(filters=16, kernel_size=9, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    #model.add(Dense(input_dimension//2, input_dim=input_dimension, kernel_initializer=linear_init, activation='relu'))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_units, input_dim=init_shape[0], kernel_initializer=linear_init, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_units//2, kernel_initializer=linear_init, activation='relu'))
    model.add(Dense(n_class, kernel_initializer=linear_init, activation='softmax'))

    model.compile(loss=str(loss), optimizer=str(optimizer), metrics=metrics)
    #sgd = SGD(lr=learning_rate, momentum=momentum)
    #model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])
    return model

if __name__ =='__main__':
    m = create_model(init_shape = (88200,1), n_class =3)
    m.summary()
#model.fit(X_train, y_train, epochs=5, batch_size=128)
#predictions = model.predict_proba(X_test)
