from dataset import load_dataset
from model import make_model
from envs import *

# load dataset
((X_a_train, X_v_train), y_train), ((X_a_test, X_v_test), y_test) = load_dataset()

# load model
model = make_model()

# compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


try:
  # train the model
  model.fit(x=[X_a_train, X_v_train],y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
  # evaluate the model
  model.evaluate(x=[X_a_test, X_v_test], y=y_test, batch_size=BATCH_SIZE)

except:
  pass

finally:
  #freeze all model layers
  for layer in model.layers:
    layer.trainable = False
  # save the model
  model.save(SAVED_MODEL)
  print('model freezed and saved')
