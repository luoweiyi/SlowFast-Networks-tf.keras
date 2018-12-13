from model import SlowFast_Network
import numpy as np

X_train_slow=np.random.rand(3,4,224,224,3)
X_train_fast=np.random.rand(3,32,224,224,3)
y_train=np.random.rand(3,10)

model = SlowFast_Network(clip_shape=[64,224,224,3],num_class=10,alpha=8,beta=1/8,tau=16)
print(model.summary())
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([X_train_slow,X_train_fast], y_train, batch_size=1)
