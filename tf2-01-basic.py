import numpy as np
import tensorflow as tf

# x and y data
x_train = [1,2,3,4]
y_train = [1,2,3,4]

# keras의 다차원 계층 모델인 Sequential를 레이어를 만든다.
tf.model = tf.keras.Sequential()
# 입력이 1차원이고 출력이 1차원임을 뜻함 - Dense는 레이어의 종류
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

# Optimizer - Stochastic gradient descent - 확률적 경사 하강법
sgd = tf.keras.optimizers.SGD(learning_rate=0.01)

# cost/loss funcion
# loss를 mean_squared_error 방식을 사용한다는 의미로 mse 라고 써도 인식한다.
tf.model.compile(loss='mean_squared_error',optimizer=sgd)

#fit the line
# 텐서 플로우 1과 다르게 세션을 만들어서 돌릴 필요가 없다.
# 간단하게 만들어서 학습을 시작한다.
tf.model.fit(x_train, y_train, epochs=2000)

print(tf.model.predict(np.array([5])))