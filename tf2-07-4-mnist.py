import tensorflow as tf


mnist = tf.keras.datasets.mnist

#MNIST 데이터셋을 로드하여 준비합니다. 샘플 값을 정수에서 부동소수로 변환합니다:
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#print(x_train)

# 모델을 만든다.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#모델을 훈련하고 평가합니다:
model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
