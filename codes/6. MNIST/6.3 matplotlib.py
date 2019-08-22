# matplotlib는 시각화를 위해 그래프를 쉽게 그릴 수 있도록 해주는 파이썬 라이브러리

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  # matplotlib의 pyplot 모듈을 import

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)


########
# 신경망 모델 구성
########

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


########
# 신경망 모델 학습
########

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(30):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')


########
# 결과 확인
########

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))


########
# 결과 확인(matplotlib)
########

labels = sess.run(model, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1})
# 테스트 데이터를 이용해 예측 모델을 실행하고 결괏값을 labels에 저장

fig = plt.figure()  # 손글씨를 출력할 그래프를 준비
for i in range(10):  # 테스트 데이터의 첫 번째부터 열 번째까지의 이미지와 예측한 값을 출력합니다.
    # 2행 5열의 그래프를 만들고, i+1 번째에 숫자 이미지를 출력합니다.
    subplot = fig.add_subplot(2, 5, i + 1)
    # 이미지를 깨끗하게 출력하기 위해 x와 y의 눈금을 출력하지 않습니다.
    subplot.set_xticks([])
    subplot.set_yticks([])
    # 출력한 이미지 위에 예측한 숫자를 출력합니다.
    # np.argmax는 tf.argmax와 같은 기능의 함수
    # 결괏값인 labels의 i번째 요소가 원-핫 이코딩 형식으로 되어 있으므로 해당 배열에서 가장 높은 값을 가진 인덱스를 예측한 숫자로 출력
    subplot.set_title('%d' % np.argmax(labels[i]))
    # 1차원 배열로 되어 있는 i 번째 이미지 데이터를 28x28 형식의 2차원 배열로 변형하여 이미지 형태로 출력
    # cmap 파라미터를 통해 이미지를 그레이 스케일로 출력
    subplot.imshow(mnist.test.images[i].reshape((28, 28)), cmap='gray')

plt.show()  # 그래프를 화면에 표시