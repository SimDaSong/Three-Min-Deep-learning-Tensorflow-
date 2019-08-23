# 머신러닝 학습 방법은 크게 지도 학습과 비지도 학습으로 나눌 수 있는데 비지도 학습 중 가장 널리 쓰이는 신경망이 오토인코더
# 입력값과 출력값을 같게 하는 신경망으로서 가운데 계층의 노드 수가 입력값보다 적은 것이 특징
# 이런 구조로 인해 데이터를 압축하는 효과가 있으며 노이즈 제거에 매우 효과적

# 입력층으로 들어온 데이터를 인코더를 통해 은닉층으로 내보내고 은닉층의 데이터를 디코더를 통해 출력층으로 내보낸 뒤 만들어진 출력값을 입력값과 비슷해지도록 만드는 가중치를 찾아냄

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


########
# 하이퍼파라미터로 사용할 옵션들
########

learning_rate = 0.01  # 최적화 함수에서 사용할 학습률
training_epoch = 20  # 전체 데이터를 학습할 총횟수
batch_size = 100  # 미니배치로 한 번에 학습할 데이터(이미지)의 개수
n_hidden = 256  # 은닉층의 뉴럭 개수
n_input = 28*28  # 입력값의 크기. 사용할 MNIST의 이미지 크기가 28*28이므로 784가 됨


########
# 신경망 모델 구성
########

X = tf.placeholder(tf.float32, [None, n_input])  # 이 모델은 비지도 학습이므로 Y 값이 없음


########
# 인코더와 디코더 만들기
########

# 오토인코더의 핵심 모델
# 인코더와 디코더를 만드는 방식에 따라 다양한 오토인코더를 만들 수 있음

# 인코더
# n_input 값보다 n_hidden 값이 더 작아서 입력값을 압축하고 노이즈를 제거하면서 입력값의 특징을 찾아내게 됨
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))

encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))


# 디코더
W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_decode))


########
# 가중치들을 최적화 하기 위한 손실 함수 만들기
########

# 출력값을 입력값과 가장 비슷하게 만들어야 함
# 입력값인 X를 평가하기 위한 실측값으로 사용하고 디코더가 내보낸 결괏값과의 차이를 손실값으로 설정
# 그리고 이 값의 차이는 거리 함수로 구함

cost = tf.reduce_mean(tf.pow(X - decoder, 2))


########
# RMSPropOptimizer 함수를 이용한 최적화 함수를 설정
########

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


########
# 학습 진행
########

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(training_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch+1), 'Avg. cost =', '{:.4f}'.format(total_cost / total_batch))

print('최적화 완료!')


########
# 디코더로 생성해낸 결과를 이미지로 출력
########

# 총 10개의 테스트 데이터를 가져와 디코더를 이용해 출력값으로 만듦
sample_size = 10

samples = sess.run(decoder, feed_dict={X: mnist.test.images[:sample_size]})

# numpy 모듈을 이용해 MNIST 데이터를 28*28 크기의 이미지 데이터로 재구성 한 뒤, matplotlib의 imshow 함수를 이용해 그래프에 이미지로 출력
# 위쪽에는 입력값의 이미지를, 아래쪽에는 신경망으로 생성한 이미지를 출력
fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

plt.show()

