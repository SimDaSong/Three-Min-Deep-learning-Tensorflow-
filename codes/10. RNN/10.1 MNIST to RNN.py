# 이미지 인식에 CNN이 있다면 자연어 인식에는 순환 신경망이라고 하는 RNN이 있다

########
# 하이퍼파라미터, 변수, 출력층을 위한 가중치와 편향 정의
########

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

learning_rate = 0.001
total_epoch = 30
batch_size = 128

# RNN은 순서가 있는 데이터를 다루므로 한 번에 입력 받을 개수와 총 몇 단계로 이뤄진 데이터를 받을지를 결정해야 함
n_input = 28  # 가로 픽셀 수
n_step = 28  # 세로 픽셀 수

n_hidden = 128
n_class = 10

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])
W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# n_hidden개의 출력값을 갖는 RNN 셀을 생성
cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
# 신경망 구성을 위한 함수는 BasicRNNCell 이외에도 다양한 방식이 있음

# dynamic_rnn 함수를 이용해 RNN 신경망을 완성
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


########
# 최종 출력값 만들기
########

# RNN 신경망에서 나오는 출력값은 각 단계가 포함된 [batch_size, n_step, n_hidden] 형태이기 때문에
# 다음 코드 같이 은닉층의 출력값을 가중치 W와 같은 상태로 만들어줘야 행렬곱을 수행하여 원하는 출력값을 얻을 수 있음
# 참고로 dynamic_rnn 함수의 옵션 중 time_major의 값을 True로 하면 [n_step, batch_size, n_hidden]의 형태로 출력됨

# outputs: [batch_size, n_step, n_hidden]
# -> [n_step, batch_size, n_hidden]
outputs = tf.transpose(outputs, [1, 0, 2])
# -> [batch_size, n_hidden]
outputs = outputs[-1]

# 최종 결괏값 만들기
model = tf.matmul(outputs, W) + b


########
# 손실값 구하고 신경망을 최적화하는 하수를 사용
########

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


########
# 신경망을 학습시키고 결과 확인
########

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(total_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, n_step, n_input))

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)

test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
test_ys = mnist.test.labels

print('정확도:', sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys}))