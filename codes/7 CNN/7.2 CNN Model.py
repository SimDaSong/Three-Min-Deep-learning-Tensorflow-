import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


########
# 모델 구성
########

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# None: 입력 데이터의 개수
# 1: 특징의 개수. MNIST 데이터는 회색조 이미지나 채널에 색상이 한 개 뿐이므로

Y = tf.placeholder(tf.float32, [None, 10])  # 출력값. 10개의 분류
keep_prob = tf.placeholder(tf.float32)  # 드롭 아웃을 위한 keep_prob 플레이스 홀더


########
# 첫 번째 CNN 계층 구성
########

# 3X3 크기의 커널을 가진 컨볼루션 계층을 만듦
# 입력층 X와 첫 번째 계층의 가중치 W1을 가지고, 오른쪽과 아래쪽으로 한 칸씩 움직이는 32개의 커널을 가진 컨볼루션 계층을 만들겠다는 코드
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')  # padding='SAME'은 커널 슬라이딩 시 이미지의 가장 외곽에서 한 칸 밖으로 움직이는 옵션.
                                                                # 이미지의 테두리까지도 좀 더 정확하게 평가 가능
L1 = tf.nn.relu(L1)  # 컨볼루션 계층 완성


########
# 폴링 계층 생성
########

L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 앞서 만든 컨볼루션 계층을 입력층으로 사용하고 커널 크기를 2X2로 하는 풀링 계층을 만듦
# strides=[1,2,2,1] : 슬라이딩 시 두 칸 씩 움직이겠다


########
# 두 번째 계층 생성
########

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))  # 3X3 크기의 커널 64개로 구성한 컨볼루션 계층
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 2X2 크기의 풀링 계층


########
# 10개의 분류를 만들어내는 계층 구성
########

W3 = tf.Variable(tf.random_normal([7*7*64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 7*7*64])
# 직전의 풀링 계층의 크기가 7*7*64이므로 tf.reshape 함수를 이용해 7*7*64 크기의 1차원 계층으로 만들고 이 배열 전체를 최종 출력값의 중간 단계인 256개의 뉴런으로 연결하는 신경망을 만들어줌
# 완전 연결 계층
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)


########
# 모델 구성의 마지막
########

# 직전의 은닉층인 L3의 출력값 256개를 받아 최종 출력값인 0~9 레이블을 갖는 10개의 출력값을 만듦
W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L3, W4)


########
# 최적화 함수를 만듦
########

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


########
# 학습을 시키고 결과를 확인
########

init = tf.global_variables_initializer()
sess= tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)  # 모델에 입력값을 전달하기 위해 MNIST 데이터를 28*28 형태로 재구성

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

print('정확도:', sess.run(accuracy, feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1), Y: mnist.test.labels, keep_prob: 1}))
