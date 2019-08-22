# Drop Out: 과적합을 피하기 위한 방법 중 효과가 좋은 방법 중 하나
# 학습 시 전체 신경망 중 일부만을 사용하도록 함. 즉, 학습 단계마다 일부 뉴런을 제거함으로써 일부 특징이 특정 뉴런들에 고정되는 것을 막아 가중치의 균형을 잡도록 하여 과적함을 방지
# 학습 시 일부 뉴런을 학습시키지 않기 때문에 신경망이 충분히 학습되기까지의 시간은 조금 더 오래걸리긴 함

# 최근에는 드롭 아웃보다 배치 정규화(Batch Normalization)이라는 기법이 많이 사용됨
# 이 기법은 과적합을 막아줄 뿐 아니라 학습 속도도 향상시켜 주는 장점이 있음

########
# MNIST 데이터셋을 사용하기 위한 준비
########

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


########
# 신경망 모델 구성하기
########

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)


########
# 2개의 은닉층이 구성된 신경망 생성
########

# 드롭아웃 기법을 사용해 학습하더라도, 학습이 끝난 뒤 예측 시에는 신경망 전체를 사용하도록 해줘야 함
# 이렇게 플레이스홀더를 만들어, 학습 시에는 0.8을 넣어 드롭 아웃을 사용하도록 하고 예측 시에는 1을 넣어 신경망 전체를 사용하도록 만들어야 함

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
L1 = tf.nn.dropout(L1, keep_prob)  # 추가된 부분

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, keep_prob)  # 추가된 부분
# 0.8은 사용할 뉴런의 비율.

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)


########
# 최적화를 수행하도록 그래프를 구성
########

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


########
# 실제 학습 진행
########

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(30):  # 드롭아웃은 느리게 진행되기 때문에 epoch를 30번으로 늘려 더 많이 학습 시킴
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})  # 학습 시에는 0.8을 넣어 드롭 아웃을 사용

        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost = ', '{:3f}'.format(total_cost / total_batch))

print('최적화 완료!')


########
# 예측 결과인 model의 값과 실제 레이블인 Y의 값을 비교
########

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))


########
# 정확도(확률) 구하기
########

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

print('정확도:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))  # 예측 시에는 1을 넣어 신경망 전체를 사용

