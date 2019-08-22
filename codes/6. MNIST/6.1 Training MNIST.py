########
# MNIST 데이터셋을 사용하기 위한 준비
########

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data  # 텐서플로에 내장된 tensorflow.examples.tutorials.mnist.input_data 모듈을 임포트
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)  # MNIST 데이터를 내려 받고 레이블을 동물 분류 예제에서 본 원-핫 인코딩 방식으로 읽어들임


########
# 신경망 모델 구성하기
########

X = tf.placeholder(tf.float32, [None, 784])  # MNIST의 손글씨 이미지는 28*28 픽셀로, 총 784개의 특징으로 이뤄져 있음
Y = tf.placeholder(tf.float32, [None, 10])  # 레이블은 0~9이니 10개의 분류로 나눔
# tf.placeholder(dtype, shape, name)
# dype : placeholder에 저장되는 data형
# shape : 행렬의 차원. 원하는 배치 크기로 정확하게 명시해줘도 되지만, None으로 설정하면 텐서플로가 알아서 계산함
# name : 플레이스 홀더의 이름


########
# 2개의 은닉층이 구성된 신경망 생성
########

# 간략하게 보여주기 위해 편향 사용 안함
# 784(입력, 특징 개수) -> 256(첫 번째 은닉층 뉴런 개수) -> 256(두 번째 은닉층 뉴런 개수) -> 10(결괏값 0~9 분류 개수)
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))  # 표준편차가 0.01인 정규분포를 가지는 임의의 값으로 뉴런(변수)를 초기화 시킴
L1 = tf.nn.relu(tf.matmul(X, W1))
# tf.matmul : 행렬곱 연산을 수행하는 함수. 각 계층으로 들어오는 입력값에 각각의 가중치를 곱하고 tf.nn.relu 함수를 이용하여 활성화 함수로 ReLU를 사용하는 신경망 계층을 만듦

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)  # 0~9의 숫자를 나타내는 요소 10개짜리 배열이 출력됨. 가장 큰 값을 가진 인덱스가 예측 결과에 가까운 숫자.
# 출력 층에는 보통 활성화 함수를 사용하지 않음


########
# 최적화를 수행하도록 그래프를 구성
########

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
# tf.nn.softmax_cross_entropy_with_logits 함수로 각 이미지에 대한 손실값 구함
# tf.reduce_mean 함수를 이용해 미니배치의 평균 손실값을 구함

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
# Optimizer 함수를 사용하여 이 손실값을 최소화 하는 최적화를 수행하도록 그래프를 구성


# 앞서 구성한 신경망 모델을 초기화하고 학습을 진행할 세션을 시작
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


########
# 실제 학습 진행
########

batch_size = 100  # 미니배치의 크기를 100으로 설정
total_batch = int(mnist.train.num_examples / batch_size)  # 학습 데이터의 총 개수인 mnist.train.num_examples를 배치 크기로 나눠 미니배치가 총 몇 개인지 저장해둠
# MNIST는 데이터가 매우 크므로 학습에 미니배치를 사용

# MNIST 데이터 전체를 학습하는 일을 총 15번 반복
# 학습 데이터 전체를 한 바퀴 도는 것을 epoch(에포크)라고 함
for epoch in range(15):
    total_cost = 0

    # 미니배치의 총 개수만큼 반복하여 학습
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # mnist.train.next_batch 함수를 이용해 학습할 데이터를 배치 크기만큼 가져온 뒤 입력값인 이미지 데이터는 batch_xs에, 출력값인 레이블 데이터는 batch_ys에 저장함

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs, Y: batch_ys})
        # sess.run을 통해 최적화시키고 손실값을 가져와서 저장함
        # 이때, feed_dict 매개변수에 입력값 X와 예측을 평가할 실제 레이블값 Y에 사용할 데이터를 넣어줌

        # 손실값을 저장
        total_cost += cost_val

    # 한 세대의 학습이 끝나면 학습한 세대의 평균 손실값을 출력
    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost = ', '{:3f}'.format(total_cost / total_batch))

print('최적화 완료!')


########
# 예측 결과인 model의 값과 실제 레이블인 Y의 값을 비교
########

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
# 예측한 결괎값은 원-핫 인코딩 형식이며 각 인덱스에 해당하는 값은 해당 숫자가 얼마나 해당 인덱스와 관련이 높은가를 나타냄
# 가장 값이 큰 인덱스(여기서는 7)가 가장 근접한 예측 결과
# 이는 손실값을 sotfmax_cross_entropy_with_logits을 이용해 구했기 때문이며 초깃값이나 예측 모델, 손실값을 구하는 방식에 따라 결과가 달라질 수 있음

# tf.argmax(model, 1)은 두 번째 차원(1번 인덱스의 차원)의 값 중 최댓값의 인덱스를 뽑아내는 함수.
# model로 출력한 결과는 결괎값을 배치 크기만큼 가지고 있음. 따라서 두 번째 차원이 예측한 각각의 결과임(ex. [None, 10])

# tf.equal 함수를 통해 예측한 숫자와 실제 숫자가 같은지 확인


########
# 정확도(확률) 구하기
########

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
# is_correct를 0과 1로 변환하고 변환한 값들을 tf.reduce_mean을 이용해 평균을 내어 정확도 측정

# 테스트 데이터를 다루는 객체인 mnist.test를 이용해 테스트 이미지와 레이블 데이터를 넣어 accuracy를 계산
print('정확도:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

