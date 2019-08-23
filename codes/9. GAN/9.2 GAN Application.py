# 숫자를 무작위로 생성하지 않고 원하는 숫자를 지정해 생성하는 모델 만들기
# 노이즈에 레이블 데이터를 힌트로 넣어주는 방법 사용

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

total_epoch = 100
batch_size = 100
n_hidden = 256
n_input = 28*28
n_noise = 128
n_class = 10

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])  # 결괏값 판정용이 아니라 노이즈와 실제 이미지에 각각에 해당하는 숫자를 힌트로 넣어주는 용도
Z = tf.placeholder(tf.float32, [None, n_noise])

# MNIST 데이터의 레이블은 원-핫 인코딩한 10개의 값으로 구성되어 있음


########
# 생성자 신경망 구성
########

# 변수들을 선언하지 않고 tf.layer를 사용
# tf.variable_scope를 이용해 스코프를 지정해줄 수 있어서 나중에 이 스코프에 해당하는 변수들만 따로 불러올 수 있음

def generator(noise, labels):
    with tf.variable_scope('generator'):
        inputs = tf.concat([noise, labels], 1)  # tf.concat 함수를 통해 noise 값에 labels 정보를 간단하게 추가

        hidden = tf.layers.dense(inputs, n_hidden, activation=tf.nn.relu)  # tf.layers.dense 함수를 이용해 은닉층(hidden)을 만들고
        output = tf.layers.dense(hidden, n_input, activation=tf.nn.sigmoid)  # 진짜 이미지와 같은 크기의 값을 만드는 출력층(output)도 구성

    return output


########
# 구분자 신경망 구성
########

# 구분자는 진짜 이미지를 판별할 때와 가짜 이미지를 판별할 때 똑같은 변수를 사용해야 함
# 그러기 위해 scope.reuse_variables 함수를 이용해 이전에 사용한 변수를 재사용하도록 작성

def discriminator(inputs, labels, reuse=None):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        inputs = tf.concat([inputs, labels], 1)
        hidden = tf.layers.dense(inputs, n_hidden, activation=tf.nn.relu)
        output = tf.layers.dense(hidden, 1, activation=None)  # 활성화 함수를 사용하지 않은 이유는 손실값 계산에 sigmoid_cross_entropy_with_logits 함수를 사용하기 위함

    return output


########
# 노이즈 생성 유틸리티 함수
########

# 노이즈를 균등분포로 생성하도록 작성
def get_noise(batch_size, n_noise):
    return np.random.uniform(-1., 1., size=[batch_size, n_noise])


########
# 생성자를 구성하고 진짜 이미지 데이터와 생성자가 만든 이미지 데이터를 이용하는 구분자를 하나씩 만들어줌
########

# 이때 생성자에는 레이블 정보를 추가하여 추후 레이블 정보에 해당하는 이미지를 생성할 수 있도록 유도
# 가짜 이미지를 만들 땐 진짜 이미지 구분자에서 사용한 변수들을 재사용하도록 reuse 옵션을 True로 설정

G = generator(Z, Y)
D_real = discriminator(X, Y)
D_gene = discriminator(G, Y, True)


########
# 손실 함수 만들기
########

# GAN 논문의 방식과는 약간 다르게 작성
# sigmoid_cross_entropy_with_logits 함수를 이용해 코드를 좀 더 간편하게 작성
loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.zeros_like(D_gene)))

loss_D = loss_D_real + loss_D_gene  # 이 값을 최소화 하면 구분자(경찰)을 학습시킬 수 있음

# loss_G는 생성자(위조지폐범)를 학습시키기 위한 손실값
loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.ones_like(D_gene)))

# tf.get_collection 함수를 이용해 discriminator와 generator 스코프에서 사용된 변수들을 가져온 뒤 이 변수들을 최적화에 사용할 각각의 손실 함수와 함께 최적화 함수에 넣어 학습 모델 구성을 마무리
vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

train_D = tf.train.AdamOptimizer().minimize(loss_D, var_list=vars_D)
train_G = tf.train.AdamOptimizer().minimize(loss_G, var_list=vars_G)


########
# 학습 진행
########

# 플레이스홀더 Y의 입력값으로 batch_ys 값을 넣어줌에 주의
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Y: batch_ys, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Y: batch_ys, Z: noise})

    print('Epoch:', '%04d' % epoch, 'D loss: {:.4}'.format(loss_val_D), 'G loss: {:.4}'.format(loss_val_G))

    # 학습 중간중간에 생성자로 만든 이미지를 저장하는 코드를 작성
    # 플레이스 홀더 Y에 입력값을 넣어줌에 주의
    # 위쪽에는 진짜 이미지를 출력하고 아래쪽에는 생성한 이미지를 출력

    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Y: mnist.test.labels[:sample_size], Z: noise})

        fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

        for i in range(sample_size):
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()

            ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('samples2/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('최적화 완료!')

