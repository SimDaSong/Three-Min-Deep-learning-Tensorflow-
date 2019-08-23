import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)


########
# 하이퍼파라미터들을 설정
########

total_epoch = 100
batch_size = 100
learning_rate = 0.0002
n_hidden = 256
n_input = 28*28
n_noise = 128  # 생성자의 입력값으로 사용할 노이즈의 크기
# 랜덤한 노이즈를 입력하고 그 노이즈에서 손글씨 이미지를 무작위로 생성해내도록 할 것


########
# 플레이스홀더 설정
########

# GAN 역시 비지도 학습이므로 오토인코더처럼 Y를 사용하지 않음
X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])
# 구문자에 넣을 이미지가 실제 이미지와 생성한 가짜 이미지 두 개이고 가짜 이미지는 노이즈에서 생성할 것이므로 노이즈를 입력할 플레이스 홀더 Z를 추가


########
# 생성자 신경망에 사용할 변수들을 설정
########

# 첫 번째 가중치와 편향은 은닉층으로 출력하기 위한 변수들
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))

# 두 번째 가중치와 편향은 출력층에 사용할 변수들
# 따라서 가중치 변수 크기는 실제 이미지 크기와 같아야 함
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))


########
# 구분자 신경망에 사용할 변수들을 설정
########

# 은닉층은 생성자와 동일하게 구성
# 구분자는 진짜와 얼마나 가까운가를 판단하는 값으로 0~1사이의 값을 출력
# 따라서 하나의 스칼라값을 출력하도록 구성
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))

# 실제 이미지를 판별하는 구분자 신경망과 생성한 이미지를 판별하는 구분자 신경망은 같은 변수를 사용해야 함.
# 같은 신경망으로 구분을 시켜야 진짜 이미지와 가까 이미지를 구분하는 특징들을 동시에 잡아낼 수 있기 때문


########
# 생성자와 구분자 신경망 구성
########

# 생성자 신경망
def generator(noise_z):
    hidden = tf.nn.relu(tf.matmul(noise_z, G_W1)+G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)

    return output

# 생성자는 무작위로 생성한 노이즈를 받아 가중치와 편향을 반영하여 은닉층을 만들고
# 은닉층에서 실제 이미지와 같은 크기의 결괏값을 출력


# 구분자 신경망
def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)  # 0~1 사이의 스칼라값 하나를 출력하도록하였으며 이를 위한 활성화 함수로 sigmoid 함수를 사용

    return output


########
# 무작위한 노이즈를 만들어주는 간단한 유틸리티 함수
########
def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))


########
# 노이즈 Z를 이용해 가짜 이미지를 만들 생성자 G를 만들고 이 G가 만든 가짜 이미지와 진짜 이미지 X를 각각 구분자에 넣어 입력한 이미지가 진짜인지 판별하도록 함
########

G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X)


########
# 손실값 구하기
########


# 생성자가 만든 이미지를 구분자가 가짜라고 판단하도록 하는 손실값(경찰 학습용)과 진짜라고 판단하도록 하는 손실값(위조지폐범 학습용)을 구해야 함
# 경찰을 학습시키기 위해서는 진짜 이미지 판별값 D_real은 1에 가까워야 하고
# 가짜 이미지 판별값 D_gene은 0에 가까워야 함

loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))

# 위조지폐범 은 가짜 이미지 판별값 D_gene를 1에 가깝게 만들기만 하면됨
loss_G = tf.reduce_mean(tf.log(D_gene))

# 즉, GAN의 학습은 loss_D와 loss_G를 모두 최대화 하는 것


########
# 손실값을 이용해 학습 시키기
########

# loss_D를 구할 땐 구분자 신경망에 사용되는 변수들만 사용하고 loss_G를 구할 땐 생성자 신경망에 사용되는 변수들만 사용해서 최적화 해야 함
D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

# 변수를 최적화하는 함수들을 구성
# loss를 최대화해야 하지만 최적화에 쓸 수 있는 함수는 minimize 뿐이므로 최적화하려는 loss_D와 loss_G에 음수 부호를 붙여줌
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)


########
# 학습 시키기
########

# 두 개의 손실값을 학습시켜야 함
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)
loss_var_D, loss_var_G = 0, 0  # loss_D와 loss_G의 결괏값을 받을 변수 선언

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

    print('Epoch:', '%04d' % epoch,
          'D loss: {:.4}'.format(loss_val_D),
          #'D loss: {:.4}'.format(loss_var_D), # 오타 주의
          'G loss: {:.4}'.format(loss_val_G))
          #'G loss: {:.4}'.format(loss_var_G))

    ########
    # 학습 결과 확인하기
    ########

    # 노이즈를 만들고 이것을 생성자 G에 넣어 결괏값 만들기
    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Z: noise})

        # 이 결괏값들을 28*28 크기의 가짜 이미지로 만들어 samples 폴더에 저장하기
        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('최적화 완료!')

