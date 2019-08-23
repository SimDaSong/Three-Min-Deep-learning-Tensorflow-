# 영문자 4개로 구성된 단어를 학습시켜 3글자만 주어지면 나머지 한 글자를 추천하여 단어를 완성하는 프로그램
# 한 글자가 한 단계의 입력값이 되고, 총 글자 수가 전체 단계가 됨

import tensorflow as tf
import numpy as np

# 입력으로 알파벳 순서에서 각 글자에 해당하는 인덱스를 원-핫 인코딩으로 표현할 값을 취하기 위해
# 알파벳 글자들을 배열에 넣고 해당 글자의 인덱스를 구할 수 있는 연관 배열(딕셔너리)도 만둘어 둠
# # {'a': 0, 'b': 1, 'c': 2, ..., 'j': 9, 'k', 10, ...}
char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 학습에 사용할 단어를 배열로 저장
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']

# 단어들을 학습에 사용할 수 있는 형식으로 변환해주는 유틸리티 함수를 작성
def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        # 여기서 생성하는 input_batch 와 target_batch 는
        # 알파벳 배열의 인덱스 번호 입니다.
        # [22, 14, 17] [22, 14, 14] [3, 4, 4] [3, 8, 21] ...
        input = [num_dic[n] for n in seq[:-1]]
        # 3, 3, 15, 4, 3 ...
        target = num_dic[seq[-1]]
        # one-hot 인코딩을 합니다.
        # if input is [0, 1, 2]:
        # [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]]
        input_batch.append(np.eye(dic_len)[input])
        # 지금까지 손실함수로 사용하던 softmax_cross_entropy_with_logits 함수는
        # label 값을 one-hot 인코딩으로 넘겨줘야 하지만,
        # 이 예제에서 사용할 손실 함수인 sparse_softmax_cross_entropy_with_logits 는
        # one-hot 인코딩을 사용하지 않으므로 index 를 그냥 넘겨주면 됩니다.
        target_batch.append(target)

    return input_batch, target_batch


########
# 신경망 모델 구성
########

# 옵션들 설정
learning_rate = 0.01
n_hidden = 128
total_epoch = 30

n_step = 3  # 단어의 전체 중 처음 3글자를 단계적으로 학습할 것이므로
n_input = n_class = dic_len  # 입력과 출력 값은 알파벳의 원-핫 인코딩을 사용할 것이므로
# spare_sotfmax_cross_entropy_with_logits 함수를 사용하더라도 비교를 위한 예측 모델의 출력값은 원-핫 인코딩을 사용해야 함

# 신경망 모델 구성
X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.int32, [None])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# 두 개의 RNN 셀을 생성
# 여러 셀을 조합해 심층 신경망을 만들기 위해서
# DropoutWrapper 함수를 사용하여 RNN에도 과적합 방지를 위한 드롭아웃 기법을 쉽게 적용 가능
cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

# 앞서 만든 셀들을 MultiRNNCell 함수를 사용하여 조합하고 dynamic_rnn 함수를 사용하여 심층 순환 신경망(Deep RNN)을 만듦
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

# RNN의 첫 예제인 MNIST 예측 모델과 같은 방식으로 최종 출력층을 만듦
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

# 손실함수와 최적화 함수를 사용하며 마무리
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


########
# 신경망 학습 시키기
########

sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, target_batch = make_batch(seq_data)  # make_batch 함수를 이용하여 seq_data에 저장한 단어들을 입력값(처음 세 글자)과 실측값(마지막 한 글자)로 분리하고
# 이 값들을 최적화 ㅎ마수를 실행하는 코드에 넣어 신경망을 학습시킴

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')


########
# 결괏값으로 예측한 단어를 정확도와 함께 출력
########

prediction = tf.cast(tf.argmax(model, 1), tf.int32)
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

# 여기서는 실측값을 원-핫 인코딩이 아닌 인덱스를 그대로 사용하므로 실측값, 즉 Y는 정수
# 따라서 argmax로 변환한 예측값도 정수로 변경해줌
# 정확도를 구할 땐 입력값을 그대로 비교


# 학습에 사용한 단어들을 넣고 예측 모델을 돌림
input_batch, target_batch = make_batch(seq_data)

predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X: input_batch, Y: target_batch})


# 모델이 예측합 값들을 가지고 각각의 값에 해당하는 인덱스의 알파벳을 가져와서 예측한 단어를 출력
predict_words = []
for idx, val in enumerate(seq_data):
    last_char = char_arr[predict[idx]]
    predict_words.append(val[:3] + last_char)

print('\n=== 예측 결과 ===')
print('입력값:', [w[:3] + ' ' for w in seq_data])
print('예측값:', predict_words)
print('정확도:', accuracy_val)


