# Sequence to Sequence는 구글이 기계 번역에 사용하는 신경망 모델
# 순차적으로 정보를 입력 받는 RNN과 출력하는 신경망을 조합한 모델
# 번역이나 챗봇 등 문장을 입력받아 다른 문장을 출력하는 프로그램에서 많이 사용

# Sequence to Sequence 모델에는 디코더에 입력이 시작됨을 알려주는 심볼, 디코더에 출력이 끝났음을 알려주는 심볼, 빈 데이터를 채울 때 사용하는 아무 의미 없는 심볼이 필요
# 여기서는 해당 심볼들을 'S', 'E', 'P'로 처리

########
# 데이터 만들기
########

# 글자들을 학습시키기 위해서는 원-핫 인코딩 형식으로 바꿔야 하므로 영어 알파벳과 한글들을 나열한 뒤 한 글자씩 배열에 집어넣음
# 그런 다음 배열에 넣은 글자들을 연관 배열(키/값 쌍) 형태로 변경

import tensorflow as tf
import numpy as np

char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

seq_data = [['word', '단어'], ['wood', '나무'],
            ['game', '놀이'], ['girl', '소녀'],
            ['kiss', '키스'], ['love', '사랑']]


# 입력 단어와 출력 단어를 한 글자씩 떼어낸 뒤 배열로 만든 후
# 원-핫 인코딩 형식으로까지 만들어주는 유틸리티 함수를 만듦
# 데이터는 인코더의 입력값, 디코더의 입력값과 출력값, 이렇게 총 세 개로 구성됨

def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        # 인코더 셀의 입력값. 입력단어의 글자들을 한글자씩 떼어 배열로 만든다.
        input = [num_dic[n] for n in seq[0]]
        # 디코더 셀의 입력값. 시작을 나타내는 S 심볼을 맨 앞에 붙여준다.
        output = [num_dic[n] for n in ('S' + seq[1])]
        # 학습을 위해 비교할 디코더 셀의 출력값. 끝나는 것을 알려주기 위해 마지막에 E 를 붙인다.
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        # 출력값만 one-hot 인코딩이 아님 (sparse_softmax_cross_entropy_with_logits 사용)
        target_batch.append(target)

    return input_batch, output_batch, target_batch


# 신경망 모델에서 사용할 하이퍼퍼라미터, 플레이스홀더, 입출력 변수용 수치들을 정의
learning_rate = 0.01
n_hidden = 128
total_epoch = 100

n_class = n_input = dic_len


# 인코더의 입력값, 디코더의 입력값과 출력값에 사용할 플레이스홀더를 구성
# 인코더와 디코더의 입력값: [batch_size, time steps, input size]
# 디코더 출력값: [batch size, time steps]


# 신경망 모델 구성
# RNN의 특성 상 입력 데이터에 단계가 있으며 입력값들은 원-핫 인코딩을 사용하고 디코더의 출력값은 인덱스 숫자를 그대로 사용하기 때문에
# 입력값의 랭크(차원)가 하나 더 높음
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets = tf.placeholder(tf.int64, [None, None])


# 입력 단계는 배치 크기처럼 입력받을 때마다 다를 수 있으므로 None으로 설정
# 단, 같은 배치 때 입력되는 데이터는 글자 수, 즉 time steps가 모두 같아야 함


# RNN 모델을 위한 셀 구성
# 인코더 셀과 디코더 셀을 만들어야 함

with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)

    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)

with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state=enc_states, dtype=tf.float32)

# 셀은 기본셀을 사용하였고 각 셀에 드롭아웃을 적용 함
# 디코더를 만들 때 초기 상태 값(입력 값이 아님)으로 인코더의 최종 상태 값을 넣어줘야 함


# 출력층을 만들고 손실 함수와 최적화 함수를 구성
model = tf.layers.dense(outputs, n_class, activation=None)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# 학습 시키기

sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, output_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={enc_input: input_batch,
                                  dec_input: output_batch,
                                  targets: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),
          'cost =', '{:.6f}'.format(loss))

print('최적화 완료!')


# 결과를 확인하기 위해 단어를 입력받아 번역 단어를 예측하는 함수를 만들기
def translate(word):
    # 이 모델은 입력값과 출력값 데이터로 [영어단어, 한글단어] 사용하지만,
    # 예측시에는 한글단어를 알지 못하므로, 디코더의 입출력값을 의미 없는 값인 P 값으로 채운다.
    # ['word', 'PPPP']
    seq_data = [word, 'P' * len(word)]

    input_batch, output_batch, target_batch = make_batch([seq_data])

    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.
    prediction = tf.argmax(model, 2)

    result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                                 dec_input: output_batch,
                                 targets: target_batch})

    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    decoded = [char_arr[i] for i in result[0]]

    # 출력의 끝을 의미하는 'E' 이후의 글자들을 제거하고 문자열로 만든다.
    end = decoded.index('E')
    translated = ''.join(decoded[:end])

    return translated


print('\n=== 번역 테스트 ===')

print('word ->', translate('word'))
print('wodr ->', translate('wodr'))
print('love ->', translate('love'))
print('loev ->', translate('loev'))
print('abcd ->', translate('abcd'))



