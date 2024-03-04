import numpy as np


class SGD:
    """SGD 옵티마이저"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:

    # 생성자: 학습률(learning rate) 및 모멘텀 값 설정
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr  # 학습률 설정
        self.momentum = momentum  # 모멘텀 설정
        self.v = None  # 속도 초기화

    # 파라미터 업데이트 함수
    def update(self, params, grads):

        # 첫 번째 호출 시 속도(v)를 파라미터와 동일한 형상의 0으로 초기화
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        # 모든 파라미터에 대해
        for key in params.keys():

            # 속도 업데이트
            self.v[key] = self.momentum * self.v[key] + self.lr * grads[key]

            # 파라미터 업데이트
            params[key] -= self.v[key]


class Nesterov:

    # 학습률(learning rate) 및 모멘텀 계수 설정
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr  # 학습률
        self.momentum = momentum  # 모멘텀 계수
        self.v = None  # 속도

    # 파라미터 업데이트 함수
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        # 각 파라미터에 대해
        for key in params.keys():

            # 현재 속도를 기반으로 예측된 위치 계산
            w_pred = params[key] + self.momentum * self.v[key]

            # 예측된 위치에서의 그래디언트 계산
            # 여기서는 간단하게 현재 위치에서의 그래디언트를 사용합니다.
            g = grads[key]

            # 속도 업데이트
            self.v[key] = self.momentum * self.v[key] - self.lr * g

            # 파라미터 업데이트
            params[key] += self.v[key]


class AdaGrad:

    # 학습률(learning rate) 설정
    def __init__(self, lr=0.01):
        self.lr = lr  # 학습률 설정
        self.h = None  # 이전 기울기의 제곱 합을 저장할 변수 초기화

    # 파라미터 업데이트 함수
    def update(self, params, grads):

        # 첫 번째 호출 시 h를 파라미터와 동일한 형상의 0으로 초기화
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        # 각 파라미터에 대해
        for key in params.keys():

            # 기울기의 제곱 합을 h에 누적
            self.h[key] += grads[key] * grads[key]

            # 파라미터 업데이트 (AdaGrad 특징 부분)
            # 0으로 나누는 것을 방지하기 위한 작은 상수 추가
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop:

    # 학습률(learning rate)와 감쇠율(decay rate) 설정
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr  # 학습률 설정
        self.decay_rate = decay_rate  # 감쇠율 설정
        self.h = None  # 이전 기울기의 제곱 합의 이동 평균을 저장할 변수 초기화

    # 파라미터 업데이트 함수
    def update(self, params, grads):

        # 첫 번째 호출 시 h를 파라미터와 동일한 형상의 0으로 초기화
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        # 각 파라미터에 대해
        for key in params.keys():

            # h를 감쇠율로 감소시키고
            self.h[key] *= self.decay_rate

            # 기울기의 제곱의 (1-감쇠율) 비율을 더한다.
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]

            # 파라미터 업데이트 (RMSprop 특징 부분)
            # 0으로 나누는 것을 방지하기 위한 작은 상수 추가
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:

    # Adam의 기본 하이퍼파라미터들을 초기화
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr  # 학습률
        self.beta1 = beta1  # 모멘텀에 사용되는 계수
        self.beta2 = beta2  # RMSprop에 사용되는 계수
        self.iter = 0  # 반복 횟수 저장용
        self.m = None  # 1차 모멘텀용 누적 값
        self.v = None  # 2차 모멘텀용 누적 값 (제곱된 기울기의 이동 평균)

    # 파라미터 업데이트 함수
    def update(self, params, grads):

        # 첫 번째 호출 시 m, v 초기화
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        # 반복 횟수 증가
        self.iter += 1

        # 학습률의 바이어스 보정 (bias correction)
        # 이는 초기 불안정한 학습을 안정적으로 도와줌
        lr_t = (
            self.lr
            * np.sqrt(1.0 - self.beta2**self.iter)
            / (1.0 - self.beta1**self.iter)
        )

        # 각 파라미터에 대해
        for key in params.keys():

            # 1차 모멘텀 계산
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])

            # 2차 모멘텀 (제곱된 기울기의 이동 평균) 계산
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            # 파라미터 업데이트
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
