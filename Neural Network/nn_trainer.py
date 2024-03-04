import numpy as np
from collections import OrderedDict


class Sigmoid:
    """
    시그모이드 레이어
    """

    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


def sigmoid(x):
    """시그모이드 함수"""
    return 1 / (1 + np.exp(-x))


class Relu:
    """
    ReLU 레이어
    """

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class FCLayer:
    """완전 연결 레이어"""

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """완전 연결 레이어의 순전파(forward propagation)"""
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = (
            np.dot(self.x, self.W) + self.b
        )  # [n x before_layer_size] * [before_layer_size x after_layer_size] + [1 x after_layer_size]

        return out

    def backward(self, dout):
        """완전 연결 레이어의 역전파(backpropagation)"""
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)

        return dx


# BatchNormalization 클래스: 신경망의 학습을 안정화하고 가속화하는 Batch Normalization 기법을 구현합니다.
class BatchNormalization:

    # 초기화 함수
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):

        # gamma, beta: 학습 가능한 스케일 및 시프트 파라미터
        # 정규분포에 약간의 변형을 가함
        # -> 모델 성능 극대화를 위해 선형 변형은 학습을 통해 익히도록 한다.
        self.gamma = gamma
        self.beta = beta

        # momentum: 평균 및 분산의 움직이는 평균을 계산할 때 사용되는 모멘텀 값
        self.momentum = momentum

        # 학습 중 아닌 상황(예: 평가)에서 사용할 실행 중 평균 및 분산
        # 추론 시에는 배치크기가 1이거나 예측하려는 샘플 수에 따라 다양하기 때문에
        # 전체 학습 데이터셋을 대표하는 일반적인 평균과 분산의 추정치를 계산한다.
        self.running_mean = running_mean
        self.running_var = running_var

        # 입력 데이터의 형태를 저장 (예: (batch_size, features))
        self.input_shape = None

        # 역전파 시 사용될 중간 값들
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    # 순전파 함수
    def forward(self, x, train_flag=True):
        self.input_shape = x.shape

        # 4D 텐서인 경우 2D로 변경
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        # running_mean 및 running_var 초기화
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        # 학습 시
        if train_flag:

            # 현재 배치의 평균 및 분산 계산
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            # 중간 값들 저장
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std

            # 실행 중 평균 및 분산 업데이트
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )

        # 평가 시
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        # 최종 출력 계산
        out = self.gamma * xn + self.beta

        return out.reshape(*self.input_shape)

    # 역전파 함수
    def backward(self, dout):

        # 4D 텐서인 경우 2D로 변경
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        # 역전파 계산을 위한 그래디언트들
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        # 학습 가능한 파라미터들의 그래디언트 저장
        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx.reshape(*self.input_shape)


class Dropout:
    """
    IDEA: 학습이 잘 된 모델이라면, 모델 중 일부 노드를 비활성화시켜도 여전히 좋은 성능을 유지할 것이다.
    과적합을 방지하기 위한 Dropout 을 구현.
    - 일부 노드에서 학습이 진행되지 않게끔 한다.
    """

    def __init__(self, dropout_ratio=0.5):
        # drop_ratio: 드롭아웃할 뉴런의 비율
        # 0.5 는 50% 의 뉴런을 무작위로 비활성화시킴
        self.dropout_ratio = dropout_ratio

        # mask: 드롭아웃할 뉴런을 결정하는 불리언 마스크
        self.mask = None

    def forward(self, x, train_flag=True):
        if train_flag:

            # 입력 데이터 x 와 동일한 모양의 무작위 배열을 생성하고,
            # dropout_ratio 보다 큰 값만 True 로 설정
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio

            # mask 를 사용해 x 의 일부 뉴런을 꺼버림.
            return x * self.mask

        else:
            # Dropout의 비율만큼 스케일 조정하여 출력
            return x * (1.0 - self.dropout_ratio)

    # 역전파 함수
    def backward(self, dout):

        # mask를 사용하여, 순전파 때 꺼진 뉴런은 그래디언트도 전달되지 않게 함.
        return dout * self.mask


class Softmax:
    """소프트맥스 레이어"""

    def __init__(self):
        self.loss = None
        self.y_true = None
        self.y_pred = None

    def forward(self, x, y_pred):
        """소프트맥스 레이어의 순전파(forward propagation)"""
        self.y_true = softmax(x)
        self.y_pred = y_pred
        self.loss = cross_entropy_error(self.y_true, self.y_pred)

        return self.loss

    def backward(self, dout=1):
        """소프트맥스 레이어의 역전파(backpropagation)"""
        batch_size = self.y_pred.shape[0]
        if self.y_pred.size == self.y_true.size:
            dx = (self.y_true - self.y_pred) / batch_size
        else:
            dx = self.y_true.copy()
            dx[np.arange(batch_size), self.y_pred] -= 1
            dx = dx / batch_size

        return dx


def softmax(x):
    """소프트맥스 함수"""
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)

    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y_true, y_pred):
    if y_true.ndim == 1:
        y_pred = y_pred.reshape(1, y_pred.size)
        y_true = y_true.reshape(1, y_true.size)

    if y_pred.size == y_true.size:
        y_pred = y_pred.argmax(axis=1)

    batch_size = y_true.shape[0]

    return -np.sum(np.log(y_true[np.arange(batch_size), y_pred] + 1e-7)) / batch_size


class Net:
    def __init__(
        self,
        input_size,
        hidden_size_list,
        output_size,
        use_dropout=False,
        dropout_ratio=0,
        use_batchnorm=False,
        activation="relu",
        weight_init_std="relu",
        weight_decay_lambda=0,
    ):
        # 네트워크의 초기화
        self.input_size = input_size  # 입력 크기 (예: 이미지의 픽셀 수)
        self.output_size = output_size  # 출력 크기 (예: 분류할 클래스 수)
        self.hidden_size_list = hidden_size_list  # 은닉층의 뉴런 수 리스트
        self.hidden_layer_num = len(hidden_size_list)  # 은닉층의 개수
        self.weight_decay_lambda = weight_decay_lambda  # 가중치 감쇠
        self.use_dropout = use_dropout  # 다롭아웃 사용 여부
        self.use_batchnorm = use_batchnorm  # 배치 정규화 사용 여부

        # 신경망의 가중치
        self.params = {}

        # 가중치 초기화
        self.__init_weight(weight_init_std)

        # 활성화 함수 지정 (ReLU, Sigmoid 등)
        activation_layer = {"sigmoid": Sigmoid, "relu": Relu}

        # 신경망의 레이어
        self.layers = OrderedDict()

        for idx in range(1, self.hidden_layer_num + 1):

            # 완전 연결 레이어 (Fully connected layer)
            self.layers[f"FC{idx}"] = FCLayer(
                self.params[f"W{idx}"], self.params[f"b{idx}"]
            )

            # 배치 정규화 사용 여부에 따른 레이어 추가.
            if self.use_batchnorm:
                self.params[f"gamma{idx}"] = np.ones(hidden_size_list[idx - 1])
                self.params[f"beta{idx}"] = np.zeros(hidden_size_list[idx - 1])
                self.layers[f"BN{idx}"] = BatchNormalization(
                    self.params[f"gamma{idx}"], self.params[f"beta{idx}"]
                )

            # 활성화 레이어 (Activation Layer: 예를 들면 ReLU나 Sigmoid)
            self.layers[f"Act{idx}"] = activation_layer[activation]()

            # 드롭아웃 사용 여부에 따른 레이어 추가
            if self.use_dropout:
                self.layers[f"Dropout{idx}"] = Dropout(self.dropout_ratio)

        # 출력층 (last layer)
        idx = self.hidden_layer_num + 1
        self.layers[f"FC{idx}"] = FCLayer(
            self.params[f"W{idx}"], self.params[f"b{idx}"]
        )

        # 소프트 맥스 함수를 마지막 계층으로 설정 (분류 문제이므로)
        self.last_layer = Softmax()

    def __init_weight(self, weight_init_std):
        """
        신경망의 가중치를 초기화.
        Xavier 방식과 He 방식을 사용한다.
        """

        # 전체 네트워크의 각 층의 뉴런 수를 리스트로 구성
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        # 모든 층을 순회하며 가중치를 초기화
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std

            # ReLU 활성화 함수를 사용할 경우 He 초기화 방법 사용
            # 표준편차가 sqrt(2/n)인 정규분포
            # -> ReLU가 음수 값을 모두 0으로 만들기 때문에, sigmod 보다 분산을 조금 더 크게(x2) 해서 그래디언트 소실 문제를 완화
            if str(weight_init_std).lower() in ("relu", "he"):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])

            # Sigmoid 활성화 함수를 사용할 경우 Xavier 초기화 방법 사용
            elif str(weight_init_std).lower() in ("sigmoid", "xavier"):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])

            # 가중치와 편향 초기화
            self.params[f"W{idx}"] = scale * np.random.randn(
                all_size_list[idx - 1], all_size_list[idx]
            )  # [이전 layer 개수 x 현재 layer 개수]
            self.params[f"b{idx}"] = np.zeros(all_size_list[idx])
