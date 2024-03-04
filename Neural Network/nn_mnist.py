import os
import gzip
import pickle
import numpy as np
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import random

# MNIST를 다운받을 경로
URL = "http://yann.lecun.com/exdb/mnist/"

# 재현성을 위해 랜덤 시드를 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


class MNIST:

    def __init__(self):
        # MNIST 데이터셋의 파일명 (딕셔너리)
        self.key_file = {
            "train_img": "train-images-idx3-ubyte.gz",
            "train_label": "train-labels-idx1-ubyte.gz",
            "test_img": "t10k-images-idx3-ubyte.gz",
            "test_label": "t10k-labels-idx1-ubyte.gz",
        }

        # MNIST를 저장할 디렉토리 (`./data/`)
        self.dataset_dir = os.path.join(os.getcwd(), "data")

        # Pickle로 저장할 경로
        # 딕셔너리, 리스트, 클래스 등의 자료형을 변환 없이 그대로 파일로 저장하고 이를 불러올 때 사용하는 모듈
        self.save_file = self.dataset_dir + "/mnist.pkl"

        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None

    def download_dataset(self):

        # 해당 경로가 없을 시 디렉토리 새로 생성
        os.makedirs(self.dataset_dir, exist_ok=True)

        # 해당 경로에 존재하지 않는 파일을 모두 다운로드
        for filename in self.key_file.values():
            if filename not in os.listdir(self.dataset_dir):
                urlretrieve(URL + filename, os.path.join(self.dataset_dir, filename))
                print("Downloaded %s to %s" % (filename, self.dataset_dir))

    def load_mnist(self, normalize=True, flatten=True, one_hot_label=False):
        """
        MINST 데이터셋 읽기
        """

        # Pickle 화 되어있는지 확인
        if not os.path.exists(self.save_file):
            self._init_mnist()

        # Pickle 화 된 MNIST 데이터셋 가져오기
        with open(self.save_file, "rb") as f:
            dataset = pickle.load(f)

        # 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화
        if normalize:
            for key in ("train_img", "test_img"):
                dataset[key] = dataset[key].astype(np.float32)
                dataset[key] /= 255.0

        # 레이블을 원-핫(one-hot) 배열로 변환
        if one_hot_label:
            dataset["train_label"] = self._change_one_hot_label(dataset["train_label"])
            dataset["test_label"] = self._change_one_hot_label(dataset["test_label"])

        # 입력 이미지를 1차원 배열로 만듦
        if not flatten:
            for key in ("train_img", "test_img"):
                dataset[key] = dataset[key].reshape(-1, 1, 28, 28)  # N, C, W, H

        self.X_train = dataset["train_img"]
        self.y_train = dataset["train_label"]
        self.X_test = dataset["test_img"]
        self.y_test = dataset["test_label"]

    def suffle_validation(self):
        """
        학습 데이터 중 일부를 검증 데이터로 활용
        """

        # 학습 데이터의 인덱스를 랜덤하게 셔플
        indices = np.arange(self.X_train.shape[0])
        np.random.shuffle(indices)

        # 섞인 인덱스를 기반으로 train과 valid 데이터 분할
        valid_idx = indices[:10000]
        train_idx = indices[10000:]

        self.X_valid, self.y_valid = self.X_train[valid_idx], self.y_train[valid_idx]
        self.X_train, self.y_train = self.X_train[train_idx], self.y_train[train_idx]

    def dataset_visualization(self):
        plt.figure(figsize=(7, 7))
        for n, i in enumerate(
            np.random.randint(0, len(self.X_train), size=16), start=1
        ):
            plt.subplot(4, 4, n)
            plt.imshow(self.X_train[i].reshape(28, 28), cmap="gray")
            plt.title(f"Label: {self.y_train[i]}", fontsize=14)
            plt.axis("off")

        plt.suptitle("MNIST Dataset", fontsize=20)
        plt.tight_layout()
        plt.show()

    def _init_mnist(self):
        """
        MNIST 데이터셋을 Pickle 화
        """
        dataset = self._convert_numpy()
        with open(self.save_file, "wb") as f:
            pickle.dump(dataset, f, -1)

    def _convert_numpy(self):
        """
        Numpy array 로 불러온 MNIST 데이터셋을 딕셔너리로 매핑
        """
        dataset = {}
        dataset["train_img"] = self._load_img(self.key_file["train_img"])
        dataset["train_label"] = self._load_label(self.key_file["train_label"])
        dataset["test_img"] = self._load_img(self.key_file["test_img"])
        dataset["test_label"] = self._load_label(self.key_file["test_label"])
        return dataset

    def _load_img(self, file_name):
        """
        MNIST 데이터셋 이미지를 Numpy array 로 변환하여 불러오기
        """
        file_path = self.dataset_dir + "/" + file_name
        with gzip.open(file_path, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28, 28)
        return data

    def _load_label(self, file_name):
        """
        MNIST 데이터셋 라벨을 Numpy array 로 변환하여 불러오기
        """
        file_path = self.dataset_dir + "/" + file_name
        with gzip.open(file_path, "rb") as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _change_one_hot_label(self, labels):
        T = np.zeros((labels.size, 10))
        for idx, row in enumerate(T):
            row[labels[idx]] = 1
        return T
