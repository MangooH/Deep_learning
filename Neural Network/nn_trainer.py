import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class Trainer:
    def __init__(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
        model,
        optimizer,
        max_iterations,
        batch_size,
    ):

        # 데이터셋
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        # 모델
        self.model = model

        # 옵티마이저
        self.optimizer = optimizer

        self.train_size = X_train.shape[0]
        self.batch_size = batch_size
        self.max_iterations = max_iterations

        self.train_loss_list = []
        self.train_acc_list = []
        self.valid_loss_list = []
        self.valid_acc_list = []

    def run(
        self,
    ):
        """신경망과 옵티마이저를 이용해서 학습하는 함수"""
        start_time = datetime.datetime.now()

        for i in range(self.max_iterations):
            ####################################################################
            # 학습
            ####################################################################
            batch_mask = np.random.choice(self.train_size, self.batch_size)
            X_batch = self.X_train[batch_mask]
            y_batch = self.y_train[batch_mask]

            # 예측 수행
            y_pred = self.model.predict(X_batch, train_flag=True)

            # 손실 계산
            train_loss = self.model.loss(y_batch, y_pred)
            self.train_loss_list.append(train_loss)

            # 정확도 계산
            train_acc = accuracy(y_batch, y_pred, self.batch_size)
            self.train_acc_list.append(train_acc)

            # Gradient 계산
            grads = self.model.gradient()

            # Optimizer 업데이트
            self.optimizer.update(self.model.params, grads)

            ####################################################################
            # 검증
            ####################################################################
            batch_mask = np.random.choice(self.X_valid.shape[0], self.batch_size)
            X_batch = self.X_valid[batch_mask]
            y_batch = self.y_valid[batch_mask]

            # 예측 수행
            y_pred = self.model.predict(X_batch, train_flag=False)

            # 손실 계산
            valid_loss = self.model.loss(y_batch, y_pred)
            self.valid_loss_list.append(valid_loss)

            # 정확도 계산
            valid_acc = accuracy(y_batch, y_pred, self.batch_size)
            self.valid_acc_list.append(valid_acc)

            if i % 100 == 0:
                elpased_time = datetime.datetime.now() - start_time
                msg = f"\033[31m[Elpased Time: {elpased_time}]\033[0m "
                msg += f"Iter: {i:>4} "
                msg += f"Train Loss : {train_loss:.4f} "
                msg += f"Train Acc : {train_acc:.2f} "
                msg += f"Valid Loss : {valid_loss:.4f} "
                msg += f"Valid Acc : {valid_acc:.2f} "
                print(msg)

    def show_results(
        self,
    ):
        """학습된 결과를 시각화해주는 함수"""
        plt.figure(figsize=(16, 8))

        plt.subplot(2, 1, 1)
        plt.plot(self.train_loss_list)
        plt.plot(self.valid_loss_list)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend(["Train", "Valid"])
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(self.train_acc_list)
        plt.plot(self.valid_acc_list)
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend(["Train", "Valid"])
        plt.ylim(0, 1.05)
        plt.grid(True)

        plt.suptitle(f"Result - Loss & Acc", fontsize=20)
        plt.tight_layout()
        plt.show()


def accuracy(y_true, y_pred, batch_size):
    """
    정확도를 계산합니다.
    """
    y_pred = np.argmax(y_pred, axis=1)

    if y_pred.ndim != 1:
        y_pred = np.argmax(y_pred, axis=1)

    assert y_true.shape == y_pred.shape

    # 실제 라벨과 예측 라벨이 일치하는 개수를 계산하여 정확도 반환
    return np.sum(y_true == y_pred) / float(batch_size)
