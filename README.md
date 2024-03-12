## Neural Network
- pytorch 를 사용하지 않고 numpy 를 활용하여 neural network 를 구현한다.
  - FClayer
    - forward
    - backward
  - Optimizer
  - Weight Initialization
  - ... 등 

## Pytorch - 기본 모델 구현
- 기반이 되는 모델을 차근차근 구현한다.
- VGG 구현
  - 왜 7x7, 5x5 보다 3x3 Convolution layer만 사용한 모델이 좋은 성능을 냈을까
  - FC layer 를 사용하지 않고 Convolution layer만 사용하여 모델 구현하기
  - wandb를 사용한 모델의 성능 지표트래킹
