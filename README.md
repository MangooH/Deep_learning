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

## Transformer 구현
 - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) 의 이해
   - [Transformer 이해하기 1](https://enjoy-zero-to-one.tistory.com/69)
     <img width="1094" alt="image" src="https://github.com/MangooH/Deep_learning/assets/88866306/e07f4e55-c8ec-4eb2-9c52-f89ac0b8501a">

   - [Transformer 이해하기 2](https://enjoy-zero-to-one.tistory.com/71)
     <img width="1064" alt="image" src="https://github.com/MangooH/Deep_learning/assets/88866306/ffc7c835-4e04-4c60-b80a-78dfd8d51cb5">

   - [Transformer 이해하기 3](https://enjoy-zero-to-one.tistory.com/77)
     <img width="1095" alt="image" src="https://github.com/MangooH/Deep_learning/assets/88866306/839e3f0d-3948-45f9-bf10-2a11c91ef531">

 - Tokenizer 사용법 이해
 - 기타 Method 사용법 이해
