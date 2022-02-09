# MobileNetV2: Inverted Residuals and Linear Bottlenecks

<br/>

## Abstract

MobileNetV1은 표준 conv 연산의 factorized 버전인 depthwise separable conv 구조를 활용하여 모델 경량화에 집중한 모델. MobileNetV2는 이 depthwise separable conv를 수정한 구조를 제안. 제안된 conv는 **inverted residuals, linear bottleneck**을 활용하여 성능을 향상시킴.

 -  `Inverted residuals` : shorcut-conn이 저차원의 얇은 bottleneck layer에서 수행됨. 중간의 expansion layer는 lightweight depthwise conv를 활용하여 feature를 필터링하는 역할 수행.

 - `Linear bottleneck` :  저차원의 narrow layer에서는 representation power를 최대한 보존하기 위해 비선형성을 제거하는 것이 권당됨.

`Finally, our approach allows decoupling of the in- put/output domains from the expressiveness of the trans- formation, which provides a convenient framework for further analysis.`

<br/>

## 1. Introduction

SOTA로 인식되는 모델 및 네트워크 구조가 필요로하는 연산량이 모바일과 임베디드 어플리케이션 영역이 처리가능한 역량을 뛰어넘는 경우가 많기 때문에, 본 연구는 성능과 정확도를 보존하면서도 필요한 연산량과 메모리를 획기적으로 줄이는 mobile tailored 모델을 구축하며 **inverted residual with linear bottleneck**이라는 새로운 layer 모듈을 제안했다는 점이 main contribution이다.

해당 모듈은 저차원의 compressed representation을 입력으로 받고, 중간 layer에서 고차원의 representation으로 확장되고 depthwise conv 연산을 통해 필터링되는 과정을 거친다. 이 과정을 통해 출력된 feature들은 linear convolution 연산을 통해 다시 저차원의 representation으로 projection된다.

`This module can be efﬁciently implemented using standard operations in any modern framework and allows our models to beat state of the art along multiple performance points using standard benchmarks.`

<br/>

## 2. Related work

딥러닝 네트워크의 성능 향상과 최적화에 대한 많은 연구가 존재하나, 때때로 많은 네트워크들은 매우 복잡하고 방대한 연산을 필요로한다. 따라서 본 연구는 딥러닝 네트워크가 어떻게 동작하는지에 대한 보다 직관적이고 개선된 intuition을 제공함으로써 보다 효율적인 네트워크를 설계하고자 한다.

기본적으로 MobileNetV1에 기반하고 있으며, 해당 모델의 simplicity를 유지하면서도 성능 향상을 목표로한다.

<br/>

## 3. Preliminaries, discussions and intuition

<br/>

### 3.1 Depthwise Separable Convolutions

`The basic idea is to replace a full convolutional opera- tor with a factorized version that splits convolution into two separate layers.`
 
 - `1. Depthwise conv` : 입력에 대해 채널 단위로 lightweight filtering을 적용 - **Spatial Correlation**
  
 - `2. Pointwise conv` : Depthwise conv 연산의 다채널 출력에 1x1 conv를 적용하여 새로운 feature 구축 (입력 채널들에 대한 linear combination을 통해 새로운 feature map 계산) - **Cross-channel Correlation**

MobileNetV2는 3x3 depthwise separable conv를 사용하여, 연산량 측면에서 표준 conv연산보다 약 8~9배 정도로 절감된다.

![image](https://user-images.githubusercontent.com/44194558/153124104-1761cfe0-949a-48de-b4fb-8d581af2edad.png)

<br/>

![image](https://user-images.githubusercontent.com/44194558/153124254-545d2832-8724-4415-8d2a-b8e18b1567d4.png)

 - 입력 채널 각각에 대해(depthwise) conv 연산을 적용한후, 1x1 conv를 통해 다채널 feature들에 대한 linear combination을 수행하여 새로운 feature 계산

<br>

### 3.2 Linear Bottlenecks

딥러닝 네트워크의 각 layer에 대한 [h, w, d]의 activation tensor는 hxw개의 픽셀을 가지고 있는 d 차원의 container라고 가정 (containers of hxw pixels with d dimensions).

Layer activation들의 집합은 `manifold of interest`를 생성하고, 해당 manifold는 저차원의 subspace로 embedding될 수 있다. 즉, 특정 layer에서 d 채널로 구성된 픽셀 집합들을 볼 때, 픽셀 값에 encoding된 정보들은 실질적으로 특정한 manifold위에 놓여있는 것이고, 이 manifold는 저차원의 subspace로 embedding될 수 있다.

 - **Manifold** : 고차원의 데이터가 저차원으로 압축되면서 특정 정보들이 저차원의 어떠한 영역으로 mapping되는데, 이를 manifold라고 함 (딥러닝 네트워크의 manifold는 저차원의 subspace로 mapping가능하다고 가정할 수 있음)

 - `The manifold of interest should lie in a low-dimensional subspace of the higher-dimensional activation space.`

MobileNetV1은 위의 사실을 단순히 layer의 차원을 줄여, operating space의 차원을 감소시킴으로써 활용하고 있다. 하지만, 딥러닝 네트워크가 ReLU같은 비선형 연산, 변환을 포함한다는 것을 고려하면 분명 한계가 있는 접근법이다. ReLU 연산을 구체적으로 살펴보면

 - 1. 어떤 데이터에 관련된 manifold가 ReLU를 통과하고 나서도 입력값이 음수가 아니라서 0이 되지 않은 상태라면, ReLU는 linear transformation 연산을 수행한 것이라고 이해할 수 있음. (Non-zero volume에 대해 identity matrix를 곱하는, 사실상 linear transformation과 다를 바 없다)
    
    - 양수의 값은 그대로 전파하는 linear transformation은 manifold 상의 정보를 그대로 유지한다고 볼 수 있음 

 - 2. 입력이 채널 수가 적은 ReLU 함수 계층을 거치면 정보가 손실되지만, 채널 수가 많은 ReLU 계층을 거치면 정보가 보존된다.

![image](https://user-images.githubusercontent.com/44194558/153125992-c8c6d634-39e7-4e30-b93e-a6f88648bef3.png) 

즉, 저차원으로 mapping하는 bottleneck 구조(projection conv)를 만들 때, linear transformation 역할을 하는 linear bottleneck layer를 만들어서 차원은 줄이되 manifold 상의 중요한 정보들은 그대로 유지하는 것이 컨셉이다. 

따라서, ReLU를 사용할 때는 해당 layer에 비교적 많은 채널 수를 사용하고, 해당 layer에 채널 수가 적다면 선형 함수를 사용해야 한다 (비선형성이 야기할 수 있는 정보 손실 방지).

![image](https://user-images.githubusercontent.com/44194558/153132947-c2c8b275-ec2a-417d-abf1-8844ea147113.png)

<br/>

### 3.3 Inverted residuals

Bottleneck block은 residual block과 달리 narrow-wide-narrow 형태를 띄고 있다. Narrow에 해당하는 저차원의 feature에 필요한 정보가 충분히 압축되어 있기 때문에 skip-conn으로 연결해도 필요한 정보를 깊은 layer까지 전달할 수 있다.

 - `Inspired by the intuition that the bottlenecks actually contain all the necessary information, while an expansion layer acts merely as an implementation detail that accompanies a non-linear transformation of the tensor, we use shortcuts directly between the bottlenecks.`

![image](https://user-images.githubusercontent.com/44194558/153127477-daa4b626-4c7a-4be7-9153-4054b58e855f.png)

<br/>

![image](https://user-images.githubusercontent.com/44194558/153128786-1934c50b-c3ca-4b12-a0a5-1fc766cd7305.png)

 - `Bottlenecks with sufﬁciently large expansion layers are resistant to information loss caused by the presence of ReLU activation functions.`
 
 - Expansion layer에서는 비선형 ReLU6를 사용하지만, projection layer에서는 저차원의 데이터를 출력하기 때문에 비선형 함수를 사용하면 데이터의 유용한 정보가 소실될 수 있음. 


<br/>

![image](https://user-images.githubusercontent.com/44194558/153129343-d5751014-1a9b-4a3a-a178-71632251283f.png)
 
 - Depthwise Separable conv와 비교하면 추가적인 1x1 conv가 추가되었음에도 불구하고 보다 적은 차원의 입력/출력을 활용할 수 있도록 한다 

![image](https://user-images.githubusercontent.com/44194558/153133039-9087ec4a-1500-4418-b600-0f44bfb84a51.png)


<br/>

### 3.4 Information flow interpretation

`One interesting property of our architecture is that it provides a natural separation between the input/output domains of the building blocks (bottleneck layers), and the layer transformation – that is a non-linear function that converts input to the output`

 - bottleneck layer의 입/출력은 네트워크의 capacity, 중간 layer들은 네트워크의 expressiveness와 관련있음

네트워크의 expressiveness를 capacity와 분리하여 연구할 수 있다는 장점이 있음.

`On the theoretical side: the proposed convolutional block has a unique property that allows to separate the network expressiveness (encoded by expansion layers) from its capacity (encoded by bottleneck inputs).`

<br/>

## 4. Model Architecture

`Bottleneck depth-separable conv with residuals`가 네트워크를 구성하는 기본적인 block.

![image](https://user-images.githubusercontent.com/44194558/153130533-7bf477f5-ce5d-405d-bb1a-3652aed31fa5.png)

 - ReLU6로 비선형성을 표현함

<br/>

![image](https://user-images.githubusercontent.com/44194558/153130720-89c6dccb-dd8d-4c08-a33e-5409d8de3937.png)

<br/>

![image](https://user-images.githubusercontent.com/44194558/153130898-e8ebadc1-f05e-479e-b0f4-0df12ba46304.png)

<br/>

## 5. Memory efficient inference

메모리에 저장되어야 하는 텐서의 총 수를 줄이는 최적화 문제로 이해 가능. 그래프 관점에선 에지가 특정 operation, 노드는 그 operation의 출력 텐서.

![image](https://user-images.githubusercontent.com/44194558/153132051-8639625b-be59-4258-aaa9-be356a587ed7.png)

 - R : the list of intermediate tensors
 - pi : nodes -> size(pi) : 연산 수행시 저장해야 하는 텐서의 크기

<br/>

MobileNetV2의 메모리는

![image](https://user-images.githubusercontent.com/44194558/153132399-5cd35378-70eb-40b6-a53f-72a1614f0d13.png)

 - 중간 연산은 memory와 무관하므로 따로 저장되지 않고, 저차원 채널의 입/출력이 memory에 저장됨
 - `Total size of combined inputs and outputs across all operations ... and treat inner convolution as a disposable tensor` 
