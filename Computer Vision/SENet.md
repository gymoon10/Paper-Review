# SENet: Squeeze and Excitation Networks

<br/>

## Abstract

CNN은 각 layer에서 local receptive field를 사용하여 spatial-wise, channel(depth)-wise 정보를 추출하며, 그 정보들을 결합(fuse)하여 유용한 representation(informative features)를 생성한다. 그리고 대다수의 선행 연구들은 CNN의 계층적 구조에서 spatial encoding을 개선 함으로써 CNN 네트워크의 representation power와 성능을 향상시키고자 했다.

본 연구는 SEblock의 개념을 통해 channel relationship에 주목한다. 

 - `“Squeeze-and-Excitation” (SE) block, that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.`
 
 - 입력 feature map의 global information을 요약하고, 채널 별 중요도를(가중치) 반영하여 기존 네트워크의 feature map을 재조정하는(recallibration) 역할을 수행
 
 - 정보의 압축과 재조정 

<br/>

## 1. Introduction

CNN 네트워크의 개별 layer에서 합성곱 연산에 사용되는 filter들의 집합은 입력 채널들 간의 spatial connectivity를 표현하고, local receptive field를 사용하여 spatial, channel wise 정보를 결합한다.

 - `By interleaving a series of convolutional layers with non-linear activation functions and downsampling operators, CNNs are able to produce image representations that capture hierarchical patterns and attain global theoretical receptive ﬁelds.`

실제로 컴퓨터 비전 분야의 주된 이슈는 이미지로부터 주어진 태스크에 가장 적합한 representation을 찾는 것이며, 많은 연구들이 CNN의 representation power를 개선하고자 노력해왔다.

본 연구는 기존 연구들과는 달리 channel relationship을 고려하여 새로운 네트워크 설계를 제안한다. 

 - `The Squeeze-and-Excitation (SE) block, with the goal of improving the quality of representations produced by a network by explicitly modelling the interdependencies between the channels of its convolutional features.`

 - 재조정 (feature recallibration) 과정에서 global information을 활용하여 (채널별로) informative feature들의 중요도를 반영하게 된다.

<br/>

![image](https://user-images.githubusercontent.com/44194558/153327880-71533477-b8fe-45f4-8bbc-ba13cc9ca449.png)

 - `Squeeze` : **channel descriptor** (embedding of the global dist of channel wise feature responses) 계산 (HxW의 spatial dimension을 따라 feature map을 aggregate)
 
 - `Exciatation` : takes the embedding as input and produces a collection of per-channel modulation weights (channel descriptor의 채널 별 중요도 계산).
                   해당 가중치는 feature map U에 곱해져 새로운(enhanced) feature map을 생성   


SE block은 초기 layer에서는 일반적이고(class-agnostic), low level representation을 excitation하고, 출력에 가까운 layer에서는 보다 class specific한 representation을 excitation한다. 즉, SE block에 의한 feature recallibration의 효과는 네트워크의 layer들을 거치며 축적된다. 또한, 모델의 복잡도를 크게 증가시키지 않으면서도 유의미한 성능 개선을 이끌어낸다는 장점이 있다.

<br/>

## 2. Related Work

<br/>

### Deeper architectures

VGGNet, Inception 모듈은 네트워크의 깊이를 깊게 하는 것이 유의미한 성능 개선을 가져온다는 것을 보였고, Batch Normalization은 깊은 네트워크에서 안정적인 학습을 가능하게 한다. 이에 따라 네트워크의 representation power를 향상시키기 위해 다양한 네트워크들이 제안되었다.

`In contrast, we claim that providing the unit with a mechanism to explicitly model dynamic, non-linear dependencies between channels using global information can ease the learning process, and signiﬁcantly enhance the representational power of the network.`

<br/>

### Attention and gating mechanisms

`Attention can be interpreted as a means of biasing the allocation of available computational resources towards the most informative components of a signal.`

 - feature map에서 보다 중요한 channel의 정보에 초점을 맞출 수 있도록

SE block은 lightweight gating mechanism을 활용하여 channel-wise relationship을 효율적으로 모델링함으로써 네트워크의 representational power를 향상시킨다.

<br/>

## 3. Squeeze and Excitation Blocks

입력 X에 특정 필터 v_c를 적용하여 새로운 feature map U를 계산하는 일반적인 conv 연산은 다음과 같이 표현


![image](https://user-images.githubusercontent.com/44194558/153334640-be7f5e58-4595-4de3-b8ad-d479898761f6.png)

 - 출력 U는 모든 채널에 대한 summation을 통해 계산
  
 - channel dependency는 특정 필터 v_c에 embedding되어 있지만, local spatial correlation과 섞여있다 (entangled).

`We expect the learning of convolutional features to be enhanced by explicitly modelling channel interdependencies, so that the network is able to increase its sensitivity to informative features which can be exploited by subsequent transformations.` 

 - `Squeeze` : acces to global information
 
 - `Excitation` : recallibrate filter responses

<br/>

### 3.1 Squeeze: Global Information Embedding

Conv 연산에서 각각의 학습된 filter들은 local receptive field와 함께 연산되기 때문에, 연산의 output U는 해당 local receptive field의 영역을 벗어난 global한 contextual information은 충분히 활용하지 못하는 문제가 있다.

이러한 문제를 해결하기 위해 global spatial information을 압축(squeeze)하여 channel descriptor로 변환한다. 이는 global average pooling을 통해 channel-wise 통계량을 계산함으로써 얻을 수 있다 (가장 간단한 aggregation 방법으로 GAP를 채택). 채널 별 통계량 z (input specific descriptor)는 U를 spatial dimension HxW를 따라 shrink함으로써 계산 가능하다.

![image](https://user-images.githubusercontent.com/44194558/153335667-407501ee-7309-46a4-9a3b-0ab72452f645.png)

 - feature map U의 squeeze 출력은 local descriptor들의 집합
 
 - descriptor의 통계량은 전체 이미지에 대해 expressive함 

<br/>

### 3.2 Excitation: Adaptive Recallibration

Squeeze 연산을 통해 aggregation된 정보를 활용하기 위해, Excitation 연산을 활용하여 channel-wise dependency를 계산하는 것이 목적이다. 이 목적을 위해 2 가지의 기준이 충족되어야 하는데

 1. 유연함 - `It must be capable of learning a nonlinear interaction btw channels`
 
 2. Non-mutually exclusive relationship을 학습 : 단순히 하나가 아닌 여러 개의 채널들이 강조될 수 있어야 함 (<-> one hot activation)

위의 기준들을 만족하기 위해 sigmoid를 활용한 gating mechanism을 활용함.

![image](https://user-images.githubusercontent.com/44194558/153336604-66d4e700-e960-4d08-9417-8ae6daac62fe.png)

 - Input specific descripor z를 channel별 가중치로 변환

![image](https://user-images.githubusercontent.com/44194558/153336650-e1d3f322-6906-4c9e-b54f-b999245ed35e.png)

 - F_scale : channel wise multiplication (채널 별 가중치를 고려)

`SE blocks intrinsically introduce dynamics conditioned on the input, which can be regarded as a self-attention function on channels whose relationships are not conﬁned to the local receptive ﬁeld the convolutional ﬁlters are responsive to.`

<br/>

### 3.3 Instantiations

`The SE block can be integrated into standard architectures such as VGGNet [11] by insertion after the non-linearity following each convolution.`

 - SE block의 유연성 - 여러 딥러닝 아키텍쳐에 다양한 방식으로 통합될 수 있음



![image](https://user-images.githubusercontent.com/44194558/153337251-29a379d6-45f4-462f-b8d6-bb48e0e2d018.png)


<br/>

## Conclusion

`In this paper we proposed the SE block, an architectural unit designed to improve the representational power of a network by enabling it to perform dynamic channel-wise feature recalibration.`