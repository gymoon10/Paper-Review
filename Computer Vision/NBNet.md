# NBNet : Noise Basis Learning for Image Denoising with Subspace Projection

<br/>

## Abstract

이미지 디노이징을 위한 새로운 프레임 워크 NBNet은 이전 연구들과 달리 디노이징 문제를 **image-adaptive projection**의 관점을 통해 해결하고자 한다. (feature space의 reconstruction 기저 벡터를 찾고, 이에 대한 projection을 통해 신호와 노이즈를 분리하는 방법을 학습하고자 함)

이미지 디노이징은 signal subspace의 기저 벡터를 탐색하고, **기저 벡터의 span으로 생성되는 signal subspace에 입력을 projection** 함으로써 수행될 수 있다. 가장 중요한 인사이트는 projection이 입력 signal의 전반적인 구조와 정보를 보다 효율적으로 보존한다는 점에 있다.

기저 벡터의 생성과 subspace projection은 non-local한 SSA (SubSpace Attention) 모듈을 통해 이루어진다. NBNet은 기본적으로 UNet 기반이며, 여기에 SSA가 추가된 구조이다.

<br/>

## 1. Introduction

이미지 디노이징은 기본적으로 additive noise n이 존재할 때, noisy observation y에서 clean signal x를 복원하는 것이다. 

![image](https://user-images.githubusercontent.com/44194558/152287307-5e78a9f2-4df7-4686-a5ff-06f8b87586a0.png)

하지만 위의 식으로 표현되는 문제는 전형적인 ill-posed이기 때문에 많은 디노이징 방법들이 image prior, noise model을 활용한다. 실제로 최근의 많은 딥러닝 방법들이 (깨끗한 이미지 - 노이즈 이미지)의 pair로 구성된 학습 데이터로부터 학습된 image prior와 noise distribution을 활용한다. 단, CNN은 신호와 노이즈를 구별할 때 local filter response만 고려한다는 단점이 있고, 이러한 local filter response는 global structure information을 충분히 반영하지 못한다. 

따라서 본 연구는 projection을 통해 non-local image information을 적극적으로 활용한다. 이미지에 대한 기저 벡터들은 입력으로 부터 생성되고, 기저 벡터들에 의해 생성되는 subspace를 통해 디노이즈된 이미지를 복원한다.

`By properly learning and generating the basis vectors, the re- constructed image can keep most original information and suppress noise which is irrelevant to the generated basis set.` 

![image](https://user-images.githubusercontent.com/44194558/152288033-8e744af3-e178-4563-8374-41bf5d81ae6b.png)

**SSA**는 CNN과 같이 활용됨으로써 기저 벡터의 생성과 projection을 학습하도록 design.

<br/>

## 2. Related Works

디노이징 문제에 있어 대량의 (깨끗한 이미지 - 노이즈 이미지) pair로 구성된 데이터들이 필요한데, 이를 위해서 그럴듯한 이미지 노이즈를 합성하는 것이 이슈로 떠오른다. 이를 위해 다양한 방법들이 제기되었지만, 본 연구는 noise modeling에 치중하는 기존 방식들과 달리 signal subspace를 생성하는 기저 벡터에 초점을 맞추고, 이에 대한 projection을 통해 디노이징 성능을 향상시킨다.

<br/>

## 3. Method

**Basis Generation** - 이미지의 feature map에서 signal subspace를 span하는 기저 벡터를 생성

**Projection** - Feature map을 signal subspace에 projection함으로 써 변환

1. 2개의 feature map X1, X2를 입력으로 받아 K개의 기저 벡터 추정

   - X1 : low level feature / X2 : high level feature / K : 하이퍼 파라미터

2. K개의 기저 벡터의 span으로 생성되는 subspace에 X1을 projection


<br>

#### 3.1.1 Basis Generation

![image](https://user-images.githubusercontent.com/44194558/152289675-1d447d2d-d4f3-44d0-b06f-386a96bbd4e3.png)

 - V : K개의 기저 벡터 집합
 - f_theta : 합성곱 연산으로 구성

2개의 feature map을 입력으로 받아 기저 벡터를 추정하는 basis generation block의 파라미터 theta는 end-to-end 학습을 통해 갱신됨.

<br/>

#### 3.1.2 Projection

Orthogonal linear projection을 통해 이미지 feature map X1을 V에 projection.

Signal subspace에 대한 orthogonal projection 행렬은 다음과 같이 계산

![image](https://user-images.githubusercontent.com/44194558/152290834-2bde9d0d-f8c1-44b1-ae53-9acde182a62a.png)

X1은 signal subspace에 대한 projection을 통해 복원됨.

![image](https://user-images.githubusercontent.com/44194558/152290908-5da2e0f6-a8fc-4298-bf36-37b3cdae5744.png)

`The operations in projection are purely linear matrix manipulations with some proper reshaping, which is fully differentiable and can be easily implemented in modern neural network frameworks.`

<br/>

#### 3.2 NBNet Architecture

![image](https://user-images.githubusercontent.com/44194558/152291565-298a7e65-6e10-44eb-9155-19a4bdf3a75d.png)

SSA 모듈은 low level feature map X1, high level feature map X2를 입력으로 받음.

`Low-level feature maps from skip-connections are projected into the signal subspace guided by the upsampled high-level features.`

Projection된 feature는 원래의 high level feature map X2와 fusion되어, 디코더의 다음 스테이지에 전달됨.

전통적인 UNet구조가 low level, high level feature map을 직접 더하여 fusion 하는 것과 달리 NBNet은 fusion되기 전에 low level feature가 SSA 모듈을 통해 projection된다는 차이가 있다.

NBNet은 깨끗한 이미지-노이즈 이미지의 pair로 구성된 학습 데이터를 통해 훈련되며 깨끗한 이미지와 디노이징 출력 간의 l1 distance를 손실 함수로 사용한다.

![image](https://user-images.githubusercontent.com/44194558/152292091-ba02ceab-0c24-40d4-a2f2-82073d94cdc4.png)

<br/>

### 4. Evaluation and Experiments

본 연구가 제안하는 noise reduction은 노이즈 데이터에 대한 prior distribution을 사용하지 않으면서도 최적의 성능을 보임.

`This shows the effectiveness of the proposed projection method which helps separating signal and noise in feature space by utilizing image prior.`

Projection에 있어 X1, X2를 모두 고려하는 것이 좋음.

`Projecting X1 on the basis generated by X1 and X2 obtains the best PSNR, which is 39.75 dB.`


