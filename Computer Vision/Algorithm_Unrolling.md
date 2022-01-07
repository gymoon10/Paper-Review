## Algorithm Unrolling : Interpretable, Efficient Deep Learning for Signal and Image Processing

<br/>

## Abstract

딥러닝은 이미지, 신호 처리에서 놀라운 성과를 보였지만 네트워크의 black box 특성은 딥러닝 네트워크의 실용적인 활용에 방해가 된다. 본 연구는 **Algorithm Unrolling**을 통해서 딥러닝의 해석 불가능 문제를 해결하고자 한다. 

`Algorithm unrolling or unfolding offers promise in eliminating these issues by providing a concrete and systematic connection between iterative algorithms that are used widely in signal processing and deep neural networks.`

<br/>

## 1. Introduction

신호 처리 분야의 알고리즘 측면에서 딥러닝의 **learning based approaches**는 기존의 **analytic methods**에 대한 대안을 제시했다. 물리적인 프로세스에 대한 분석과 특성 공학 **(hand-crafting)**에 대한 엄밀한 분석을 요구하는 기존의 전통적인 iterative approaches와 달리, 딥러닝 기법은 자동으로 모델의 패턴과 정보를 파악하고 **(data-driven)** 네트워크의 파라미터를 최적화 함으로써 데이터를 통해 학습한 패턴과 정보를 반영하게 된다. 

수 많은 layer와 파라미터로 구성된 딥러닝 네트워크의 계층적 구조는 설계하기 어려운 복잡한 mapping을 학습하는데 용이하다. 학습 데이터만 충분하다면 딥러닝 네트워크의 이와 같은 adaptivity는 모델의 부족함(inadequacies)를 충분히 극복할 수 있게한다. 또한 inference와 계산 측면에서 효율적이라는 장점이 있다.

많은 딥러닝 기법들은 다른 문제와 태스크들에 공통적인(general) 네트워크 구조를 적용시켜 특정한 mapping (분류, 회귀 등)을 학습하게 한다. 그렇기 때문에 매우 고차원의 파라미터들을 관찰한다고 해서 네트워크 구조 내부에서 어떤 것들이 학습되엇는지 파악하는 것은 사실상 불가능한 일이다 (lack of interpretability). 

즉, 전통적인 **iterative algorithm**은 해결하고자 하는 문제에 내재되어 있는 물리적 프로세스와 도메인 사전 지식을 모델링하기 때문에 해석 가능성은 높지만, 딥러닝 네트워크는 해석 가능성의 측면에서 유용성이 매우 떨어진다. 때때로 성능 못지 않게 해석 가능성이 중요한 영역들이 존재한다는 점을 고려하면 이는 매우 심각한 단점이다.

`In other words, generic deep networks are usually difﬁcult to interpret. In contrast, traditional iterative algorithms are usually highly interpretable because they are developed via modeling the physical processes underlying the problem and/or capturing prior domain knowledge.`

이러한 interpretability는 generalizability와도 연관성이 있다. 딥러닝의 성능은 양적, 질적으로 충분한 데이터에 크게 의존하는 측면이 있다. 고품질의 훈련 데이터를 사용할 수 없는 경우 over-fitting의 문제가 크게 발생하여 일반화 가능성을 해치게 된다. 이와 같은 over-fitting은 매우 많은 수의 파라미터를 사용하는 generic neural network의 큰 단점이다.

`Without exploiting domain knowledge explicitly beyond time and/or space invariance, such networks are highly under regularized and may overﬁt severely even under heavy data augmentations.`

위와 같은 문제를 해결하고자 domain enriced or prior information gudided deep network가 제안되었지만, 도메인 지식을 네트워크 파라미터에 transfer하는 것은 매우 어려운 일이고 (non-trivial), 효과적인 사전 정보를 설계하는 것 역시 쉬운 일이 아니다. 이러한 문제의 해결을 위해서

`A promising technique called algorithm unrolling (or unfolding) was developed that has helped connect iterative algorithms such as those for sparse coding to neural network architectures.` 

![image](https://user-images.githubusercontent.com/44194558/148150305-ddca0e9b-252f-400e-9b71-13b7337b6277.png)

 - 알고리즘 단계의 1 step <-> 딥러닝 네트워크의 1 layer
 - 네트워크를 통과 <-> iterative 알고리즘을 정해진 횟수 만큼 반복
 - **iterative 알고리즘의 파라미터를 네트워크 파라미터로 transfer (unroll iterative algorithms into deep networks)** 

`The unrolled networks are highly parameter efﬁcient and require less training data. In addition, the unrolled net- works naturally inherit prior structures and domain knowledge rather than learn them from intensive training data`


<br/>

## 2. Generating Interpretable Networks Through Algorithm Unrolling

<br/>


### A. Conventional Neural Networks


일반적인 3개의 딥러닝 네트워크 아키텍쳐 (Fully connected MLP, CNN, RNN) 소개

![image](https://user-images.githubusercontent.com/44194558/148151335-1308e758-0788-4e33-bdff-9b73c2333bd1.png)

![image](https://user-images.githubusercontent.com/44194558/148151359-bb1066f1-01f4-4b87-b2a0-01d0b2fa48d3.png)

![image](https://user-images.githubusercontent.com/44194558/148151404-511a1724-008d-4ade-bd93-04140f0c1ab0.png)

<br/>

### B. Unrolling Sparse Coding Algorithms into Deep Networks

가장 대표적인 sparse coding 문제인 Iterative Shrinkage and Thresholding Algorithm을 algotrihm unrolling을 통한 end-to-end training으로 전환하여 효율성을 향상시킴.

`One can form a deep network by mapping each iteration to a network layer and stacking the layers together which is equivalent to executing an ISTA iteration multiple times.`

ISTA를 딥러닝 네트워크로 unrolling하면 학습 데이터와 역전파를 통해 학습이 가능하고, 계산 측면에서 효율성이 매우 높음.

`Consequently, the sparse coding problem can be solved efﬁciently by passing through a compact LISTA network.`

<br/>

#### Learned ISTA

신호에 대한 parsimonious representation (mathematic model parameterized with small number of parameters) 획득이 목적. 다음에 대한 convex 최적화를 제공하는 sparse code x를 찾아야 함.

![image](https://user-images.githubusercontent.com/44194558/148154148-410a4a20-f926-4b15-b971-3bca9be62f60.png)

![image](https://user-images.githubusercontent.com/44194558/148153540-2f994be9-0142-4216-9f10-5382dd66740c.png)

위의 최적화 문제는 다음과 같이 해결

![image](https://user-images.githubusercontent.com/44194558/148153649-b110412f-a201-409e-a655-f1b14cca297c.png)

**위의 iteration을 딥러닝 네트워크 layer로 recast(unroll)**

![image](https://user-images.githubusercontent.com/44194558/148153948-2bda9101-3673-418c-b8e1-95f4193728e2.png)

 - Unroll하는 과정에서 W에 대한 implicit substitution인 W_t, W_e 생성
 - 해당 파라미터들은 original parameterization W에 대한 일반화이자, unrolled 딥러닝 네트워크의 representation power를 강화

<br/>

### C. Algorithm Unrolling in General

`We unroll the algorithm into a deep network by mapping each iteration into a single network layer and stacking a ﬁnite number of layers together.`

알고리즘의 매 iteration step은 파라미터 theta_l(l=0...L-1)를 보유하고 있는데, unrolling을 통해 이 파라미터들이 딥러닝 네트워크에 대응될 수 있게 되며 실제 데이터셋을 활용하여 end-to-end로 학습된다.


![image](https://user-images.githubusercontent.com/44194558/148158942-6b154a82-1e3e-4b2f-8518-b1c3c697226c.png)

Unrolling 방식은 계산 뿐 아니라 성능 측면에서도 다음과 같은 향상을 이끌어낼 수 있다.

 - 설계하기 어려운 (연산에 사용되는) 필터, 딕셔너리의 계수들을 역전파를 통해 구할 수 있다.
  
 - 맞춤형 설계(custom modification) 가능
    - 위의 LISTA에서 W_t, W_e는 매 iteration마다 학습됨 (no longer held fixed throughout the network)
    - 해당 파라미터 값들은 layer마다 동일하게 공유될 수 있지만 다를 수도 있음


 - 도메인 지식을 unrolling을 통해 encoding하기 때문에 파라미터 감소  
  
 - Task & target specific

<br/>

## 3. Unrolling in Signal and Image Processing Problems

<br/>

### A. Applications in Computational Imaging

Inverse problem을 해결하는 것이 핵심. Sparse coding, low-rank matrix pursuit, 변분 방식 등이 Model based inversion이고 이러한 모델 기반 방식들을 iterative approaches와 접목시킬 수 있음.

#### Super-resolution

단순한 interpolation 방식 대신 이미지의 전체적인 구조를 반영함으로써 성능을 향상. (learning dictionaries to encdoe local image structures into sparse codes)

![image](https://user-images.githubusercontent.com/44194558/148160920-871b46cc-e10e-4a67-be6c-f171e1d5bcb6.png)
 - LISTA 서브 네트워크가 end-to-end 에 삽입되어 입력 이미지 패치로부터 sparse code a를 추정
 - 학습 가능한 딕셔너리 D가 sparse code를 reconstructed patch로 mapping

#### Blind image deblurring

블러링된 sharp 이미지로부터 blur kernel, 원본 sharp 이미지 추정 (jointly estimate). Blur kernel은 low pass nature, ill-posed (which may have more than one solution) 문제인것에 비해, 기존 방식들은 blur kernel을 추정하기 위해 stable feature를 추출하는 문제가 있다. 원본 Sharp 이미지는 추정된 kernel들을 사용하여 반복적으로 샤프 이미지를 복원한다.

네트워크는 이런 모듈들을 unroll하고 concat함으로써 구축된다.

![image](https://user-images.githubusercontent.com/44194558/148161832-af10d00b-1b43-47f9-9a04-77faa73d782d.png)

 - Feature extraction은 small CNN과 유사
 - Kernel, image estimation은 최소 제곱법으로 학습

최근에 제안된 DUBLID (Deep Unrolling for Blind Deblurring)의 성능 비교

![image](https://user-images.githubusercontent.com/44194558/148176308-bc6f1c97-ad9d-41f4-b14e-85bd6957f8df.png)

<br/>

#### DUBLID

`Unrolling approach for blind image deblurring by enforcing sparsity constraints over ﬁltered domains and then unrolling the half-quadratic splitting algorithm`

Spatially invariant blurring process는 다음과 같은 convolution 형태로 표현

![image](https://user-images.githubusercontent.com/44194558/148177370-37a3d6c1-e432-4f87-83da-4c66a5f1aa7d.png)

 - y : blurred image
 - x : latent sharp image
 - k : blur kernel (unknown)
 - n : gaussian random noise

문제의 해결을 위해 아래와 같은 update가 매 iteration마다 발생함

![image](https://user-images.githubusercontent.com/44194558/148178316-636ea1ef-953b-44b6-ac91-87be48ecf6da.png)

원본 sharp image는 다음의 최소자승법 문제를 해결함으로써 복원됨 (g, k 이용)

![image](https://user-images.githubusercontent.com/44194558/148178461-2add529f-0982-41b3-aac9-a245d9614abd.png)

위의 두 식을 딥러닝 네트워크에 unrolling 함으로써 g, z, k에 대한 갱신이 이루어지는 L개의 layer를 획득. 하나의 layer마다 하나의 image retrieval

 - g : sharp image에 대한 gradients (x, y 방향)
 - f : blurred image의 x, y 편미분을 계산하는 C개의 filter
 - k : blur kernel
 - z : ?

학습 시에 손실로 spatially invariant한 MSE를 이용하여 deblurred image, blur kernel의 가능한 spatial shift의 영향을 배제.

### B. Applications in Medical Imaging

Reduced scanning time에 대응되는 적은 수의 관측 및 측정으로부터 원래의 신호를 복원. 

#### ADMM-CSNet

![image](https://user-images.githubusercontent.com/44194558/148182102-dbb72e8a-ee85-48b4-8a0d-dba9a5a616f6.png)

 - y : linear measurements
 - phi : measurement matrix
 - x : original siganal (to be reconstructed)

아래 식에 대한 최적화 문제

![image](https://user-images.githubusercontent.com/44194558/148182258-77318d6f-9b29-4b01-bf5a-9460e1feae66.png)

최적화 문제의 duality에 의해 아래와 같이 변형

![image](https://user-images.githubusercontent.com/44194558/148182351-39ead526-c5a5-4a89-874b-4c09c9a5685d.png)

 - a : dual varibles

위 식은 dual variable a에 의해 **alternately minimize**됨 (매 iteration마다 dual variable을 갱신하면서)

![image](https://user-images.githubusercontent.com/44194558/148182571-1f3ca653-5a42-4ceb-8ea7-3e6f03e2e4bc.png)

위의 반복적인 연산들을 concat함으로써 네트워크를 구축하는 것이 가능하고, RMSE 손실을 이용하여 학습됨.

![image](https://user-images.githubusercontent.com/44194558/148183268-347fd361-c4d6-47db-90d4-f445b013abdd.png)

