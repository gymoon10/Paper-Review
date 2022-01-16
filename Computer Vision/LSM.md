# LSM: Learning Subspace Minimization for Low-level Vision

<br/>

## Abstract

본 연구는 Low-level vision 태스크의 최적화 문제에 대한 새로운 접근법을 제시.

`We replace the heuristic regularization term with a learnable subspace constraint, and preserve the data term to exploit domain knowledge derived from the ﬁrst principle of a task.`

LSM 기법은 다양한 low-level vision 태스크의 네트워크 구조와 파라미터들을 통합하여(completely shared parameters) multi-tasking이 가능한 단일 네트워크를 학습할 수 있도록 한다.

<br/>

## 1. Introduction

다양한 vision 태스크들은 다음과 같은 energy minimization problem으로 표현될 수 있다.

![image](https://user-images.githubusercontent.com/44194558/149647956-bbf1c753-12b5-4cb3-b633-982b6729474b.png)

 - x : 최적 solution
 - D(x) : Data term, first principle of task에 의해 규정됨  ex) optical flow의 색 항상성 가정 
 - R(x) : Regularization term 

Regularization term은 x를 픽셀 단위로 정규화하여 유사한 픽셀들이 서로 유사한 solution value를 갖도록 하지만 종종 heuristic한 경우가 있다. 이런 정규화 과정은 data term 만으로는 최적화 문제를 해결하기에 충분하지 않고, ill-posed problem인 경우가 많은 low level vision 태스크에 있어 반드시 필요하지만, 통상적인 L^2 smoothness 정규화는 객체의 경계선 근처에서 지나치게 과한 smoothing을 적용하는 단점이 있다. (noise는 smooth out시키고, sharp edge는 최대한 보존하도록 정규화 하는 것이 이상적) 이에 따라 많은 edge-preserving regularization 기법이 제안되어 왔으나 부족한 점이 많다. 

따라서 본 연구는 pixel-level similarity에 주목하기 보다는 image-level context 정보를 활용하고자 하며, D(x)는 보존한 채 heuristic한 R(x)를 subspace constraint로 대체하는 새로운 기법을 제안한다. (기존의 heuristic한 regularize term 대신, CNN을 활용한 image feature로 정규화를 수행)

![image](https://user-images.githubusercontent.com/44194558/149648130-19d8a3d2-b0ba-4e88-84b1-79505c190588.png)

 - V : k개의 기저 벡터에 의해 span되는 K-dim 공간 

<br/>

본 연구는 최적 solution x가 다수의 layer로 구성되어 있고(e.g motion layers for optical flow), v1~vk의 각 기저 벡터들이 이 layer에 대응된다고 가정하여 image level context를 활용해 문제를 정규화하고자 한다. 이에 따라 최적 solution x는 기저 벡터들의 선형 결합 형식으로 표현되며, 그 계수를 찾는 문제가 된다.

 `We can represent the solution x as a linear combination of these basis vectors and solve the combination coefﬁcients, leading to a compact minimization that not only is efﬁcient but also enables end-to-end training and outperforms the conventional regularization term R(x).`

 LSM 프레임워크는 Feature pyramid를 활용하여 점진적으로 V를 발전시키며, Eq.2를 해결한다. 각 pyramid 단계에서 CNN(image feature), intermediate solution x에 대한 D(x)의 미분을  활용하여 v를 갱신한다.

`Since the generation of V receives the task- speciﬁc data term as the input, it decouples the task-speciﬁc characteristics from the subspace generation and uniﬁes the network structures and the parameters for various tasks.`

<br/>

`As a consequence, our LSM framework enables joint multi-task learning with completely shared network struc- tures as well as parameters, and even makes zero-shot task generalization possible, where a trained network is plug-and- play for an unseen task without any parameter modiﬁcation, as long as the corresponding data term D(x) is formulated.`

본 연구가 제안하는 방식은 domain knowledge (태스크의 제 1원칙으로부터 도출된 data term 최소화)와 CNN (subspace constraint 생성 학습)을 결합하여 많은 태스크에서 SOTA 성능을 보이는 장점이 있다.

<br/>

## 2.Related Works

모든 태스크에서 적절한 objective function과 regularization term을 정의하는 것은 중요한 이슈.

`We only preserve the data term since it is usually derived from the ﬁrst principle of a task, and replace the heuristic regularization term to a learned subspace constraint that captures the structure of the desired solution at the whole image context level and enables end-to-end training to boost the performance from data.`

LSM 프레임워크는 CNN을 사용하지만, 그 결과를 직접 활용하기 보다는 문제에 대한 solution을 subspace에 constraint하여 date term 최적화를 용이하게 하는 용도로 활용한다. Data term D(x)는 주어진 태스크에 대한 제 1원칙으로부터 도출되고, task-specific formulation을 네트워크 파라미터로부터 분리한다.

`Therefore, our framework uniﬁes the network structures as well as parameters for different tasks and even enables zero-shot task generalization, which are difﬁcult for fully CNN based methods.` 

<br/>

## 3. Learning Subspace Minimization

<br/>

### 3.1 Overview


![image](https://user-images.githubusercontent.com/44194558/149648812-f96fcc6d-a4a2-4f78-bf27-19d44e87fca2.png)

 - 단일 이미지로부터 4개의 feature map 계산
 - F1~F4 순서대로 strides={32, 16, 8, 4}, channels={512, 256, 128, 64}

각 피라미드 단계에서 D(x)를 정의하고 Eq.2를 해결한다. 이때 D(x)는 intermediate solution x에 대해 2nd order Taylor expansion을 활용하여 근사되고, 다음과 같은 quardratic minimization problem으로 표현된다. 

![image](https://user-images.githubusercontent.com/44194558/149648723-e7eeebb3-4476-45a0-a1f7-e95a3f6220bb.png)

 - D : 2nd order derivatives (표기에 주의, data term과 구별)
     - `The structure of D is task dependent: it is a diagonal matrix for one-dimensional tasks or block diagonal for multi-dimensional tasks.` 
 - d : 1st order derivatives

Eq.2의 subspace constraint를 유지하기 위해 incremetal solution delta x를 제안한다. Delta x는 기저 벡터 v1...vk의 선형 결합으로 표현된다.

![image](https://user-images.githubusercontent.com/44194558/149648770-bec3280d-7fb3-4f09-ae86-34aaf03ce80e.png)

 - c : combination coeffs
 - V : 각 열이 k개의 기저 벡터에 대응하는 dense matrix (표기에 주의, subspace V와 구별)


![image](https://user-images.githubusercontent.com/44194558/149648820-f6da4980-4ae6-4375-bd61-3519fcbb2b0e.png)

 - Subspace V는 image context(feature map), minimization context를 활용하여 생성됨 (x + delta x)
 - Intermediate solution x를 갱신하고 다음 단계의 피라미드에 전달

<br/>

### 3.2 Subspace Generation

Subspace generation의 2 원칙은 다음과 같다.

1. Low-level vision 태스크들은 ill-posed인 경우가 많기 때문에 data term D(x)만 활용하는 것은 불충분하다. 따라서 subspace V를 생성하기 위해서는 image context information(feature map)을 활용하는 것이 중요하다. 개별 기저 벡터 vk는 spatially smooth하게 만든다 (객체 경계선을 제외하고).

2. 최적화 문제의 목적 함수(data term)은 반복적으로 최적화된다. 개별 iteration에서 intermediate solution x는 목적 함수 상에서 각기 다른 위치에 존재하며, 목적 함수에 대한 미분값이 최적화를 위한 delta x의 방향과 크기를 결정한다. 따라서 subspace V의 생성에 있어 estimated solution과 GT의 간극을 최소화하기 위해서는 minimization context를 고려해야 한다. 

위의 2 원칙을 고려하여 다음과 같은 과정으로 subspace V를 생성한다.


![image](https://user-images.githubusercontent.com/44194558/149649523-9549734e-a52d-4c79-8d5d-7314234d904e.png)


 1. FPN의 c 채널 Feature map F에 1x1 conv를 적용시켜 m=c/8 채널의 image context를 계산. 
    - 차원을 감소시킴으로써 계산 복잡도를 줄이고, image context와 minimization context의 균형을 맞춤
 
 2. 2m 채널 minimization context 계산. 
    - c 채널의 feature map을 m개의 그룹으로 분리한 후, 각 그룹마다 derivatives 계산.
    - 1st, 2nd derivatives를 concat


 3. Intermediate solution x를 정규화
 
 4. 1~3의 결과를 concatenate하여 (3m+1)채널의 input feature 생성 & Integral Image 계산 
 
 5. Multi-scale의 context 정보를 고려하기 위해 context feature들을 4개의 서로 다른 크기의 kernel을 사용하여 average pool (feature map의 크기는 유지)
 
 6. 각 scale에 개별적으로 1x1 conv를 적용시키고 concat하여 8m 채널의 multi scale features 생성.
 
 7. 4개의 residual block을 사용하여 K 차원의 subspace 생성. 각 피라미드 단계별로 k={2, 4, 8, 16}

<br/>

### 3.3 Subspace Minimization

![image](https://user-images.githubusercontent.com/44194558/149649586-5a1a3a97-42ec-4a10-b96a-6e03f328fc99.png)

<br/>

위의 과정으로 Subspace V가 생성되면 Eq.4의 해결이 가능.

![image](https://user-images.githubusercontent.com/44194558/149649626-567907f1-d969-4abd-bfdc-060dff15c4d7.png)  
![image](https://user-images.githubusercontent.com/44194558/149649629-d748723f-1e2f-435b-a1a2-1803b4cd9863.png)

Intermediate solution x를 x <- x + Vc로 갱신하면 아래와 같이 subspace constraint를 위배하게 된다.

  - 현재의 solution x는 마지막 iteration의 subspace에 속해있기 때문에 새롭게 생성된 subspace V에 위치할 것이라는 보장 x

![image](https://user-images.githubusercontent.com/44194558/149649721-5df1353d-d712-45c4-be16-d4f2ef803b41.png)

따라서 x를 현재 새롭게 생성된 subspace V에 project하는 방법을 고려한다.

![image](https://user-images.githubusercontent.com/44194558/149649801-bc92c90b-99dc-4abc-85d4-e45127565ad9.png)

<br/>

## 4. Experiments

`Our framework can be applied to a low-level vision task as long as its ﬁrst-order and second-order differentiable data term can be formulated.`

`Note that the whole network structure and all parameters are shared for all tasks, while previous works [13, 16, 37] only share the backbone and use different decoders/heads to handle different tasks.`

 - `We ﬁx the learned parameters and do not require any extra information between tasks.`

<br/>


![image](https://user-images.githubusercontent.com/44194558/149649966-c1470722-0c7b-414b-8799-cc243a2bdab5.png)

<br/>

![image](https://user-images.githubusercontent.com/44194558/149649979-54acd381-ce71-475d-ba84-16c874992ec7.png)

<br/>

![image](https://user-images.githubusercontent.com/44194558/149650003-04a45a59-de6d-46b2-87dc-23588573c66b.png)

<br/>

### Ablation Studies

Minimization context, Subspace projection의 중요성

`It is difﬁcult to learn an uniﬁed subspace generator solely from image context, because different tasks requires different subspace even on the same image.`

`Maintaining the subspace constraint via projection is necessary not only in theory but also in practice for better performance.` 

 - `The predicted subspace V is learned to be consistently towards the ground truth solution. In contrast, learning without projection violates the subspace constraint, and make the minimization less constrained and training more difﬁcult.`

<br/>

### Visualization of Generated Subspaces

Subspace constraint는 Low level task는 다수의 layer로 구성되어있다는 가정을 이용. Optical flow와 segmentation 태스크에서 Subspace를 생성하는 몇 개의 기저 벡터를 시각화 하여 해당 가정이 실제로 부합하는 지 확인.

![image](https://user-images.githubusercontent.com/44194558/149650184-83440429-b649-4c25-9964-c4aba02cbc79.png)

 - 기저 벡터들이 optical flow의 motion layer, segmentation의 전경/배경 layer에 consistent한 것을 확인

`Our subspace generation network captures the intrinsic characteristics of each task.`

<br/>

## 5. Conclusion

LSM은 CNN을 활용하여 기존의 regularizaion term을 대체할 수 있는 content-aware subspace constraint를 생성한다.

또한 low level task를 해결함에 있어 문제에 내재되어 있는 underlyin nature와 주어진 태스크의 제 1 원칙을 반영하고 있는 data term을 활용한다.

이와 같은 방식은 domain knowledge(문제의 제 1원칙으로 부터 도출된 data term 최소화), CNN의 expressive power(content-aware subsapce 생성 학습)을 결합하여 성능을 향상시킨다.

`Our LSM framework supports joint multi-task learning with completely shared parameters and also gener- ate state-of-the-art results with much smaller network and faster computation.` 














