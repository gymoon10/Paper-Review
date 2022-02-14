# Efficient Net: Rethinking Model Scaling for Convolutional Neural Networks

<br/>

## Abstract

CNN 네트워크는 한정된 하드웨어 자원 내에서 보다 높은 정확도를 위해 그 크기를 키워나가는 방향으로(scaling-up) 발전되어 왔고, 본 논문은 model scaling을 보다 명확히 밝혀내고자 제안되었다.

`In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance.`

 - depth(네트워크 깊이), width(채널 수), resolution(이미지 크기)의 차원을 모두 동일하게 scaling하는 방법인 **Compound coefficient**를 제안
 
 - 깊이, 넓이, 해상도 크기가 일정한 관계가 있다는 것을 실험적으로 찾아내고, 그 관계를 수식으로 표현함 

이와 같은 scaling 방식은 MobileNet, ResNet에서 굉장히 효과적이었고, 확장 가능한 새로운 baseline 네트워크인 Efficient Net을 구축했다. 

<br/>

## 1. Introduction

CNN 네트워크를 확장(Scaling-up)하는 방법은 성능 향상에 자주 사용되는 방법이지만, 효율적인 scaling-up 과정에 대해서는 여전히 제대로 연구된 바가 없다. 기존 연구들은 주로 깊이, 넓이, 해상도 중 단 하나의 차원을 조정하는 것이 일반적이었다. 이 중 2가지 이상을 조정하는 방식이 고려될 순 있지만, 매우 사소하게 작업해야 하는 수작업들을 많이 필요로 하는 단점이 있다.

본 논문의 핵심이 되는 질문은

`In particular, we investigate the central question: is there a principled method to scale up ConvNets that can achieve better accuracy and efﬁciency?`

사전 조사와 실험에 따르면 네트워크의 깊이, 넓이, 해상도들 사이의 균형을 맞추는 것은 성능 향상에 있어 핵심적인 역할을 하고, 이들간의 균형은 간단한 상수비로 구해질 수 있다. 따라서 본 논문은 간단하면서도 효율적인 **Compound scaling** 방식을 제안한다. 기존 연구들의 임의적인 조정, 튜닝과는 달리 네트워크의 깊이, 넓이, 해상도 차원을 균등하게 scaling한다.
 
 - `Unlike conventional practice that arbitrary scales these factors, our method uniformly scales network width, depth, and resolution with a set of ﬁxed scaling coefﬁcients. `

직관적으로도 입력 이미지의 해상도가 커지면 네트워크가 보다 넓은 영역을 커버할 수 있는 receptive field를 필요로 하기 때문에 보다 많은 수의 layer를 필요로하고(depth), 크기가 큰 이미지에서 fine grained feature를 추출하기 위해 보다 많은 수의 채널이 요구된다(width).

`We are the ﬁrst to empirically quantify the relationship among all three dimensions of network width, depth, and resolution.`

<br/>

## 2. Related work

`ConvNets have become increasingly more accurate by going bigger`

 - 하드웨어 자원은 무한하지 않기 때문에 효율성이 중요

`Deep ConvNets are often over-parameterized.`

 - MobileNet, ShuffleNet 등 효율성을 위한 model compression 방식들이 많이 제안되어 왔지만, 보다 대용량의 네트워크를 어떤 방식으로 compression 할 지는 아직 불분명

 - 본 논문은 Model scaling 방식을 통해 대용량 네트워크의 효율성을 향상시키고자 함

`In this paper, we aim to study model efﬁciency for super large ConvNets that surpass state-of-the-art accuracy. To achieve this goal, we resort to model scaling.`

<br/>

## 3. Compound Model Scaling

<br/>

### 3.1 Problem Formulation

최선의 architecture를 찾는 것에 집중하는 네트워크 설계 방법과는 달리, 기존의 baseline network에 대해 깊이, 넓이, 해상도 등의 차원을 확장시키는 방식을 Model scaling이라고 함.

일반적으로 모델을 scaling하는 방식은 아래와 같이 너비(b), 깊이(c), 입력 해상도(d)의 차원을 조절하는 것이다.

![image](https://user-images.githubusercontent.com/44194558/153823995-14753061-0d50-42dc-b4a0-bcfc8126eb27.png)

Baseline 네트워크에서 입력값이 각 layer(f)를 거쳐 최종 출력을 생성하는 과정을 수식으로 표현하면


![image](https://user-images.githubusercontent.com/44194558/153824312-511774db-98a8-4e28-9701-e5a60927e3db.png)

![image](https://user-images.githubusercontent.com/44194558/153824275-7ef4c563-5d17-43c4-8020-94b5fb87fefb.png)

<br/>

일반적인 CNN 디자인 방식은 최적의 layer architecture인 F_i를 찾는데 주목하지만, model scaling 방식은 F_i에 대한 변경없이 깊이, 너비, 해상도를 확장시키는데 주목한다.
 
 - F_i를 변경하지 않기 때문에 한정된 자원 하에서 최고의 accuracy를 갖는 깊이, 너비, 해상도 차원을 찾는 최적화 문제


![image](https://user-images.githubusercontent.com/44194558/153824885-11f7bdfa-b6a4-447a-b09f-f107324952f9.png)

 - w, d, r : coefficients for scaling network

<br/>

### 3.2 Scaling Dimensions

**Depth**

깊은 네트워크일 수록 보다 풍부하고 semantic한 feature를 얻을 수 있지만, 기울기 소실 문제가 발생할 수 있음. 기울기 소실을 방지하기 위해 skip-connection, batch 정규화 방식들이 제안되어 왔으나, baseline 모델의 layer에 서로 다른 depth coefficient d를 통해 scaling 하는 방식은 효율이 떨어진다 (한계효용체감).

![image](https://user-images.githubusercontent.com/44194558/153825577-d5fa9b3b-4d61-48e9-9591-410b81b9b90d.png)


**Width**

네트워크의 너비를 scaling하는 방식은 small size 모델에서 많이 사용고, 너비가 큰 네트워크일 수록 fine grained feature를 획득하고, 학습시키기 용이하다. 하지만, 극단적으로 너비가 크고, 얕은 네트워크에서는 고수준의 feature를 획득하기 어렵다. 너비 차원 w를 확장시키는 것은 성능의 측면에서 금방 포화되는 경향이 있다.

![image](https://user-images.githubusercontent.com/44194558/153825965-c3e91f1a-a5a5-496a-9774-04cd1b3450f7.png)

**Resolution**

해상도가 클 수록 fine grained feature를 획득하기 용이하다. 하지만, 해상도 크기 역시 일정 수준 이상에서는 금방 포화되는 경향이 있다.

![image](https://user-images.githubusercontent.com/44194558/153826141-62eaa435-89c1-4887-903c-afa3d927fd49.png)

<br/>

`Scaling up any dimension of network width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger models.`

<br/>

### 3.3 Compound Scaling

`We empirically observe that different scaling dimensions are not independent.`

 - 해상도가 큰 이미지를 처리하기 위해서는, 네트워크의 깊이를 깊게 하여 큰 receptive field를 확보하는 것이 중요하고, 이에 따라 고해상도 이미지에서 보다 fine grained된 feature를 얻기 위해 네트워크의 너비도 크게 해야 한다.
 
 - 해상도, 너비, 깊이 중 하나의 coefficient를 조정하는 것이 아닌, 균형적인 scaling이 필요함

아래의 표는 더욱 깊은 깊이, 높은 해상도, 큰 너비의 조건을 갖추었을 때 더 높은 성능 향상이 있음을 보임


![image](https://user-images.githubusercontent.com/44194558/153827438-94112a31-5019-4fb8-ba6a-70c5b3714a2b.png)

<br/>

`In order to pursue better accuracy and efﬁciency, it is critical to balance all dimensions of network width, depth, and resolution during ConvNet scaling.`

본 논문이 제안하는 Compound scaling은 깊이, 너비, 해상도의 차원을 아래와 같이 모두 균등하게 scaling함

![image](https://user-images.githubusercontent.com/44194558/153827616-e1d340ce-8e73-4b38-914e-06f98694438b.png)

 - a, b, r는 small grid search로 결정되는 상수 (제약 조건 하에서 a, b, r를 찾음)
 - phi는 주어진 연산량에 따라 사용자가 결정하는 하이퍼 파라미터 상수

<br/>

## 4. EfficientNet Architecture

![image](https://user-images.githubusercontent.com/44194558/153827974-db4c4de5-73cf-4837-aac7-d6d89d1391b0.png)

 - MnasNet 구조와 비슷함

![image](https://user-images.githubusercontent.com/44194558/153828133-025f0008-65de-4405-8cd1-d792ecc9ea7c.png)

<br/>

## 5. Experiments


![image](https://user-images.githubusercontent.com/44194558/153828342-1f38b7ad-d6db-4b67-a7c9-78056a1d256c.png)
