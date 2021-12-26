# Feature Pyramid Networks for Object Detection

<br/>

## Abstract

Feature pyramids는 recognition system에 있어 다양한 scale의 객체들을 검출하는 기본적인 요소. 하지만 방대한 메모리와 계산량의 문제로 인해 비교적 최근의 객체 검출 모델들은 pyramid representation을 사용하는 것을 피해왔음.

본 논문은 다양한 크기의, 피라미드 구조를 갖는 딥러닝 convolutional 네트워크를 제안 (비용 문제를 최소화하면서).

`In this paper, we exploit the inherent multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost.`

해당 구조는 다양한 컴퓨터 비전 영역에서 사용되는 feature extractor로써 유의미한 성능 향상을 이끌어 냄.
 - basic Faster R CNN에 FPN을 도입함으로써 sota 성능 도출
 - multi scale object detection에 있어 효율적이고 정확


<br/>

## 1. Introduction

방대하고 다양한 scale에서 객체를 인식하는 것은 컴퓨터 비전 분야에서 매우 중대하고 근본적인 과제. 객체 검출은 기본적으로 입력 이미지를 CNN에 전달시켜 생성된 최종 feature map을 이용함. 그 과정 속에서 이미지에 대한 high level feature를 얻을 수 있지만, 해상도가 축소되기 때문에 작은 물체를 탐지하기 어려움.
이를 해결하기 위해 아래와 같은 방법들을 활용.

![image](https://user-images.githubusercontent.com/44194558/147306604-0267cb26-f7c0-4245-896b-f151c8b8ac17.png)

 - 피라미드들은 sclae-invariant (객체의 scale 변화는 피라미드 구조의 계층을 변경함으로써 상쇄됨)
 - 객체의 위치 뿐 아니라 pyramid level 역시 고려하기 때문에 다양한 scale에서 객체를 검출하는 것이 가능해짐

이와 같이 featurized image pyramid를 사용하는 것의 장점은
 - 다양한 scale에서의(level) feature representation을 획득
 - 각 feature representation은 semantic 측면에서 유용. 
 - 일반적인 CNN 구조에서 입력에 가까운 층은 해상도가 높지만 semantic 측면에서 활용도가 떨어지는 low level feature

`The principle advantage of featurizing each level of an image pyramid is that it produces a multi-scale feature representation in which all levels are semantically strong, including the high-resolution levels.`


featurized image pyramid를 사용하는 것의 단점은
 - Inference 시간이 오래걸림
 - memory 문제 (end-to-end로 학습 시키는 것이 현실적으로 불가능)
 - train/test-time inference inconsistency (test 단계에서 피라미드는 한 번만 사용됨)
 - 위와 같은 문제로 인해 Fast & Faster R CNN의 기본 옵션은 피라미드 구조를 사용하지 않음

<br/>

**Single feature map**

![image](https://user-images.githubusercontent.com/44194558/147307810-a0b2d0f8-8382-4f22-b82b-bf3aefcd2022.png)

 - YOLO 같은 one stage detector
 - CNN을 통과할 수록 이미지에 담겨 있는 정보들이 추상화되어(high level) 작은 물체들에 대한 정보가 소실됨

<br/>

**Featurized image pyramid**

![image](https://user-images.githubusercontent.com/44194558/147308000-7d45ae7c-dbcf-4cb0-b115-df92f53811ab.png)

 - 입력 이미지를 다양한 크기로 resize한 뒤 CNN에 통과시켜 feature map 획득
 - multi scale featurized representation을 통해 작은 크기의 객체도 검출 가능
 - 연산량 증가 (현실적인 활용도 매우 낮음)

<br/>

**Pyramidal feature hierarchy**

![image](https://user-images.githubusercontent.com/44194558/147308172-e03a9904-05b9-4103-80d5-8ebaf17cc7db.png)

 - SSD에서 사용 (최초로 CNN의 pyramidal feature을 사용)
 - CNN 순전파로 계산된 여러 계층에서의 multi scale feature map을 재사용 - free of cost
 - SSD는 고해상도의 feature map을 사용하지 않아 작은 물제에 대한 정보가 소실되는 문제 발생 (conv4_3 이후의 feature map 결과만 사용)

<br/>


**Feature pyramid network** - 본 논문에서 제안하는 방법

`The goal of this paper is to naturally leverage the pyramidal shape of a ConvNet’s feature hierarchy while creating a feature pyramid that has strong semantics at all scales.`

Top down pathway, lateral connection을 사용해 low resolution & semantically strong feature (high level), high resolution & semantically weak (low level) feature를 결합

--> `a feature pyramid that has rich semantics at all levels` (Pyramidal feature hierarchy의 고해상도 feature map 미사용 문제 해결), 단일 이미지 입력으로부터 빠르게 구축됨 (Featurized image pyramid의 연산량 문제 극복)  

![image](https://user-images.githubusercontent.com/44194558/147308686-01d5c275-d1dc-4f28-9058-faecd66326ca.png)


![image](https://user-images.githubusercontent.com/44194558/147310107-e8b93583-61fe-4425-af28-55aca3eede9b.png)


 - CNN 순전파를 통과하면서 단계별로 feature map 생성
 - 가장 상위 layer (출력에 가까운) 부터 거꾸로 내려오면서 (top down) top down pathway, lateral connection을 사용하여 feature map을 합친 뒤  객체 검출 수행
 - 각 level에서 독립적으로 prediction 수행
 - 하나의 입력으로부터 빠르게 계산, 모든 scale의 semantic 정보를 담고 있기 때문에 speed, memory, power 손실 x

<br/>

## 2. Feature Pyramid Networks

`Our goal is to leverage a ConvNet’s pyramidal feature hierarchy, which has semantics from low to high levels, and build a feature pyramid with high-level semantics through out.`

`Our method takes a single-scale image of an arbitrary size as input, and outputs proportionally sized feature maps t multiple levels, in a fully convolutional fashion.`

본 논문에서는 ResNet을 활용. 피라미드 구조는 bottom up pathway, top down pathway, lateral connection을 활용하여 구축됨.

**Bottom-up pathway**

Backbone CNN의 순전파 연산 결과 (feed forward compuation) -> 다양한 scale의 feature map들로 구성된 feature hierarchy 구축.

동일한 크기의 출력을 갖는 CNN 연산들을 하나의 stage로 묶고, 각 stage의 마지막 layer 출력을 추출.

본 논문에서는 ResNet을 사용하기 때문에 각 stage 별 마지막 residual block의 feature map(feature activations output)을 추출.
--> {C2, C3, C4, C5}, C1은 사용 x

**Top-down pathway**


![image](https://user-images.githubusercontent.com/44194558/147319940-dc0c171f-b410-454c-b04c-5cc3f822993b.png)

1. 상위 feature map을 (spatially coarser, semantically stronger) nearest neighbor upsampling 기법을 활용하여 해상도를 2배씩 키우고, 1x1 conv 연산으로 채널수를 감소시킴 (일종의 차원 축소 역할, 중요한 특징에 대한 weight 값을 어느 정도 보존하면서)

 - ![image](https://user-images.githubusercontent.com/44194558/147320039-627e817d-c660-44d8-b864-ed1a934c5e27.png)
 
 - 1x1 conv 
    - 1 x 1 x 원하는 채널 수 
    - 입력 받는 것에 비해 적은 수의 채널 사용 -> 차원이 축소된 정보로 연산량을 크게 줄일 수 있음, 같은 컴퓨팅 자원으로 보다 깊은 네트워크 설계가 가능
    - bottleneck (>< 모양) : 1x1 conv로 좁아졌다가(채널 감소), 원래 적용시키고자 했던 3x3 conv 연산을 수행하고, 다시 1x1 conv로 차원을 깊게 만드는 block
       - 파라미터의 수를 줄이면서 레이어를 깊이 쌓음 (ResNet에서 뛰어난 효과) 

  
2. lateral connection을 통해 1에서 upsampling된 feature map과 같은 크기의 feature map을 결합. (element wise addition - dimension을 기준으로 행렬 내에서 같은 위치의 원소 끼리 연산)

<br/>

## 3. Applications

### 3.1 Feature Pyramid Networks for RPN

Feature pyramid network 기법을 기존의 object detection 모델들에 적용하여 성능을 확인. 본 논문에서는 Faster R CNN의 RPN, classifier에 적용하여 실험 진행.


**Faster R CNN**


![image](https://user-images.githubusercontent.com/44194558/147322891-4200072f-2e71-44ce-a89c-6cdc50eaf6a0.png)

![image](https://user-images.githubusercontent.com/44194558/147324185-aa5a7d28-52ae-4ad1-b910-de411fdaa46b.png)

![image](https://user-images.githubusercontent.com/44194558/147324208-ee941707-e59a-4a8d-9021-7b824fb4cf8d.png)

<br/>

Faster R CNN에서 single scale의 feature map을 FPN으로 대체.

 - Feature extractor인 backbone network를 VGG16에서 ResNet101로 대체.
 - ResNet을 통과시키며 중간 feature map들을 생성
 - 상위 layer부터 내려오면서 해당 layer의 기존 feature map들과 결합된 feature map (P5, P4, P3, P2 생성)
 - 각 feature map에 3x3 conv를 적용하여 classification, bbox regression 수행

![image](https://user-images.githubusercontent.com/44194558/147324523-0a4fd1a2-e081-492a-9b89-a2825b937b2e.png)

 - P5같은 상위 feature map은 크기가 큰 객체에 대한 정보를 보유 (512x512 anchor box)
 - P2같은 하위 feature map은 작은 물체애 대한 정보를 보유 (32x32 anchor box)

기존의 RPN이 intermediate layer에 1x1 conv를 anchor box수 만큼 채널 수를 취해서 계산한 것과 달리, FPN을 적용한 RPN에서는 P5...P2가 각각 특정한 크기의 anchor box를 의미. 이 feature map에 predictor head 네트워크를 적용시켜 예측 수행.

<br/>

**Feature Pyramid Networks for Fast R CNN**

기존 구조

![image](https://user-images.githubusercontent.com/44194558/147325662-9531daf8-8429-4881-9a04-7ff610082fe1.png)
  
  - Feature map에 RPN을 통해 얻은 ROI를 사상시킴

<br/>

![image](https://user-images.githubusercontent.com/44194558/147325215-68daaf97-0ac9-42b3-b912-579c743f80ef.png)

  - 다양한 크기의 feature map 존재 
  - ROI 크기에 따라 사상시킬 feature map을 결정하는 수식 제안
  - ![image](https://user-images.githubusercontent.com/44194558/147332466-3e282cb1-ae5a-478a-9522-21095fd6a6f5.png)




