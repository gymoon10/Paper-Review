# Panoptic Segmentation

<br/>

`Panoptic is “including everything visible in one view”, in our context panoptic refers to a unified, global view of segmentation`

<br/>

## Abstract

본 논문이 제안하는 panoptic segmentation은 원래 별개의 독립적인 태스크로 인식되어왔던 semantic segmentation (개별 픽셀에 class label 할당), instance segmentation (detect + 개별 object에 대한 segment) 를 하나로 합침.

사람의 눈은 두 가지 태스크를 동시에 할 수 있지만 기존의 컴퓨터 비전 영역에서는 쉽지 않았음.

![image](https://user-images.githubusercontent.com/44194558/158942314-3ec73412-07d4-442f-9469-29731920bd08.png)

 - Sematic segmentation의 경우 개별 object(instance)간의 구분이 없음. 서로 다른 자동차지만, class 측면에서는 전부 자동차이기 때문에 모두 파랑색으로 표현됨.
 
 - Instance segmentationd은 이미지에 존재하는 모든 개별 object를 검출. 검출된 object들은 서로 구별되는 색을 가짐. 자동차들은 색상을 통해 구별되기 때문에 class가 부여되지 않음. 
 
    - 검출되지 않는 object들은 (배경쪽에 있는) segmentation되지 않음. 

<br/>

`The proposed task requires generating a coherent scene segmentation that is rich and complete, an important step toward real-world vision systems.`

본 연구는 이미지 segmentation에 대한 **unified view**를 제안하고자 한다.

<br/>

## 1. Introduction

<br/>

초기 컴퓨터 비전 연구에서 주목 받았던 것들은 셀 수 있는 `things` (ex. 사람, 동물 등) 였지만, 점차 서로 비슷한 질감과 물질로 구성된 무정형의 `stuff`도 (ex. 하늘, 풀, 길 등) 주목을 받기 시작했다.

Semantic segmentation은 stuff에 주목한다. Stuff는 그 형태가 정형적이지 않기 때문에 개별 픽셀에 class label을 부여하는 작업을 거친다. 반면, Instance segmentation(or object detection)은 things에 주목하며, 개별 object를 segmentation mask나 bbox로 명확히 구별하는 것을 목적으로 한다.

이와 같은 stuff <-> things의 이분법(dichotomy)은 최근까지 유지되어 왔고, 본 연구는 stuff, thing간의 통합(reconciliation) 가능성에 의문을 던진다. 이를 위해 다음과 같은 task들을 제안한다.

1. `Encompasses both stuff and thing classes`

2. `Uses a simple but general output format` 

3. `Introduces a uniform evaluation metric` 

Panoptic Segmentation(PS) 태스크의 목적은 개별 픽셀에 semantic label을 부여함과 동시에 instance id를 할당하는 것이다 (자동차 객체에 자동차 class label을 부여 + 해당 자동차의 고유한 식별 id 부여).

PS의 가장 중요한 측면 중 하나는 평가 과정에서 사용되는 task metric이다. 기존에 사용되던 metric들은 stuff, thing 둘 중 하나의 태스크에만 특화되어 있으므로 (disjoint), 본 연구는 joint metric인 Panoptic Quality(PQ)를 제안한다.

 - PQ는 stuff, thing에 대한 성능을 균등하게 (uniform manner) 평가함.
 
`The panoptic segmentation task encompasses both semantic and instance segmentation but introduces new algorithmic challenges` 

  - PS는 semantic segmentation과 달리, 개별 object instance를 구별지어야 함 (fully convolutional net에 대한 challenge).
  
  - PS는 instance segmentation과 달리, object segment들은 non-overlapping되어야 함 (개별 object에 독립적으로 적용되는 region based method에 대한 challenge)

  - 제안 : `Generating coherent image segmentations that resolve inconsistencies between stuff and things is an important step toward real-world uses`

또한 두 독립적인 시스템의 출력을 결합하는 post-processing을 위한 sub-optimal heuristic을 제안한다.

<br/>

## 2. Related Work

1. Multitask learning
   
PS는 multi-task learning이 아닌 이미지  segmentation에 대한 single, unified view를 제공하는 방식으로, single coherent scene segmentation을 필요로 한다.


2. Joint segmentation tasks
   
`By addressing both stuff and things, using a simple format, and introducing a uniform metric, we hope to encourage broader adoption of the joint task.`

<br/>

## 3. Panoptic Segmentation Format

<br/>

### Task format

개별 픽셀을 (semantic class, instance id)의 pair로 mapping. Instance id는 동일한 class에 속하는 여러 객체들을 distinct segment로 분리한다. GT annotation 역시 pair 형식으로 미리 지정된다.

<br/>

### Stuff and thing labels

Semantic label set L은 subset인 L_st (stuff), L_th (thing)의 합집합으로 정의됨 (두 subset간의 교집합 x). 특정 픽셀이 어떤 semantic class(stuff) 하나에 할당되었을 때, 그 픽셀의 instance id는 semantic class와는 독립이다 (irrelevant).

<br/>

### Relationship to semantic segmentation

Semantic segmentation의 보다 엄격한 일반화 버전. 만약 GT annotation이 instance를 specify하지 않거나, 모든 class가 stuff에 속한다면 task metric은 다르지만 task format은 기존의 semantic seg와 동일하다.

<br/>

### Relationship to instance segmentation

기존의 instance seg는 overlapping segments를 허용하지만, PS는 하나의 픽셀에 하나의 semantic label과 하나의 instance id를 부여하기 때문에 overlapping이 허용되지 않는다.

<br/>

## 4. Panoptic Segmentation Metric

`Unified metric for stuff and things`

  - treat stuff and thing classes in a uniform way
  
  - involves 1.segment matching and 2. PQ computation givem matches

<br/>

### 4.1 Segment Matching

예측 segment와 GT segment간의 IoU가 0.5보다 클 때, 둘이 서로 matching된다고 정의한다. 이 요구 조건에 의해 non-overlapping property를 충족시킬 수 있을 뿐 아니라 unique matching역시 가능하게 된다.

 - 이미지에 대해 예측 seg, GT seg가 주어졌을 때, 개별 GT seg는 최대 1개의 (IoU가 0.5보다 큰)예측 seg와 매칭될 수 있다.

![image](https://user-images.githubusercontent.com/44194558/158949748-747be388-2320-48aa-b7c9-7edb709b01d9.png)

 - 2번째 식에서 IoU(p1, g)가 0.5보다 크면 IoU(p2, g)는 0.5보다 작아지게 되므로 unique matching이 가능해짐

<br/>

### 4.3 PQ Computation

Matching된 gt seg, 예측 seg를 기준으로 계산. 

  - `We calculate PQ for each class independently and average over classes. This makes PQ insensitive to class imbalance.`

개별 class에 대해 예측 seg - GT seg의 unique matching의 결과는 TP, FP, FN으로 분류될 수 있음.

![image](https://user-images.githubusercontent.com/44194558/158950559-5937bc53-5403-47b1-9559-fd1e86058a30.png)

<br/>

PQ는 위의 분류 결과를 바탕으로 계산됨

![image](https://user-images.githubusercontent.com/44194558/158950619-67119039-3db1-4b0e-bc11-e7f9c8381f66.png)

 - 올바르게 matching된 segments에 대한 평균 IoU와, 올바르지 않은 matching에 대한 penalized term 존재

위의 식을 PQ = SQ x RQ 곱셈 형식으로 분해하면

![image](https://user-images.githubusercontent.com/44194558/158950708-3b48f417-e82a-407a-9415-935d1bcff962.png)

 - SQ : matched segment들에 대한 IoU 평균 (segmentation quality)
 
 - RQ : FP/FN이 없으면 1, 많으면 많을 수록 1보다 작아짐 (recognition quality)

 - `It measures performance of all classes in a uniform way using a simple and interpretable formula.`

<br/>

SQ, RQ는 서로 독립적인 값은 아니며, 모든 class(stuff & thing)에 대해 계산되며 각각 segmentation quality, recognition quality를 측정함. SQ, RQ를 활용한 분해를 통해 interpretability를 제공했지만, 이것이 PQ가 단순한 semantic segmentation metric, instance segmentation metric과의 combination이란 뜻은 아님.




