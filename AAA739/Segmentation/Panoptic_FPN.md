# Panoptic Feature Pyramid Networks

<br/>

## Abstract

최근 Panoptic Segmentation처럼 instance seg와 semantic seg 이 두 태스크를 함께 해결하려는 관심이 생기고 있지만, 아직은 서로 연산을 공유하는 것 없이 개별적으로 모델을 사용하는 수준에 불과하다.

따라서 본 연구는 instance seg, semantic seg를 네트워크 아키텍쳐 수준에서 통합하고, 이를 기반으로 단일 네트워크에서 panoptic segmentation 태스크를 해결하고자 한다. 기본적으로 Mask R CNN과, FPN에 기반하고 있다.

<br/>

## 1. Introduction

`Single network baseline for the joint task of panoptic segmentation, a task which encompasses both semantic and instance segmentation.`

Instance seg, semantic seg 각각에 대해 SOTA 성능을 보이는 방법들이 서로 굉장히 이질적이기 때문에 (parallel development & separate benchmarks), 두 가지 태스크 전부에 대해 높은 성능을 보이는 단일 네트워크를 설계하는 것은 매우 어려운 일이다.

 - Semantic seg : dilated conv를 통해 강화된 backbone을 사용하는 FCN이 주류
 
 - Instance seg : Region based의 Mask R CNN 등이 주류 

따라서 본 연구는 동시에 region based output과 dense pixel output을 출력하는 단일 네트워크를 설계하고자 한다. 기본적으로 FPN backbone 기반의 (region based) instance level recognition에 semantic segmentation branch를 더하며, 기존의 instance segmentation을 수행하는 region based branch와 병렬적으로 수행된다.

![image](https://user-images.githubusercontent.com/44194558/159152464-8a05fc5b-ce1d-4bb7-887e-ddfe2d3c0528.png)

 - ` Our method, which we call Panoptic FPN for its ability to generate both instance and semantic segmentations via FPN, is easy to implement given the Mask R-CNN framework [23].` 

Panoptic FPN은 Mask R-CNN (with FPN)의 확장된 버전이며, region based prediction과 dense pixel prediction 두 가지 branch를 모두 적절하게 훈련시킨다. 그리고 어떻게 데이터를 augment할 것인지, 어떻게 학습률을 조정할 것인지, 어떤 방식으로 minibatch를 구성하고 두 branch의 loss들을 적절하게 balancing할 것인지 등 joint setting에 대해서도 논의한다.

또한 Panoptic FPN은 dilation 연산을 사용하지 않음으로써 메모리와 계산 비용의 효율성을 보장하고, 다양한 backbone 네트워크에 적용될 수 있는 유연성을 가진다.

<br/>

## 2. Related Work

### Panoptic Segmentation

`Every competitive entry in the panoptic challenges used separate networks for instance and semantic segmentation, with no shared computation. Our goal is to design a single network effective for both tasks that can serve as a baseline for future work.`

<br/>

### Instance Segmentation

FPN을 사용하는 Mask R CNN이 본 연구의 baseline. Detection 태스크에 있어 region-based 접근법이 여전히 지배적이기 때문에, 본 연구 역시 region based instance segmentation으로 부터 시작한다.

<br/>

### Semantic Segmentation

보다 정확한 mask prediction을 위해 dilated conv를 통해 feature의 해상도를 증가시키는 방식들이 제안되어 왔다. 이러한 방식들은 실제로 효과적이며 U-Net 구조나 encoder-decoder 방식보다 훨씬 지배적이지만, 사용 가능한 backbone 네트워크를 제한한다는 단점이 있다.

 - 본 연구의 semantic FPN은 dilation 모델들보다 가벼우면서도, 보다 좋은 해상도의 feature map을 생성함

따라서 본 연구는 dilated conv의 대안으로 encoder-decoder 프레임워크를 사용한다.

<br/>

### Multi-task learning

Panoptic segmentation은 segmentation에 대한 통합된 (unified) 시각을 제공해주었지만, 엄밀히 말해서 multi-task learning은 아니었지만, 본 연구는 multi-task learning을 가능케 함.

<br/>

## 3. Panoptic FPN

<br/>

### 3.1 Model Architecture

#### FPN

Multiple spatial resolution의 feature를 활용하는 FPN구조를 거의 그대로 사용함.

1/32 ~ 1/4 해상도를 가지는 feature pyramid를 만듬 (채널수는 모두 동일하게. default=256)

### Instance Segmentation Branch

각 피라미드 단계의 모두 동일한 차원의 채널을 갖기 때문에 region based object detector를 적용하기 용이함. 기본적으로 기존의 Mask R-CNN이 객체를 찾고, 그 객체에 대한 segmentation mask를 찾는 방식과 동일.

### Panoptic FPN

FPN을 사용하는 Mask R-CNN에 대한 약간의 수정만을 통해 pixel-wise semantic segmentation을 수행하는 것이 목적이기 때문에, feature map은 다음과 같은 특징들을 가지고 있어야 함. 그리고 FPN은 아래 3가지 요건을 모두 만족할 수 있다는 장점이 있음.

1. 충분히 큰 해상도 (fine-detail을 잘 포착하기 위해)

2. 충분한 semantic information 보유 (class label을 정확히 예측하기 위해) 

3. Multi-scale의 정보 보유 (다양한 크기의 stuff를 찾아내기 위해) 

### Semantic Segmentation Branch

`To generate the semantic segmentation output from the FPN features, we propose a simple design to merge the information from all levels of the FPN pyramid into a single output. `

![image](https://user-images.githubusercontent.com/44194558/159154410-9b9ac36d-cffa-4fd6-a23a-0c078332f309.png)

 - 다양한 해상도의 feature map들을 일정한 크기 (1/4)로 up-sampling
 
 - Element wise sum 수행
 
 - Conv & up-sampling 
 
 - Instance segmentation은 기존의 Mask R-CNN(with FPN) 활용. Semantic segmentation은 FPN에 가벼운 dense pixel prediction branch를 추가. 

<br/>

### 3.2 Inference and Training

Inference 수행시 각 픽셀들은 고유한 class label, instance id가 할당됨. Panoptic FPN의 예측 결과는 overlap될 수 있기 때문에 적절한 후처리 과정이 필요함 (NMS와 유사).

Training시 다음과 같은 손실함수 이용

![image](https://user-images.githubusercontent.com/44194558/159154796-2ea1250e-89be-4f14-a56f-0fb2999e4daf.png)

 - 각 태스크에 대해 scaling weight 부여

<br/>

## 4. Experiments

