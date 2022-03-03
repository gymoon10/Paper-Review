# PWC-Net: CNNs for Optical flow Using Pyramid, Warping and Cost Volume

<br/>

## Abstract

`We present a compact but effective CNN model for optical ﬂow, called PWC-Net.`

PWC-Net은 고전적인 optical flow 알고리즘의 3가지 원칙을 CNN 모델에 적용했고, 모델의 크기를 감소시키면서도 성능을 향상시켰다.

1. Pyramidal processing

2. Warping

3. Cost volume

t, t+1 시점의 이미지가 있을 때 PWC-Net은 크게 다음과 같이 동작한다.

1. 이전 pyramid level의 optical flow 추정치를 활용하여 t+1시점 이미지의 feature를 warp

2. t+1 시점 이미지의 warped feature, t 시점의 feature를 이용하여 optical flow를 추정

<br/>

## 1. Introduction

Optical flow를 딥러닝에 활용하려는 연구는 이전부터 있어왔고, 그 중 FlowNetS & FlowNetC는 UNet 아키텍쳐를 활용하여 raw image로 부터 직접 optical flow를 추정한다. 하지만, 모델의 크기가 커서 과적합의 문제가 발생하는 단점이 있다.

SpyNet은 딥러닝과 고전적인 optical flow 알고리즘의 1, 2를 결합하여 모델 크기로 인한 과적합 문제를 완화했다 (Spatial pyramid network, initial flow를 활용해 t+1 시점 이미지를 t 시점 이미지에 맞게 warp). t 시점 이미지와 warp된 t+1 시점 이미지간의 motion은 대체로 작기 때문에, 작은 네트워크로도 두 이미지간의 motion을 추정할 수 있다. 하지만, 위의 두 모델에 비해 성능이 떨어진다는 단점이 있다.

따라서 본 연구는 다음과 같은 질문을 던진다.

 - `Is it possible to both increase the accuracy and reduce the size of a CNN model for optical ﬂow?`

SpyNet은 딥러닝과 고전적인 optical flow 알고리즘을 결합한 방식의 가능성을 보여주었으나, 성능 측면에서 부족함이 있고, 두 이미지간의 차이 (disparity)를 설명하는데 있어 훨씬 변별력있는 (discriminative) representation인 cost volume을 활용하지 않는다는 한계가 있다. 따라서 본 연구는 서로 다른 pyramide level을 warping layer를 통해 연결함으로써 큰 displacement flow (disparity, motion)를 추정하고자 한다. 

 - `We can link different pyramid levels using a warping layer to estimate large displacement ﬂow.`

또한 real time optical flow 예측에 있어 'full' cost volume을 계산하는 것은 계산 측면에서 금기시되어 왔으나, 본 연구는 featurue pyramid의 개별 levle에서 serch range를 제한함으로써 'partial' cost volume을 계산한다. 

 `Our network, called PWC-Net, has been designed to make full use of these simple and well-established principles. It makes signiﬁcant improvements in model size and accuracy over existing CNN models for optical ﬂow.`

<br/>

## 2. Previous Work (+ Background)

### Cost Volume

참고 : https://m.blog.naver.com/dnjswns2280/222073493738

**Stereo matching**

한 기준 영상에서의 한 점에 대한 동일한 점을 목표 영상에서 찾는 과정
 
 - 동일한 포인트의 위치 차이인 시차 (disparity)를 계산하는 과정. 
 
 - Disparity image / Depth map : 계산된 시차 값들을 이미지로 표현

![image](https://user-images.githubusercontent.com/44194558/156498094-ab04ae2b-7a7e-44b1-916a-5d0820654777.png)

<br/>

**Cost volume**

두 영상의 강도 (intensity)의 차이를 픽셀 단위로 계산하여 유사도를 측정하는 것. 픽셀 단위로 계산된 매칭 비용을 쌓은 것을 Cost volume이라고 함.

 - `A cost volume stores the data matching costs for associating a pixel with its corresponding pixels at the next frame [20].`
 
 - `discriminative representation of the disparity (1D flow)` 

<br/>

기존 연구들은 단일 이미지로부터 full cost volume을 계산하기 때문에 연산량과 메모리 측면에서 비효율적이었지만, 본 연구는 multiple pyramid level에서 partail cost volume을 계산함으로써 효율적이면서도, 좋은 성능의 모델을 제안했다.

<br/>

### CNN models for dense prediction tasks in vision

`Here we use dilated convolutions to integrate contextual information for optical ﬂow and obtain moderate performance improvement.`

 - Segmentation같은 dense prediction 태스크에 있어 dilated conv가 이미지의 contextual information과 보다 refined된 detail들을 잘 보존할 수 있다는 장점이 있다. 

DenseNet 아키텍쳐를 사용

<br/>

## 3. Approach

![image](https://user-images.githubusercontent.com/44194558/156518057-467d2524-2a9b-441b-a3e5-6e5ce62be47d.png)

<br/>

### Feature pyramid extractor

Raw image는 빛 변화와 그림자의 영향을 많이 받기 때문에 (variant), fixed image pyramid가 아닌 learnable feature pyramid를 사용. 2개의 입력 이미지가 주어지면 conv layer를 사용하여 feature representation들로 구성된 L-level pyramid를 구성함.

<br/>

### Warping layer

`At the l th level, we warp features of the second image toward the ﬁrst image using the ×2 upsampled ﬂow from the l+1 th level.`

 - 이전 level의 pyramid로부터의 flow를 up-sampling하고, 2번째 이미지에 더해주여 warp 수행

![image](https://user-images.githubusercontent.com/44194558/156512634-0985b832-5b1c-448d-9578-5dc5c7266119.png)

Warping 연산은 large motion을 추정하고, geometric distortion을 보정할 수 있음.

<br/>

### Cost volume layer

`We use the features to construct a cost volume that stores the matching costs for associating a pixel with its corresponding pixels at the next frame [20].`

Cost volume을 계산하기 위해 첫 번째 이미지의 feature, 두 번째 이미지의 feature 사이의 correlation을 계산.

![image](https://user-images.githubusercontent.com/44194558/156517247-4a20ae2c-9feb-4fa4-ac9b-61ce314c1a4e.png)

이 때 제한된 범위의 d 픽셀들에 대해서만 partail cost volume을 계산함.

<br/>

### Optical flow estimator

첫 번째 이미지, cost volume, upsampled flow를 입력으로 받아서 피라미드 각 level의 optical flow를 예측.

DenseNet connection을 사용하여 estimator의 성능을 향상시킴.

<br/>

### Context network

추정된 optical flow를 정제. 고전적인 방법에서 median filtering 등의 방법을 통해 optical flow에 대한 후처리를 수행하는 것과 유사한 맥락. Multi level에서 추정된 optical flow를 하나로 합쳐주는 역할.

<br/>

### Training loss

![image](https://user-images.githubusercontent.com/44194558/156517961-2f058e1e-461c-4fc4-a8d3-29bf9f380478.png)

<br/>

## 4. Experimental Results


![image](https://user-images.githubusercontent.com/44194558/156518879-3ed264ef-0aaa-48b7-8bbc-d52b8b3bc3d4.png)

### Cost volume

Range 2를 갖는 partial cost volume으로도 입력에 대해 200 픽셀에 해당하는 큰 motion들을 충분히 잡아낼 수 있었음.

2보다 작은 range의 경우 큰 motion을 무시하고, 작은 motion에만 집중하는 경향이 있었음.

<br/>

### Warping

`Warping allows for estimating a small optical ﬂow (increment) at each pyramid level to deal with a large optical ﬂow.`

Warping 연산을 사용하지 않는 경우 partial cost volume의 default 연산 range를 4로 증가시켜 low resolution pyramid level (상위 레벨)에서도 충분히 motion을 포착할 수 있도록 해야함.

<br/>

## Conclusion

`Combining deep learning with domain knowledge not only reduces the model size but also improves the performance.`