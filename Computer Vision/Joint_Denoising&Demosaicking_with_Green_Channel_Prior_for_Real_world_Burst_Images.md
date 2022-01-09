# Joint Denoising and Demosaicking with Green Channel prior for Real world Burst images

<br/>

## Abstract

Denoising, demosaicking은 raw color filter array data로부터 full color 이미지를 복원하는 태스크에 있어서 핵심적이고, 서로 연관성이 높은 작업들이다. 딥러닝의 발전과 많은 후속 연구 덕에 denoising, demosaicking을 동시에(jointly) 수행하는 등 많은 발전이 있었다.

하지만 기존의 CNN-based joint denoising, demosaicking은 단일(Single) 이미지에 대해 작업을 수행하고 white gaussian noise를 가정하기 때문에 현실에 적용하는데 한계가 있다 (JDD-S). 본 연구는 G 채널이 다른 채널에 비해 2배 더 샘플링되고, CFA raw data에서 보다 높은 품질을 보유하고 있다는 사실을 이용하여 JDD-B 태스크를 위한 Green Channel Prior Network를 제안한다.

 - `Burst images` : A series of rapid succession images used to capture objects and people in motion. Burst images are captured at a high speed and the series can contain any number of images. The number is usually determined by the shutter speed and mount of available storage space on the device.

해당 GCP-Net에서는 G 채널에서 추출된 GCP feature들이 전체 이미지에 대한 feature extraction, upsampling을 지도(guide)하는데 활용된다. Burst 이미지이기 때문에 발생하는 프레임간의 변동을 반영하기 위해 offset 정보 역시 GCP feature로부터 추정된다.

<br/>

## 1. Introduction

모자이크된 CFA 데이터에서 missing color들에 대한 interpolation을 수행하여 복원하는 과정을 demosaciking이라 하고, 데이터에는 노이즈가 존재한다 (특히 저조도 환경에서). 카ㅔ라의 Image Signal Processing pipeline에서 denoisng, demosaicking은 고품질의 이미지를 얻는데 매우 중요한 역할을 수행한다.

이전의 demosaicking, denoising은 서로 독립적으로 설계되고 ISP 과정에서 sequential하게 처리 되었으나, 최근에는 JDD 기법들이 제안되고 있다. 기존의 JDD 기법들은 non local self similarity 같은 image prior에 의존하는 경향이 있는데, 이와 같은 handcrafted prior들은 이미지의 복잡한 구조를 복원하는데 있어 한계가 있다.

 - demosaicking, denoising을 sequential하게 처리하면 demosaicking error가 denoising process를 더 어렵게 만들거나, denoising artifacts가 demosaicking process에서 더 확대되는 문제가 발생할 수 있음.

이에 따라 최근의 딥러닝 기반 JDD는 CNN을 활용한 data driven method를 활용한다. 해당 딥러닝 모델은 noisy, mosaicked image와 clean cull GT의 pair 데이터를 통해 학습된다.
`By learning deep priors from a large amount of data, those CNN based methods achieve much better JDD performance than traditional model based methods.`

기존의 JDD 기법들은 단일 CFA 이미지를 대상으로 하기 때문에 JDD-S라고 불리며 현실의 CFA 데이터에 적용하기에는 다음과 같은 측면에서 한계가 있다.
 1. 강한 noise가 존재할 경우 큰 성능 저하  ex) 저조도 환경, 렌즈와 센서가 작은 스마트폰 카메라
 2. Additive White Gaussian Noise를 가정하기 때문에 현실의 복잡한 노이즈를 정확히 설명할 수 없음

최근에 단일 이미지 대신 burst 이미지를 사용함으로써 denoising 성능을 크게 향상시킬 수 있다는 것이 밝혀졌다 (특히 저조도 환경에서). 따라서 본 연구는 JDD-B를 제안하며, 보다 현실적인 노이즈 모델링을 통해 고품질의 비디오 시퀀스(burst images)에서 ISP 파이프라인을 반전시키고, 노이즈를 추가하여 노이즈가 많은 burst 이미지를 깨끗한 GT 이미지로 합성할 수 있다.

Single chip 디지털 카메라로 촬영한 이미지의 G 채널은 다음과 같은 측면에서 다른 채널 보다 품질과 SNR이 높고, 텍스쳐 정보를 많이 보유하고 있다.
 1. 대부분의 CFA 패턴에서 G 채널이 더 많이 샘플링됨
 2. G 채널의 sensitivity가 더 높음

따라서 본 연구는 Green Channel Prior를 활용하여 JDD-B 네트워크를 설계한다. `In GCP-Net, we extract the GCP features from green channel to guide the deep feature modeling and upsampling of the whole image. The GCP features are also utilized to estimate the offset within frames to relief the impact of noise.`

![image](https://user-images.githubusercontent.com/44194558/148670211-cd0937d9-2174-4ce3-bb30-89021b5b95ec.png)


<br/>

## 2. Related Work

Denoisng, demosaicking을 개별적으로 수행하는 것보다 JDD-S가 높은 성능을 보임. 최근에는 green channel prior를 활용하여 upsampling을 지도하는 방식이 제안되기도 했음 (참고 : https://github.com/gymoon10/Paper-Review/blob/main/Computer%20Vision/Joint_Demosaicing%26Denoising_with_Self_Guidance.md). 본 연구는 G 채널의 정보를 활용하는 Self guided Netwokr에서 더 나아가, 칼라 채널간의 noise imabalance를 분석한다.

Burst 이미지 처리는 카메라와 객체의 움직임으로 인한 프레임간의 offset을 추정해야 하는 문제가 생긴다. 제안된 alignment (서로 다른 프레임간에서 움직인 객체들을 알맞게 대응) 프레임워크는 크게 3가지로 분류된다.

1. Pre-alignment
  - Optical flow 사용. 큰 움직임과 심한 노이즈가 있는 경우 정확한 optical flow를 계산하기 어렵다는 단점.

2. Kernel based
  - CNN을 사용하여 spatially varing kernel을 예측. Aligning, denoising을 동시에 수행.

3. Aligning in feature domain
  - Video super resolution에 있어 SOTA 성능. 
  - Lie et al : Localization 네트워크를 활용하여 deep feature들 간의 spatial transform 파라미터를 추정.
  - TDAN, EDVR : Feature domain 간의 shift를 align하기 위해 offset을 추정. Spatial transform의 GT를 필요로 하지 않음.

`In this work, we perform alignment in feature domain and utilize deformable convolution to implicitly compensate for offsets. Moreover, we design an inter-frame module which not only utilizes multi-scale information, but also considers temporal constraint.`

* Deformable convolution 참고 : https://jamiekang.github.io/2017/04/16/deformable-convolutional-networks/

<br/>

## 3. Methods

### A. Problem Specification

![image](https://user-images.githubusercontent.com/44194558/148670651-a388cfc8-9eb7-4cd3-a853-cb2b3f3f17eb.png)

현실의 raw image에서 노이즈는 signal dependent. 광학 센싱에 의한 shot noise는 포아송 분포를 따르고, 판독 회로에 의한 read noise는 가우시안 분포를 따른다. 특정 시점 t의 clean raw 이미지 x에 대응되는 noisy raw 이미지는 다음과 같이 표현된다.

![image](https://user-images.githubusercontent.com/44194558/148670713-8c32f05f-37e7-46e8-b810-69f84a0ecf0d.png)

 - `We can synthesize noisy burst images with clean ground truth by reversing the ISP pipeline on high quality video sequences and adding noise into them.` 

<br/>

### B. Green Channel Prior

CMOS 센서는 다른 파장이나 색의 빛에 대해 다른 sensitivity를 가지고 있고, 대부분의 조도 환경에서의 Bayer pattern CFA 이미지의 G 채널은 R, B에 비해 더 밝다. 위에서 언급했듯 현실의 노이즈는 Poisson shot noise를 포함하고 SNR(신호 대 잡음비)에서 신호와 노이즈 간에는 square root 관계가 성립한다. 밝기가 높은 G 채널은 다른 채널에 비해 SNR이 높다.

  - **SNR** : 잡음이 잡힐 때 최종 신호가 얼마나 큰지 (클 수록 음질 등 신호의 품질이 좋음) 
  - 참고 :https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=true4voice&logNo=220948728068

다음의 그림에서 대부분의 노이즈 이미지에서 G 채널의 SNR이 높은 것을 확인할 수 있고, G 채널은 다른 채널에 비해 2배 많은 샘플링 비율을 가진다.

![image](https://user-images.githubusercontent.com/44194558/148670923-bb16c63a-a5a4-4c53-b467-ae4f2253d60a.png)

`Overall, the green channel preserves better image structure and details than the other two channels. In this paper, we call the prior knowledge that the green channel has higher SNR, higher sampling rate and hence better channel quality than red/blue channels the green channel prior (GCP), which is carefully exploited in this paper to design our JDD-B network.`


### C. Network Structure

![image](https://user-images.githubusercontent.com/44194558/148670964-acaf969d-9fdc-4658-b58c-923acd5bf146.png)

 - Raw noisy 이미지 y를 동일한 크기의 RGGB 4 채널로 분해
 - y의 noise map 지정 (S.D of signal dependent noise at that position)
 - GCP-Net의 입력으로 noisy raw 이미지들과 대응되는 noise map들을 제공

<br/>

네트워크는 크게 2개의 brach로 구성.

**GCP Branch**

Noisy G 채널과 noise level map과의 concat을 입력으로 받아 freen feature f_t_g 출력

![image](https://user-images.githubusercontent.com/44194558/148671163-1177108d-437b-4993-87d0-19d05381b745.png)

 - M_g는 다수의 Conv+LReLU block으로 구성
 - 해당 GCP feature을 reconstruction branch의 layer wise guide 정보로 사용 

<br/>

**Reconstruction Branch**

Full color 이미지를 추정, 복원하기 위해 burst 이미지, noise map, GCP feature를 활용. 크게 IntraF Module, InterF Module, Merge Module로 구성.

`1. IntraF Module` : 매 frame의 deep feature를 모델링하고, feature extraction 과정에서 GCP feature를 활용

`2. InterF Module` : Feature domain에서 DConv를 사용하여 프레임간의 shift를 조정. Alignment 과정에서의 노이즈를 줄이기 위해 cleaner GCP feature로부터 offset 추정.

`3. Merge Module` : Aligned feature, GCP feature를 모두 활용하여 adaptive upsampling을 통해 full-resolution 이미지를 복원

<br/>

### D. Intra-frame Module

![image](https://user-images.githubusercontent.com/44194558/148671341-d0a89875-3c5c-4485-8493-ab08437aa8cd.png)

 - y_t, m_t에 대한 간단한 conv 연산을 통해 initial feature M_0 계산 
 - 이후 4개의 concatenated GCA block을 통과
   - 이 과정에서 GCP feature들이 feature extraction의 guide역할을 수행
   - Spatial, channel dependent noise를 잘 처리할 수 있도록 dual attention 매커니즘이 사용됨
 - GCA block에선 layer wise guidance 사용  

![image](https://user-images.githubusercontent.com/44194558/148671399-0de98ce6-8247-46ea-bc33-b8d48178a964.png)

Green Channel Attention block의 구조는 다음과 같음

![image](https://user-images.githubusercontent.com/44194558/148671447-62bbd241-3c74-4308-85dd-a7abba1c9091.png)

 - GCP feature들이 feature extraction을 지도하기 위해 사용됨

#### 1. Burst 이미지의 feature map, GCP feature를 활용하여 enhanced feature 생성

![image](https://user-images.githubusercontent.com/44194558/148671690-2248be89-55e2-4fa9-89b8-fac5b809fc0d.png)

 - pixel wise scaling and bias를 사용해 GCP 정보를 결합

#### 2. 2개의 residual block에 의해 Green Guided unit 생성


![image](https://user-images.githubusercontent.com/44194558/148671703-198e6bb7-e27e-4216-95c9-acf20ffe3a30.png)

 - learned features 생성

#### 3. Channel Attention & Spatial Attention

`As normal Conv layers treat spatial and channel features equally, it is not appropriate to handle the real-world noise which is channel and spatial dependent. To further enhance the representational power of standard Conv+ReLU blocks, channel attention and spatial attention [34], [35], [36] are designed to model the cross- channel and spatial relationship of deep features.`

추정된 attention map을 활용하여 2의 learned features를 rescaling (z_c, z_s)

**CA** 

 1. Channel descriptor 생성을 위해 Global Average Pooling을 사용하여 (H, W, C) feature map을 (1, 1, C)로 변환
 2. 2번의 1x1 conv, 시그모이드 활성화를 통해 channel descriptor를 처리

 **SA**
 
 추정된 spatial attention map을 사용하여 deep feature간의 spatial dependency를 모델링

 1. Pooling 방식 대신 2회의 Conv 연산을 통해 adaptive하게 획득됨 

<br/>

위의 과정을 거쳐 GCA block의 최종 output은 다음과 같음

![image](https://user-images.githubusercontent.com/44194558/148671818-02333217-f064-4254-a35d-5ec70eaba7e4.png)

<br/>

### Inter-frame Module

IntraF Module의 feature를 InterF module의 reference frame feature와 align. 프레임간의 temporal dependency를 모델링하는 것이 목적.

![image](https://user-images.githubusercontent.com/44194558/148671856-542c351a-9b0b-433f-9439-a9b68ac3ab7c.png)

 - 프레임간의 offset을 조정하기 위해 deformable conv 사용
 - 심한 노이즈와 인접 프레임간의 상관성을 배제하기 위해 offset 추정에서 GCP feature를 사용
 - 큰 motion을 처리하기 위해 pyramidal processing이 사용됨
 - LSTM regularization 활용 

#### 1. t 번째 프레임의 특정 s 스케일에 대해 inter-frame GCP feature 계산

![image](https://user-images.githubusercontent.com/44194558/148671973-ce753099-ab80-40e1-a9b4-81551a1e4f05.png)

#### 2. ConvLSTM을 사용해서 regularization

 - ConvLSTM : 2D sequence data modeling method (특정 s scale에 대해 여러 프레임의 burst 이미지들이 존재함)


![image](https://user-images.githubusercontent.com/44194558/148672002-f44c3025-689e-4bf6-8eb9-ebdb41af634a.png)


#### 3. Offset 추정

LSTM 만으로는 복잡한 motion을 처리하기 어려움. 큰 motion을 처리하기 위해 multi-scale 정보들을 결합하여 offset을 보다 정확하게 추정.

![image](https://user-images.githubusercontent.com/44194558/148672044-df7e2326-b84e-42be-bbbc-f7aba7192eda.png)


#### 4. Aligend feature 계산

![image](https://user-images.githubusercontent.com/44194558/148672078-2ac6fb53-abe7-455f-83ad-34a3f299cf37.png)


최종 aligned feature는 다음과 같이 계산됨

![image](https://user-images.githubusercontent.com/44194558/148672099-c2df5243-299c-4dbb-8228-d80f1e59c7ab.png)


<br/>

### F. Merge Module

#### 1. Aligned features are fistly concatenated and adaptively merged  

![image](https://user-images.githubusercontent.com/44194558/148672155-66b8c139-4f32-4ab6-af3a-fbf9e37532a6.png)

#### 2. GCP adaptive upsampling을 통해 full resolution feature 생성

![image](https://user-images.githubusercontent.com/44194558/148672188-dda636d8-9a3b-403d-98aa-eca5e6e7fd00.png)

 - f_g : Green Guided feature
 - Up-sampling interpolation을 위해 Transpose Conv 사용

#### 3. 최종 예측 

![image](https://user-images.githubusercontent.com/44194558/148672212-3c70b401-c82e-4ead-87b7-3cda5dc98c9b.png)

 - M_u 로는 3 scale U-NET 사용
   - multi scale information 활용 & enlarging receptive field
   - M_u의 upsampling, downsampling 방법으로는 3x3의 strided, transpose conv로 구성 

<br/>

### G. Loss

#### 1. Reconstruction loss in the linear color space

![image](https://user-images.githubusercontent.com/44194558/148672254-475cf22c-f4ac-418c-a686-d823f464c9e7.png)

#### 2. Loss in sRGB color space

![image](https://user-images.githubusercontent.com/44194558/148672479-1e0fe876-b516-455e-9a9a-1e9465c4b72d.png)

최종 손실은


![image](https://user-images.githubusercontent.com/44194558/148672498-90a054ef-0091-4665-8bd8-341875e24560.png)
