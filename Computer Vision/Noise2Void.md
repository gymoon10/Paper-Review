# Noise2Void

<br/>

## Abstract

Image denoising의 딥러닝 기법들은 주로 noisy input-clean target images의 pair를 활용하여 학습을 진행하는 방법론들이 주류였으나, 최근에는 clean target없이 independent pairs of noisy images로만 학습이 가능한 Noise2Noise(N2N)의 연구 또한 이루어졌다.

본 연구가 제안하는 Noise2Void(N2V)는 오직 noisy image 그 자체로만 학습이 가능하다는 점에서 noisy image pairs를 필요로 하는 N2V와 차별화된다. 의료 영상 분야의 경우 training targt(clean or noisy)를 확보하기 어려운 케이스가 많이 존재하기 때문에 N2V는 이러한 문제 상황에 큰 도움이 된다.

<br/>

## Introduction

Image denoising 태스크는 x=s+n으로 표현 가능하다 (x: noisy image / s: signal / n: signal degrading noise).

- `Image denoising is the task of inspecting a noisy image x = s + n in order to separate it into two components`

Denoising은 대게, 픽셀 값 s가 통계적으로 독립이 아니라는 가정을 따르기 때문에, unobserved pixel에 대한 image context를 이용하여 해당 픽셀을 예측하는 것이 가능하며, 대부분의 연구들은 training pairs(noisy input-clean target)를 필요로 한다.

 - `In recent years, convolutional neural networks (CNNs) have been trained in various ways to predict pixel values from surrounding image patches, i.e. from the receptive field of that pixel.`

위의 방법들은 GT가 획득 불가능한 경우 학습할 수 없다는 단점이 있기 때문에 N2N이 제안되었다. N2N은 noisy input을 clean target으로 mapping하는 대신, pair of independently degraded versions of same training imgae (s+n, s+n')간의 mapping을 학습하려고 시도한다.

하지만, N2N은 independent noises (n, n')을 갖는 same content (s)를 캡쳐하여 두 이미지를 획득해야 하는 단점이 존재한다. 그렇기에 본 연구가 제안하는 N2V는 기존의 지도학습 방법들과는 달리 noisy image pairs, clean target images가 없어도 학습이 가능하다.

 - `However, unlike N2N or traditional training, N2V can also be applied to data for which neither noisy image pairs nor clean target images are available, i.e. N2V is a self-supervised training method.`

N2V는 다음과 같은 통계적 가정에 근거한다.

 - signal s is not pixel wise independent
  
 - noise n is conditionally pixel-wise independent given the signal s


`In  summary,  our  main  contributions  are: Introduction of NOISE2VOID, a novel approach for training denoising CNNs that requires only a body of single, noisy images.`

<br/>

## 2. Related Work

`With N2V we have to stick to the more narrow task of denoising, as we rely on the fact that multiple noisy observations can help us to retrieve the true signal.`

Offline에서 학습되어, GT annotated training set으로부터 information을 추출하는 딥러닝 기반 선행 연구

 - Denoising을 regression task의 관점에서 CNN을 예측과 GT간의 loss를 최소화
 - residual learning을 기반으로한 다양한 연구 존재

<br/>

Interneal statistics method는 GT를 사용한 학습이 필요 x

 - 대신 직접 test image에 바로 적용되어 모든 필요한 information을 추출
 - N2V는 test image에 대해 직접 학습이 가능하다는 점에서 internal statistics method 카테고리에 포함

<br/>

`Like N2V, Van Den Oord et al. train a neural network to predict an unseen pixel value based on its surroundings. The network is then used to generate synthetic images.` 

 - N2V는 일종의 regression task. 각 픽셀 예측값의 확률 분포를 계산함.
 - always mask the central pixel in a square receptive ﬁeld

<br/>

## 3. Methods

<br/>

### 3.1 Image Formation

Noisy image의 생성은 x=s+n의 식으로 표현이 가능하고, 다음과 같은 joint dist를 따르며, 픽셀 값들은 통계적으로 독립적이지 않다는 것을 보인다.

![image](https://user-images.githubusercontent.com/44194558/156135029-4a24ae45-ecfe-44b0-a320-da0166ca3a0f.png)

또한, Noise는 다음과 conditional dist를 따르며 zero-mean(expectation)을 갖고, signal에 대해 독립적이라는 것을 보인다. 이와 같은 가정들로 인해 E[noisy image x]=clean signal s가 성립하게 된다.

![image](https://user-images.githubusercontent.com/44194558/156135480-1df51330-56e9-4556-a53b-723b101a4ce1.png)

<br/>

### 3.2 Traditional Supervised Training

Noise image x를 입력받아 signal s를 예측하는 FCN을 훈련. 

 - CNN 네트워크의 output에서 각 픽셀에 대한 prediction s_hat_i는 입력 픽셀들에 대한 receptive field x_RF_i를 가짐.
 - 픽셀의 receptive field는 해당 픽셀 주위에 있는 square patch


이러한 관점에서 본다면, CNN 네트워크는 patch의 정 가운데 존재하는 단일 픽셀 i에 대해 patch x_RF_i를 입력받아 예측값인 s_hat_i를 출력하는 함수이다 (surrounding image context이용). Overlapping patch들을 추출하고, 이를 네트워크에 일일히 입력함으로써 전체 이미지에 대한 denoising이 가능.

![image](https://user-images.githubusercontent.com/44194558/156136453-a66b0dd3-626d-4124-a49e-b9cea30fe032.png)

전통적인 supervised training의 training pairs는 (noisy input image, clean GT)인데, 이를 patch based CNN의 관점에서 보면 (patch around pixel i, extracted from training input imag j, corresponding target pixel valure)로 볼 수 있다. 그리고 다음과 같은 손실 함수에 의해 학습된다.

![image](https://user-images.githubusercontent.com/44194558/156137802-87b029dc-fb95-42b7-822b-c0460bbcd849.png)

<br/>

### 3.3 Noise2Noise Training

다음과 같은 noisy image pairs를 이용하며, noisy input에서 noisy target으로의 mapping을 학습하지만, 여전히 correct solution에 수렴하게 된다 (noisy input에 대한 기댓값이 clean signal과 같다는 사실에 기반).


![image](https://user-images.githubusercontent.com/44194558/156137972-31e69404-ec4e-4fe2-b10d-1170d362f6f7.png)

Patch based관점에서 N2N의 training pairs를 보면

![image](https://user-images.githubusercontent.com/44194558/156138347-444c84ae-78ad-45eb-b162-f1fd7d1eb635.png)

<br/>

### 3.4 Noise2Void Training

Input, target모두 single noisy training image로부터 추출 (patch를 입력으로, center pixel을 target으로 하면 단순한 identity mapping을 학습하게 됨).

네트워크 architecture는 특별한 receptive field를 가짐.

 - center에 blind spot이 존재 (일종의 masking)
 - CNN 예측은 blind spot 자리에 있는 input pixel을 제외한 square neighborhood의 모든 input pixel의 영향을 받음 (여전히 surrounding, image context를 반영하여 해당 픽셀 값을 예측할 수 있음)


![image](https://user-images.githubusercontent.com/44194558/156139251-7d587c84-a8b8-4240-87b6-2dc59c1306a8.png)

다음과 같은 empirical risk를 최소화하는 방향으로 학습됨

![image](https://user-images.githubusercontent.com/44194558/156139451-72dac223-b209-4e43-bb72-041b7bb862dc.png)

Blind spot network는 상대적으로 적은 정보만을 활용하기 때문에 (blind spot pixel은 활용 x), normal network보다는 약간 낮은 성능을 보일 수 있으나, identity를 학습하지 못한다는 큰 장점이 있다.

 - 주어진 signal과 독립적인 noise를 가정하면(eq.3), 이웃의 픽셀들은 noise value에 어떠한 정보도 전달하지 못함
 - signal은 statistical dependency를 가정하기 때문에(eq.2) 네트워크는 여전히 주변의 맥락 정보를 활용하여 signal value s_i를 추정할 수 있음
 - input patch, target value를 동일한 noisy training이미지로부터 가져옴

![image](https://user-images.githubusercontent.com/44194558/156140195-d9ed0b2b-ea7b-41b6-bda1-d686cf861c7c.png)

