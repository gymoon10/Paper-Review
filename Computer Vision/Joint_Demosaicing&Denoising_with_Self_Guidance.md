# Joint Demosaicing and Denoising with Self Guidance

<br/>

## Abstract

Joint Demosaicing & Denoising(JDD) 분야에서 기존의 방식들은 입력인 Bayer raw 이미지를 4 채널의 RGGB 이미지로 decompose한 후 신경망 네트워크의 입력으로 제공한다. 이와 같은 방식은 G 채널이 2번 샘플링 된다는 문제점을 안고 있다. 따라서 본 논문은 다음과 같은 방법론을 제안한다.

1. SGNet : Self-guidance network (the green channels are initially estimated and then works as a guidance to recover all missing values in the input image )

2. Density map guidance (to help the model deal with a wide range of frequencies)

<br/>

## 1.Introduction

Demosaicing은 RGGB Bayer pattern 등의 Color Filter Array를 적용시킨 불완전한 정보를 바탕으로 full-resolution의 칼라 이미지를 복원하는 태스크이다. 결측치를 보간하는 방식이기 때문에 ill-posed problem이고 noise에 의한 오염이 존재하기 때문에 난이도가 높은 low-level vision 태스크. Demosaicing, denoising을 sequential하게 처리하는 기존의 방식과 달리 최근의 방식을은 joint approch를 사용한다.

Bilinear interpolation을 사용한 demosaicing은 에지 근방에 zippering artifact가 발생하는 문제가 있어, 이를 보완하고자 edge-adaptive interpolation 방식이 제안되었음. Super resolution 같이 이미지의 유용한 정보들을 활용하여 결측 픽셀을 보간하는 방식과 유사한 면이 있으나, high frequency region에 존재하는 more pattern등의 문제가 있어 여전히 해결하기 어려운 태스크이다.

최근에 제안된 딥러닝 방법들은 효과적으로 이미지 채널 간, 채널 내 (intra, inter) 정보를 사용하여 결측 픽셀들을 복원하고자 한다. CNN을 사용하여 RGGB 채널간의 상관성을 학습하여 결측 픽셀들을 보완하고자 한다. 하지만 G가 R, B에 비해 많이 샘플링된다는 prior 정보를 충분히 이용하지 못하고 있다. 픽셀 복원의 성능을 향상시키기 위해서는 G 채널의 정보를 충분히 이용해야만 한다. 실제로 딥러닝 이전에는 G 채널을 먼저 복원한 이후 G와 다른 채널들간의 채널 간 상관성을 고려하여 RGB 채널을 복원하는 방식이 제안되기도 했다.

`Therefore, making full use of the information from the green channels can be beneﬁcial to the recovery of missing pixel values.`

전형적인 CNN은 모든 이미지와 (이미지 내의) region들에 동일한 파라미터를 일괄적으로 적용하기 때문에 (content agnostic) 이미지와 region들 간에 상이한 frequency, noise에 대응하기 어렵다는 단점이 있다. 

 `In this work, similar to the traditional methods, we ﬁrst make an initial estimation based on only the green channels. The initially recovered green channels work as the guidance to conduct spatially adaptive convolution across an image. In this way, the rich information with the green channels is integrated and employed differently at different positions.`

 또한 frequency에서 차이가 있는 region들은 이미지 복원의 난이도가 상이하기 때문에, 모델로 처리하기 어려운 특정 region들을 아는 것이 demosaicing, denoising에 있어 매우 중요하다. `Estimating a density map for an image and feeding it into a model allow the model to handle a wide range of frequencies` (일종의 attention map과 유사한 역할 수행)

 <br/>

 ## 2.Related Work

 ### 2.1 Joint Demosaicing and Denoising

 딥러닝 기법을 활용하여 demosaicing, denoising을 동시에 수행하는 것이 높은 성능을 보임. (can get rid of the accumulation of errors after one's processing)

 ### 2.2 Guided Image Restoration

많은 이미지 restoration 태스크에서 이미지에 대한 external information을 활용함. 예를 들어 Bilateral filter 등은 external images를 활용하여 필터 파라미터를 조정하고, 딥러닝 방법들은 super resolution 태스크에서 guidance information을 활용하여 이미지를 복원함. 

ex) RGB 이미지를 guide로 사용하여 depth map을 upsampling하여 복원, semantic information을 supre resolution의 guide로 활용

위의 방법들은 모두 external guidance information을 활용. 본 연구는 mining하기 어렵고 prior knowledge를 요구하기도 하는 self guidance strategy를 활용하며 green channel guidance, density map guidance를 제안한다.

<br/>

## 3.The Proposed Method

### 3.1 Overview

`Making full use of the information within an input RGGB raw image is crucial`

입력 이미지와 함께 green channel, density information을 guide로 활용.

![image](https://user-images.githubusercontent.com/44194558/148518278-afb7fad2-d303-49af-9e4e-43453e78d965.png)


1. RGGB 이미지를(2H x 2W)를 4개의 채널로 분해
2. G 채널의 결측 픽셀 elements를 예측 (G 채널에 보다 많은 정보가 존재) -> G 채널에 대한 estimate를 guidance로 사용 
3. Density map 계산 (to represent the difﬁculty levels of different regions)
4. Density map을 네트워크의 additional input으로 제공

<br/>

### 3.2 Density-map guidance

한 이미지내에 complex region with high frequencies, smooth region with low frequencies가 공존하기 때문에 전체 이미지를 일률적으로 처리하는 것은 sub-optimal한 방법. 따라서 density map을 통해 네트워크에 이미지의 각 position들에 대한 난이도 정보를 제공한다.

Density map에서 dense texture는 high frequency pattern에 less texture는 low frequency pattern에 대응.

![image](https://user-images.githubusercontent.com/44194558/148518986-053400e0-0d95-42e3-910d-2af18b8c2453.png)

 - I_gray : 4 채널의 평균 이미지

계산된 density map은 decompose된 4 채널 이미지들과 concat되어 main branch의 입력으로 제공됨.

<br/>

### 3.3 Green-channel guidance

Bayer pattern에서 G 채널이 많기 때문에 G 채널의 픽셀 element를 복원하는 것이 보다 쉬운 태스크. G 채널에 대한 initial estimation은 나머지 채널의 복원에 있어 효율적임. (`Usually in an RGB image, the green channel shares similar edges with the red and blue channels; having an initial estimation of the green channel can also beneﬁt the reconstruction of the red and blue channels.`)

1. Bayer 이미지의 2x2 블락마다 G 픽셀들을 추출하고, noise map과 함께 green channel reconstruction banch의 입력으로 제공됨.
   - green channel reconstruction banch는 Residual in Residual Dense Block으로 구성 ( `allows local  information  to  be  sufﬁciently  explored  and  extracted.`)
   - ![image](https://user-images.githubusercontent.com/44194558/148520322-f623dadf-f73f-4aae-b094-1a959915504b.png)

2. green channel reconstruction branch의 마지막 단계에서는 depth-to-space layer가 사용되어 복원된 green channel I_hat_g 생성
   - super resolution과 유사하게 missing elements의 복원과 2xupsampling을 필요로 함

3. Fusion 단계에서 2의 결과인 estimated green channel에 대한 합성곱 연산 수행
   - main reconstruction branch에서 중간 feature map 출력에 대해 spatially(pixel) adaptive convolution이 사용
   - SAC는 content aware (<-> content agnostic), `the convolution is conducted differently at different positions` 
   - ![image](https://user-images.githubusercontent.com/44194558/148521294-d59652f5-e110-4843-abf3-1ec8cbe3014e.png)
      - G는 i, j 두 position간의 거리(유사도)에 따라 다른 가중치 -> `core to make the convolution operation spatially adaptive` 

참고 : https://www.youtube.com/watch?v=gsQZbHuR64o

<br/>

### 3.4.1 Adaptive-threshold edge loss

고주파 디테일이 많은 지역에 가중치를 두기 위한 loss

`Regions with many high frequency details are more important and should draw more attention than easy-to-recover regions during training`

![image](https://user-images.githubusercontent.com/44194558/148521966-89a3dd38-04b9-42f6-a81f-8c215e691916.png)

 - 이미지를 여러 개의 patch로 분할하고 patch 별로 adaptive ths를 계산 (edge가 많으면 낮은 ths)
 - non edge 부분이 더 많기 때문에 loss를 계산할 때 적절한 가중치 B 설정

<br/>

### 3.4.2 Edge-aware smoothness loss

Texture 영역에서 edge를 유지하면서 부드러운 영역에서 노이즈를 동시에 제거하는 방법을 학습

![image](https://user-images.githubusercontent.com/44194558/148523307-3dd89313-ddd5-4741-b878-a97903d52283.png)