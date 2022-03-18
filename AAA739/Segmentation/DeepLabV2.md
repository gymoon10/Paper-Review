# DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs

<br/>

참고

https://www.youtube.com/watch?v=JiC78rUF4iI&t=807s

https://www.youtube.com/watch?v=HTgvG57JFYw&t=1223s

https://gaussian37.github.io/vision-segmentation-aspp/

https://m.blog.naver.com/laonple/221017461464

<br/>

## Abstract

<br/>

`In this work we address the task of semantic image segmentation with Deep Learning and make three main contributions that are experimentally shown to have substantial practical merit.`

<br/>

### 1. Atrous Convolution (Dilated Conv)

<br/>

  - Conv with up-sampled filters (desne prediction task에 있어 매우 유용)
  
  - 필터 내부에 zerro padding을 추가하여 적은 계산 비용으로 receptive field의 크기를 늘림 
      
      - 정보의 손실을 최소화 하면서 해상도를 크게 가져갈 수 있음
  
      - 보다 넓은 global context 정보를 반영할 수 있음 (일반적인 conv의 경우 공간 정보의 손실이 있기 때문의 up-sampling시 해상도가 떨어지는 단점 존재)

<br/>

### 2. Atrous Spatial Pyramid Pooling

<br/>

  - `To robustly segment objects at multiple scales`
  
  - Multi-scale에 보다 잘 대응할 수 있도록 atrous conv의 확장 계수를 다양하게(6, 12. 18, 24) 적용하고, 그 결과를 취합 (Maximum value만 취함)
      
      - 다양한 recpetive field 정보를 이용할 수 있는 장점
  
  - trous는 프랑스어로 hole (zero-padding)을 의미 

<br/>

### 3. Fully Connected Conditional Random Field

<br/>

  - Localization 성능 향상 효과
    
     - 일반적인 classification 모델을 기반으로 segmentation을 수행하게 되면 계속 feature map의 해상도가 줄어들기 때문에 detail한 정보를 얻을 수 없음
     
     - 위와 같은 문제를 해결하고자 Skip-conn, 마지막의 pooling layer 2개를 atrous conv로 대체하는 방식이 제안되었지만 한계가 존재함
      
  - Classification network (ex. VGG16) 뒤의 Post-processing으로 사용함으로써 detail을 높임 (coarse to fine) 
  
  - DCNN에서 여러 단계의 conv+pooling을 거치기 때문에 shor-range가 아닌 전체 픽셀을 모두 연결하는 fully connected 방식을 사용함

<br/>

## 1. Introduction

<br/>

분류 모델들은 DCNN layer를 거치면서 transformation에 강건한(invariant) 추상적인 data representation을 추출하는데, 그 과정에서 공간 정보(spatial information)이 추상화되기 때문에(coarse) dense prediction인 segmentation에는 적합하지않다. 일반적인 DCNN을 segmentation에 적용하는데 있어 다음과 같은 이슈들이 존재한다.

**문제**

1. 해상도의 감소
    
   - 반복된 pooling, downsampling(with stride) 연산


2. multiple scale로 존재하는 객체들

   - Atrous conv만으로는 multi-scale을 보기 어려움 (해상도 유지가 목적)
   
   - 입력을 multi-scale로 넣어주면 성능은 무조건 향상되지만 계산 비용이 증가하기 때문에 보다 효율적인 방식이 필요함 

3. DCNN의 invariance 특성으로 인한 localization 성능 감소  

   - `object centric classifier requires invariance to spatial transformations, inherently limiting the spatial accuracy of a DCNN`


<br/>

**해결**

1. Atrous conv 

  - `In practice, we recover full resolution feature maps by a combination of atrous convolution, which computes feature maps more densely, followed by simple bilinear interpolation of the feature responses to the original image size.`
  
  - VGG16에서 pooling을 대체 
  
  - **Dense feature extraction** 

<br/>

2. ASPP
   
  - `We propose a computationally efficient scheme of resampling a given feature layer at multiple rates prior to convolution.`
  
  - 다양한 확장 계수(sampling rates)를 가진 atrous conv를 병렬적으로 적용한 후, 그 결과들로부터 resampling 
  
  - **Employ a series of a atrous conv layers with increasing rates to aggregate multiscale context** 

3. Fully Connected CRF

  - `Boost our model's ability to capture fine details`
  
  - `Combine class scores computed by multi-way classifiers with the low level information(high resolution) captured by the local interactions of pixels and edges [23], [24] or superpixels [25]. ` 
  
  - DCNN의 pisel-level classifer와 결합함으로써 보다 나은 성능 기대
  
  - **Refinement of the raw DCNN scores** 

<br/>

![image](https://user-images.githubusercontent.com/44194558/158924285-c6518933-c8fb-456f-8904-ad48946ceef0.png)

이미지 분류를 위해 학습된 DCNN 네트워크가 아래의 방식들을 통해 semantic segmentation용도로 repurpose됨

 - Atrous conv를 통해 feature map의 해상도를 증가
 
 - Bilinear interpolation을 통해 score map에 대해 원본 이미지의 해상도를 복원 
 
 - Fully connected CRF를 통해 segmentation 예측 결과를 refine

<br/>

## 2. Related Work

<br/>

최근 몇 년동안 이미지 분류 태스크에 뛰어난 효과를 보인 딥러닝 모델들이 segmentation 태스크에도 transfer되기 시작했다. 

그 중에서도 본 연구는 입력 이미지에 대해 DCNN을 direct하게 적용하여, 마지막의 fully connected layer를 conv layer로 대체하는 방식에 기반하고 있다. 

그러면서도 spatial localization 성능을 개선하기 위해 up-sample, intermediate feature map과 score를 concat, 예측 결과를 refine하는 방식을 사용하고 있다.

`Our work builds on these works(원본 이미지에 DCNN을 direct하게 적용하는 방식), and as described in the introduction extends them by exerting control on the feature resolution(Atrous conv), introducing multi-scale pooling techniques(ASPP) and integrating the densely connected CRF of [22] on top of the DCNN.`

<br/>

## 3. Methods

<br/>

### 3.1 Atrous Convolution for Dense Feature Extraction and Field of View Enlargement

<br/>

`We can compute responses at all image positions if we convolve the full resolution image with a filter ‘with holes’, in which we up-sample the original filter by a factor of 2, and introduce zeros in between filter values.`

 - Filter size가 증가하지만 non-zero pixel만 연산에 활용하기 때문에 연산량, 파라미터 수 측면에서 효율적임.

Atrous conv는 input feature의 FOV(Field of View)를 보다 넓게 확장함.
작은 FOV를 통해 localization의 정확성과, 큰 FOV를 통해 context 이해를 동시에 다룰 수 있음.

<br/>

![image](https://user-images.githubusercontent.com/44194558/158934900-d9fee3c2-83a7-4832-b7ca-4e69483428db.png)

 - 일반적인 conv의 경우 stride=1, pad=1을 통해 feature 크기 유지
 
 - Atrous conv는 stride=1, pad=r을 통해 feature 크기 유지. 여기서는 filter간의 거리를 2로 설정함.
 
 - 같은 크기의 kernel를 사용함에도 불구하고 atrous conv가 더 넓은 범위의 입력을 cover하는 것을 확인.  

<br/>


![image](https://user-images.githubusercontent.com/44194558/158935151-1e1634d8-6044-40aa-b8fd-d2d5ce96d16b.png)

<br/>

![image](https://user-images.githubusercontent.com/44194558/158935079-971a3e34-a9ef-4b51-87f3-43d7f21b12de.png)

<br/>

반복적인 conv연산으로 인한 spatial resolution의 감소를 해결하기 위한 방식으로는 deconvolution 방식도 있지만, 계산 비용과 메모리 측면에서 비효율적인 반면, Atrous conv는 효율적이다.

다음 수식에서 x가 입력이고 w가 filter이다. 즉 입력 이미지에 filter를 곱할 때 확장 계수(sampling rate) r의 값에 따라 입력 픽셀을 띄엄 띄엄 선택하여 filter와 곱한다.

![image](https://user-images.githubusercontent.com/44194558/158935206-95c57110-a240-45f1-9a7c-fa88e4fbff30.png)

<br/>

ResNet이나 VGG16의 연속적인 conv layer를 atrous conv layer로 대체하면 feature response를 원본 이미지의 해상도에서 계산할 수 있다는 장점이 있지만, 비용 측면에서 비효율적이기 때문에 Bilinear interpolation과 결합하여 사용함.

 - Bilinear interpolation으로 feature map의 해상도를 원본 이미지의 크기만큼 복원한 후, atrous conv 사용.
 
 - 적정한 선에서 atrous conv를 bilinear interpolation으로 대체하는 만큼, 픽셀 단위로 정교한 segmentation을 위해 CRF를 통한 post-processing 사용. 

Atrous conv를 구현하는 방식은 2가지가 있는데, 2번 방식을 활용함.

1. `Implicitly upsample the filters by inserting holes (zeros), or equivalently sparsely sample the input feature maps [15].`

2. `Subsample the input feature map by a factor equal to the atrous convolution rate r, deinterlacing it to produce r^2 reduced resolution maps, one for each of the r×r possible shifts.`


Atrous Conv의 유용성

![image](https://user-images.githubusercontent.com/44194558/158937149-955be3ef-0cb2-4101-bc89-4e9f8ccbcb97.png)

<br/>

### 3.2 Multiscale Image Representations using Atrous Spatial Pyramid Pooling

<br/>

`To produce the final result, we bilinearly interpolatethe feature maps from the parallel DCNN branches to theoriginal image resolution and fuse them, by taking at each position the maximum response across the different scales.`


![image](https://user-images.githubusercontent.com/44194558/158936929-af5f08af-b0b4-4b7a-ad55-c171df388c37.png)

<br/>

![image](https://user-images.githubusercontent.com/44194558/158937199-03108e22-5e89-4808-a49c-f3c40b0294ba.png)


`For multi-scale input, We separately feed to the DCNN images at scale = {0.5, 0.75, 1}, fusing their score maps by taking the maximum response across scales for each position separately [17].`

<br/>

## 3.3 Structured Prediction with Fully-Connected Conditional Random Fields for Accurate Boundary Recovery

<br/>

분류 성능과 localization 정확도는 trade-off 관계에 있음. 연속적인 max-pooling을 통한 심층 모델은 분류 태스크에 있어서는 효과적이지만, segmentation에 있어서는 굉장히 smooth한 결과를 출력하게 된다.

아래 그림에서 볼 수 있듯이 DCNN의 출력인 score map은 객체의 presence와 해당 객체의 대략적인 위치는 예측할 수 있으나 경계선을 명확히 구분하지 못하고 있다 (smooth).

![image](https://user-images.githubusercontent.com/44194558/158937644-e1cb3395-63d0-41d4-9900-648ce3c0c895.png)

<br/>

본 연구는 DCNN의 recognition capacity와 Fully Connected CRF의 fine-grained localization accuracy를 결합하여 localization 성능을 향상시키고자 한다.

Enercy function으로 표현되는 Fully Connected CRF의 수식을 살펴보면

![image](https://user-images.githubusercontent.com/44194558/158938580-56d28c43-863a-4191-948c-d866f3eb8d78.png)

<br/>

![image](https://user-images.githubusercontent.com/44194558/158937985-2992c848-e63a-4d30-9579-0b6583a94b47.png)

 - Unary term은 CNN 연산을 통해 계산되며, 특정 위치의 픽셀에 대한 label assignment 확률을 표현함.
 
 - Pairwise term은 모든 픽셀들에 대해 (connecting all pairs of image pixers i, j) 픽셀값의 유사도, 위치상의 유사도를 함께 고려함. 

   - Small distance, small intensity(RGB 색상값) -> small negative value -> penalty increase
   
   - Large distance, different intensity -> large negative value -> penalty decrease 

**비슷한 위치, 비슷한 색상값을 갖는 픽셀들에 대해 같은 label이 붙을 수 있도록하고, 원래 픽셀의 거리에 따라 smooth의 수준을 결정.**

다음은 short-range CRF, fully connected CRF간 비교 (본 논문에서는 시간이 많이 걸리는 MCMC를 사용하지는 않음).

![image](https://user-images.githubusercontent.com/44194558/158939036-f56ac561-9368-4937-9ab8-4aa0949e9919.png)

<br/>

**DeepLab 동작 방식**

![image](https://user-images.githubusercontent.com/44194558/158939162-a2df745a-7768-43d2-9653-d00f44cca7b6.png)

1. DCNN을 통해 1/8 크기의 coarse score map을 계산

2. Bilinear interpolation을 통해 원본 이미지만큼의 해상도를 복원
  
   - 각 픽셀 위치에 대한 label assignment 확률 (Unary term)

3. Pairwise term까지 고려하는 CRF를 통한 post processing

<br/>

## 4. Conclusion

`Our proposed “DeepLab” system re-purposes networks trained on image classification to the task of semantic segmentation by applyting Atrous conv, ASPP, Fully Connected CRF.`