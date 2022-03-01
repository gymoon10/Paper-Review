# Noise2Noise: Learning Image Restoration without Clean Data

<br/>

## Abstract

`It is possible to learn to restore images by only looking at corrupted examples.`

 - noise 이미지를 clean signal로 mapping하는 방식을 학습하는 머신러닝 기법을 (GT인 clean 이미지 없이) 오직 noise 이미지만을 사용하여 적용.

<br/>

## 1. Introduction

기존의 denoising 방식들은 대량의 noise image-clean target으로 구성된 pair 데이터와 CNN 네트워크를 활용하여 일종의 regression 모델을 사용한다. 일반적인 image denoising task의 식은 다음과 같음.

![image](https://user-images.githubusercontent.com/44194558/155951949-bb0fe97d-08b4-411a-b441-08c2f690cf16.png)

 - L1/L2 손실 함수를 통해 그 차이를 최소화시키고, 결과적으로 noise가 제거되도록 학습됨  ex) DnCNN
 
 - 학습에 있어 항상 clean image를 필요로 함 (때로는 굉장히 어렵거나 costly함)

<br/>

`In this work, we observe that we can often learn to turn bad images into good images by only looking at bad images. Further, we require neither an explicit statistical likelihood model of the corruption nor an image prior, and instead learn these indirectly from the training data.`

<br/>

## 2. Theoretical Background


L2 loss가 모든 가능한 target의 경우를 평균낸다는 점을 이용한다 (참고: https://stats.stackexchange.com/questions/34613/l1-regression-estimates-median-whereas-l2-regression-estimates-mean). 모든 가능한 경우를 평균낸 최적의 z(denoised output)을 식으로 나타내면 다음과 같음. 

 - `using the L2 loss, the network learns to output the average of all plausible explanations`

![image](https://user-images.githubusercontent.com/44194558/155953126-66fc6b8f-cdf8-4b86-b9b8-48706b826f37.png)

여기서 본 연구는 y가 항상 clean image일 필요가 없다는 아이디어에 착안한다. y가 clean image가 아닌 random한 픽셀값으로 구성된 이미지여도 그 기댓값만 clean target과 일치한다면, 최적의 denoised output z를 찾을 수 있다.

 - `A trivial, and, at ﬁrst sight, useless, property of L2 minimization is that on expectation, the estimate remains unchanged if we replace the targets with random numbers whose expectations match the targets.`

 - `This implies that we can, in principle, corrupt the training targets of a neural newtork with zero-mean noise without changing what the network learns`

L2 loss의 특징을 이용하여 식을 다시 쓰면,

![image](https://user-images.githubusercontent.com/44194558/155954643-15793a9f-afd1-4fdf-83af-fe99adcc4309.png)

 - p(noisy|clean), p(clean)을 필요로 하지 않음


![image](https://user-images.githubusercontent.com/44194558/155955304-bfe3d44a-cfc4-4b8d-9123-70f30ae4e847.png)

 - 좌(기존 방식들), 우(본 연구가 제안하는 방식)
  
 - 오른쪽 : expectation이 clean target인 noisy target으로부터 (unobserved) clean target을 얻는 과정

