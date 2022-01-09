# Denoiseg: Joint Denoising and Segmentation

<br/>

## Abstract

의료 영상 분석에서 Segmentation 작업이 필요할 때가 있는데, segmentation 학습용 데이터셋의 구축은 많은 수의 annotation GT를 필요로 하기 때문에 현실적으로 굉장히 어렵고 비용이 많이 드는 일이다. 

본 연구는 NOISE2VOID를 확장하여 소수의 annotated GT segmentation 만을 가지고도 end-to-end로 학습될 수 있는 네트워크를 제안한다. NOISE2VOID는 self-supervised denoising scheme으로 노이즈가 있는 이미지만을 입력으로 받아(pair x) 학습되는 네트이고, 본 연구는 여기서 더 나아가 3-class segmentation을 수행한다. 

본 연구의 방식은 denoising과 segmentation이 한 네트워크에서 동시에(jointly) 학습될 때 denoising이 segmentation에 도움을 줄 수 있다는 점에 착안하였다. 실제로 관찰한 바에 의하면 segmentation 성능은 raw data의 품질이 좋을 수록 (low noise) 향상 되었다.

`The network becomes a denoising expert by seeing all available raw data, while co-learning to segment, even if only a few segmentation labels are available.` 

Denoiseg는 노이즈가 적은 고품질의 학습용 데이터를 제공하고, dense segmentation 태스크에 대한 효율적인 few shot learning (학습 데이터가 제한적인 상황에서의 AI 학습 알고리즘) 을 가능하게 한다.

<br/>

## 1. Introduction

모든 딥러닝 기반의 segmentation 기법들은 방대한 양의 labeled GT를 요구하게 된다. 

Denoising 분야에서는 self-supervised learning으로 위와 같은 annotation 문제를 효과적으로 해결하였다. 과거의 방식은 noise-cleaned image pair가 필요했지만 self supervised 방식은 noise가 있는 raw data만을 활용하여 학습될 수 있다.

다른 연구에 의하면 segmentation에 앞서 self supervised denoising을 수행하는 것이 성능을 크게 향상시킬 수 있다 (특히 가능한 segmentation GT가 적을 때). 이는 self supervised denoising 모듈이 전체 데이터에 대해 학습이 가능하기 때문이다. 그렇기 때문에 추후에 이루어지는 segmentation 모듈은 해석하기 쉬운, 고품질의 이미지를 입력받을 수 있고, 많은 GT가 없어도 높은 성능을 이끌어낼 수 있다.

본 연구가 제안하는 Denoiseg는 NOISE2VOID를 활용하지만, 기존 연구들과 달리 denoising과 segmentation을 sequential하게 처리하지 않고 단일 네트워크에서 동시에(jointly) 학습하고 예측을 수행한다. 모델 아키텍쳐로는 U-NET을 사용한다.

<br/>

## 2. Method

`We propose to jointly train a single U-Net for segmentation and denoising tasks. While for segmentation only a small amount of annotated GT labels are available, the self-supervised denoising module does beneﬁt from all available raw images.`

![image](https://user-images.githubusercontent.com/44194558/148636052-3c2af70a-0dde-4fad-919d-66e7d412ed2e.png)

 - Denoising은 모든 이미지 데이터를 학습에 사용, segmentation은 GT가 존재하는 소수(subset)의 이미지셋 사용

위의 네트워크를 jointly train하기 위해 다음의 combined loss를 사용.

![image](https://user-images.githubusercontent.com/44194558/148636146-ff7fdc8f-2ddb-486c-b949-ecacf60e6564.png)

 - m : batch size (이 중 GT sementation이 없는 것도 존재)

 - 위의 식에서 GT segmentation을 사용할 수 없는 경우 
![image](https://user-images.githubusercontent.com/44194558/148636197-5e926f83-61a1-4f5b-964d-69ffe8cdb15f.png)

<br/>

## 3. Experiments 

![image](https://user-images.githubusercontent.com/44194558/148636268-7ba10476-c545-46bc-bcd7-ba73579f6579.png)




