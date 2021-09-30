# Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

## Abstact

본 연구에서는 **CNN을 활용한 새로운 적대적 생성 네트워크 구조, Deep Convolutional GANs(DCGAN)을 제안**하고, 다양한 이미지 데이터셋을 통해 **DCGAN이 세부적인 물체부터 전체 이미지에 이르는 위계적인(hierarchial) 특징들을 학습한다**는 점을 밝힘. 또한, 학습된 특성들을 바탕으로 새로운 태스크를 수행하여 GAN 이 학습한 특성이 일반적으로 활용될 수 있음을 보인다.

## 1. Introduction

적대적 생성 네트워크는 일반적으로 학습이 불안정하다고 알려져 있으며, 이 프레임워크에서 어떤 피쳐들이 학습되는지를 이해하고 시각화하려는 시도 역시 많지 않았다. 이러한 상황에서 본 논문의 주된 기여는 다음과 같음.

- **안정적인 학습을 가능하게 하는 Convolutional GAN 의 구조와 기타 제약조건을 제시**
- 학습된 판별기를 이미지 분류 태스크에 적용하여 다른 비지도 학습 모델과 비견되는 성능을 달성
- **학습된 필터를 시각화하고 특정 필터가 특정 물체를 생성하도록 학습된다는 점을 보임**
- **잠재벡터의 산술 연산을 통해서 이미지의 의미/내용을 쉽게 조작할 수 있음을 보임**

## 2. Related work

### 2.1. Representation learning from unlabled data

대표적인 비지도 표현 학습으로는 클러스터링, 오토인코더, Deep belief networks 등이 있다.

### 2.2. Generating natural images

이미지 생성 모델은 크게 모수적/비모수적 모델로 나눌 수 있다. 비모수적 모델은 대부분 데이터베이스에 존재하는 이미지를 찾아 매칭하는 방식으로 작동하며, texture synthesis, super-resolution, in-painting 등에서 활용된다. 모수적 이미지 생성 모델로는 VAE, GAN 등이 대표적이지만, 이들이 충분히 자연스러운 이미지를 생성하지는 못했다. 이후 RNN, deconvolution 등의 레이어를 적용한 모델들이 약간의 성공을 거두었다.

### 2.3. Visualizing the internals of CNNs

Zeiler et al.(2014)은 디컨볼루션과 활성화 함수를 활용하여 각각의 컨볼루션 필터가 수행하는 기능을 대략적으로 파악하는 방법을 제안. Mordvintsev et al. 은 그래디언트를 통해 어떤 이미지가 어떤 필터들을 활성화하는지를 분석하는 방법을 제안. 

## 3. Approach and model architecture

CNN을 활용해서 GAN을 확장하려는 연구는 이전까지 성공적이지 못했다. 하지만 연구진은 여러 모델 구조를 시도해보면서 안정적인 학습, 고해상도 이미지 생성, 깊은 생성 네트워크 사용을 가능하게 하는 구조를 찾아냈다. 

1. 풀링 레이어 대신 **컨볼루션 레이어**를 사용
    - 판별기(Downsampling): Strided convolution
    - 생성기(Upsampling): **Fractional-strided convolution**
2. FC 레이어를 최대한 제거
    - 판별기: 마지막 분류 레이어에만 FC 사용
    - 생성기: 첫 레이어에만 FC 사용
3. **배치 정규화**를 사용
    - 판별기의 입력, 생성기의 출력단에서는 배치 정규화 사용하지 않음
4. ReLU 활성화 사용
    - 판별기: **LeakyReLU** + Sigmod (LeakyReLU 가 특히 고해상도 이미지에서 잘 작동했음)
    - 생성기: ReLU + Tanh

> <details>
> <summary>Fractional-strided convolution</summary>
>
> </details>
> 
> ### Fractional-strided convolution(Transposed convolution)
> 
> Fractional-strided convolution은 흔히 transposed convolution, deconvolution 등의 이름으로 불리는 연산이다(저자에 따르면 엄밀히 말해 fractional-strided convolution과 deconvolution은 서로 다르다). 
>
> 아래 그림은 `(4, 4)` 이미지에 `(3, 3)` 컨볼루션 필터를 적용한 예시로, 결과는 `(2, 2)` 크기의 피쳐 맵이다. 패딩은 적용하지 않았고, 스트라이드는 1이다.
>
> <p align="center"><img src="../assets/DCGAN/DCGAN-09.png" width="75%"></img></p>
>
> Fractional-strided convolution 은 **기존 컨볼루션의 이미지-피쳐 맵의 지역적인 연결 관계를 유지하면서 반대로 저해상도 피쳐 맵을 고해상도 이미지로 보내는 연산**이다. 위의 컨볼루션 연산이 가진 이미지-피쳐 맵의 연결 관계를 유지하는 fractional-strided convolution 연산은 아래와 같다. 
>
> <p align="center"><img src="../assets/DCGAN/DCGAN-10.gif" width="75%"></img></p>
>
> 예시에서는 `(3, 3)` 커널을 사용하여 `(2, 2)` 피쳐 맵을 `(4, 4)` 이미지로 보내고 있으며, 이미지와 피쳐 맵 사이의 연결 관계가 위의 컨볼루션 연산과 동일하게 유지되고 있다. Fractional-strided convolution 을 사용하면서 아래 그림처럼 겹치는 영역이 발생하는 경우 덧셈 연산으로 처리하게 되는데, 이는 생성된 이미지에서 흔히 나타나는 체커보드 아티팩트의 원인이 된다. 


## 4. Details of adversarial training

연구진은 Large-scale Scene Understanding(LSUN) bedrooms, Imagenet-1k, Faces 데이터셋에 대해 DCGAN 모델을 학습시키고 결과를 분석하였다. 

- 전처리: [-1, 1] 사이로 스케일링
- 배치 사이즈: 128
- 판별기 활성화함수: LeakyReLU
- 옵티마이저: Adam

### 4.1. LSUN

<p align="center"><img src="../assets/DCGAN/DCGAN-01.png" width="75%"></img></p>

연구진은 약 300만 장의 이미지가 포함된 LSUN bedrooms 데이터셋으로 모델을 학습시켰다. 위 그림은 생성기의 구조를 나타내는데, 먼저 100차원 잠재벡터를 입력으로 받아 FC 레이어에서 `(4, 4, 1024)` 크기의 피쳐 맵으로 매핑한다. 이후 fractional-strided convolution 레이어로 업샘플링을 거쳐 최종적으로 3채널 `(64, 64, 3)` 크기의 이미지를 생성한다.

적대적 생성모델이 이미지 생성에서 어느 정도의 성공을 거두기는 했으나, 오버피팅이나 데이터 암기 등의 문제가 제기되어왔다. 연구진은 학습 과정에서 모델이 데이터를 암기하는 현상을 방지하기 위해 오토인코더를 활용하여 데이터셋의 중복 제거를 시행하였고, 1 에포크 학습 후에 모델이 생성한 이미지를 점검하였다. 학습 스케줄에서 매우 작은 학습률(learning rate)과 미니배치 경사하강법을 사용하였기 때문에 1 에포크 학습 후의 모델은 데이터를 외운 상태라고 보기는 어렵다. 생성된 결과물은 모델이 데이터를 외우지 않고, 일반적인 특성 표현을 잘 학습하였음을 보여준다. 

<p align="center"><img src="../assets/DCGAN/DCGAN-02.png" width="75%"></img></p>

### 4.2. Faces

온라인에서 1만 명의 사진 약 300만 장을 크롤링한 후 해상도가 충분히 높은 얼굴 사진만을 잘라내 약 35만 장의 얼굴 데이터셋을 만들었다. 

### 4.3. ImageNet-1K

ImageNet-1K 데이터셋을 `(32, 32)` 크기로 잘라 활용하였다.

## 5. Empirical validation of DCGANs capabilities

비지도 학습의 모델의 퀄리티를 측정하기 위해서 가장 많이 사용하는 기법은 비지도 학습 모델을 통해 추출된 특성에 선형 분류기를 결합하여 분류 성능을 평가하는 것이다. 본 연구에서는 ImageNet-1K 데이터셋으로 학습한 DCGAN 모델을 CIFAR-10, StreetView House Numbers 데이터셋에 적용하여 피쳐를 추출하였다. 연구진은 판별기의 각 레이어에서 나오는 피쳐 맵을 `(4, 4)` 크기로 풀링한 후 이어붙이는 방식으로 피쳐를 추출하였다. 이후 추출된 피쳐에 선형 SVM을 적용하여 분류 성능을 측정하였다. 

### 5.1. Classifying CIFAR-10 using GANs as a feature extractor

<p align="center"><img src="../assets/DCGAN/DCGAN-03.png" width="75%"></img></p>

K-means 는 CIFAR-10 데이터셋에 대해 가장 좋은 성능을 내는 비지도 학습 베이스라인 모델이다. DCGAN은 Exemplar CNN 보다는 낮은 성능을 보였지만, K-means 기반의 베이스라인 모델들보다는 좋은 성능을 보였다.

### 5.2. Classifying SVHN using GANs as a feature extractor

<p align="center"><img src="../assets/DCGAN/DCGAN-04.png" width="75%"></img></p>

DCGAN은 SVHN-1000 classes 데이터셋에서 테스트 에러 22.48%로 SOTA 성능을 기록하였다. 분류 성능이 단지 DCGAN의 아키텍쳐로 인해 향상된 것이 아님을 밝히기 위해서 같은 구조의 지도학습 모델로 성능을 테스트하였다. 그 결과 지도학습 모델은 테스트 에러 28.87%로 훨씬 낮은 성능을 보였다.

## 6. Investigating and visualizing the internals of the networks

마지막으로, 연구진은 다양한 방식을 활용하여 생성기와 판별기가 이미지의 특성을 적절히 학습했는지 분석한다.

### 6.1. Walking in the latent space

<p align="center"><img src="../assets/DCGAN/DCGAN-05.png" width="75%"></img></p>

연구진은 가장 먼저 생성기의 잠재공간을 분석하였다. 구체적으로, **잠재벡터의 값을 조금씩 변화시키면서 생성되는 이미지의 변화를 관찰**하였다. 이러한 실험을 통해 모델이 데이터를 암기하고 있는 것은 아닌지, 위계적인 최빈값 붕괴(model collapse)가 일어난 것은 아닌지 등의 문제점을 파악할 수 있다. 만약 **잠재벡터의 변화에 따라 이미지가 너무 급격하게 변한다면, 모델이 학습 데이터를 단순히 암기했거나 위계적인 최빈값 붕괴가 발생했다고 볼 수 있다.** 반대로 잠재벡터의 변화에 따라 이미지가 부드럽게 변한다면 모델이 이미지의 특징들을 적절하게 학습했다고 볼 수 있다. 

### 6.2. Visualizing the discriminator features

<p align="center"><img src="../assets/DCGAN/DCGAN-06.png" width="75%"></img></p>

다음으로 판별기의 마지막 컨볼루션 레이어에서 발생한 그래디언트의 역전파를 시각화하였다. 오른쪽은 학습된 컨볼루션 필터에서 발생한 그래디언트로, 필터가 이미지에 존재하는 지역적인 특성들에 반응하고 있다. 특히 **대부분의 필터들이 LSUN bedrooms 데이터셋의 가장 큰 특징이 되는 객체인 침대에 반응한다**는 점을 알 수 있으며, 이는 판별기가 데이터셋의 특성을 적절하게 학습했다는 근거가 된다. 반면 왼쪽은 랜덤하게 초기화된 컨볼루션 필터로, 이미지의 특징적인 요소들을 포착하지 못하고 무작위적인 위치에서 반응을 일으키고 있다.

### 6.3. Manipulating the generator representation

판별기에 대한 분석을 통해 모델이 원본 이미지의 특징과 이미지에 존재하는 특정한 객체들에 반응하도록 학습되었다는 사실을 알 수 있었다. 그렇다면 생성기는 어떤 특성을 학습하였을까? 이 절에서는 피쳐 맵과 잠재벡터의 조작을 통해서 모델이 이미지의 특성을 적절하게 학습하였는지 분석한다. 

#### 6.3.1. Forgetting to draw certain objects

<p align="center"><img src="../assets/DCGAN/DCGAN-07.png" width="75%"></img></p>

생성된 이미지들의 퀄리티를 볼 때, 생성기는 이미지 내에 존재하는 특정 객체들을 학습한 것으로 보인다. 생성기가 어떤 방식으로 이러한 객체들을 학습했는지 확인하기 위해서, 사진으로부터 창문을 없애는 실험을 진행하였다.

연구진은 최종 이미지가 생성되기 직전의 활성화 맵을 X, 해당 액티베이션 맵이 창문 영역으로 연결되는지 여부를 Y로 하는 로지스틱 회귀모형을 학습시켰다. 이후 새로운 샘플을 생성하는 과정에서는 학습된 회귀모형을 적용하여 마지막 액티베이션 맵이 창문 영역으로 연결되는지 여부를 분류하였다. 이후 창문으로 연결되는 액티베이션 맵을 드롭한 경우와 드롭하지 않은 경우 생성된 이미지를 비교하였다. 윗쪽의 이미지는 모든 액티베이션 맵을 사용하여 생성된 이미지이고, 아래쪽의 이미지가 창문으로 연결되는 액티베이션 맵을 제거한 이미지이다. 아래쪽의 이미지에서 기존의 이미지의 전체적인 구성은 거의 동일하게 보존되어 있다. 하지만 대부분의 창문이 사라지거나 다른 물체로 대체되었음을 알 수 있다. 연구진은 이를 기반으로 모델이 이미지의 특성을 위계적으로 학습하였음을 주장한다(생성기가 만들어내는 특정 액티베이션 맵이 생성된 이미지 내의 구체적인 물체와 연결될 수 있다는 주장으로 보인다).

#### 6.3.2. Vector arithmetic on face samples 

다음으로 잠재벡터의 산술 연산을 통해 잠재벡터와 이미지의 시각적 특징 사이의 관계를 분석하였다. Word2vec 과 같은 모델에서 볼 수 있는 벡터 산술 연산을 생성기의 잠재공간에 적용해본 것이다. 연구진은 먼저 동일한 시각적 특성을 갖는 이미지를 생성하는 잠재벡터 3개씩을 평균낸 후, 이들에 산술 연산을 시행하였다. 이러한 실험을 통해 생성기가 이미지의 시각적 특성을 잠재공간에 잘 반영하고 있음을 알 수 있었다.


<p align="center">
<img src="../assets/DCGAN/DCGAN-08.png" width="75%"></img>
</p>

위의 사진에서는 안경을 쓴 남성, 안경을 쓰지 않은 남성, 안경을 쓰지 않은 여성의 잠재벡터를 활용하여 (glasses + male) - male + female 의 연산을 수행하였다. 이렇게 얻은 잠재벡터를 통해 이미지를 생성한 결과 안경을 쓴 여성의 이미지(glasses + female)를 얻을 수 있었다. 

<p align="center">
<img src="../assets/DCGAN/DCGAN-11.png" width="75%"></img>
</p>

연구진은 또한 생성기가 잠재공간에 "facial pose" 를 반영하고 있음을 밝힌다. 위 사진은 왼쪽을 바라보는 잠재벡터와 오른쪽을 바라보는 잠재벡터를 뽑아 선형보간한 후 대응되는 이미지를 생성한 결과이다. 그 결과 사진 속 인물들의 얼굴 방향이 선형적으로 부드럽게 변하는 것을 확인할 수 있었다.

## 7. Conclusion and future work

연구진은 보다 안정적으로 학습 가능한 적대적 생성 모델 구조를 제시하고, 제안된 모델이 이미지에 특성을 잘 학습한다는 점을 지도학습과 다양한 이미지 생성 태스크를 통해 밝혔다. 하지만 DCGAN 에도 여전히 최빈값 붕괴 등의 불안정성이 존재하므로, 향후 연구에서는 이를 개선/보완할 방법을 탐구해야 할 것이다. 또한 제안된 프레임워크를 통해 오디오, 비디오 등 다양한 도메인에서 실험을 진행할 수 있을 것이고, 잠재공간에 대한 더 깊은 탐구 역시 흥미로울 것이다.