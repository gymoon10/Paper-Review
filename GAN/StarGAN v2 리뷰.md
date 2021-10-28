# StarGAN v2 : Diverse Image Synthesis for Multiple Domains


<br/>

## Abstract

<br/>

좋은 image-to-image translation은 다음과 같은 조건들을 만족시키면서 서로 다른 시각 도메인들간의 mapping을 학습해야 한다.

1. 생성된 이미지의 **다양성**
 
2. 여러 도메인으로의 **확장 가능성**

기존의 연구들과 달리 StarGAN v2는 위의 두 가지를 모두 만족시킬 수 있는 모델이다.

<br/>

## 1. Introduction

<br/>

Image-to-image translation에서의 `domain`은 시각적으로 구분되는 카테고리를 의미하고, 각각의 이미지가 가지고 있는 고유한 외형을 `style`이라고 한다. 예를 들어 성별이 도메인이라면 화장, 머리 모양 등이 스타일에 해당된다. 뛰어난 IIT 모델은 도메인과 스타일을 자유롭게 넘나들며 이미지를 생성할 수 있어야 하는데 데이터셋에 많은 수의 스타일과 도메인이 존재할 수 있다는 점을 고려하면 쉽지 않은 문제이다.

스타일 다양성 문제를 해결하기 위한 다양한 방법론들이 제안되어 왔다. 이런 방법들은 생성자에 가우시안 분포에서 랜덤하게 샘플링된 저차원의 latent code를 입력으로 제시하고, domain specific한 디코더가 latent code를 생성 이미지의 다양한 스타일들로 디코딩한다. 하지만 이러한 방법들은 2개 도메인간의 mapping만을 고려하기 때문에 확장성 측면에서 한계가 있다. 예를 들어 K개의 도메인이 존재할 때 생성자는 K(K-1)개의 도메인 mapping을 개별적으로 학습해야 한다.

이러한 확장성의 문제를 해결하기 위해 단일 생성자가 가능한 모든 도메인들간의 mapping을 학습하는 StarGAN이 제안되었다. 생성자는 도메인 라벨을 추가적인 입력으로 받고, 입력 이미지를 해당 도메인으로 전환하는 방법을 학습한다. 하지만 **StarGAN은 각 도메인간의 deterministic mapping을 학습하기 때문에 데이터 분포의 multi-modal 특성을 고려하지 못한다는 한계**가 있다. StarGAN의 이와 같은 한계점은 개별 도메인이 predetermined label로 표현된다는 점에서 기인하는데, 생성자가 one-hot-vector같은 fixed label을 입력으로 받기 때문에 소스 이미지가 주어졌을 때 개별 도메인에 대해 같은 생성 이미지를 출력하게 된다. (각각의 도메인에 대해 동일한 변형만 가능)

다양성과 확장 가능성 두 마리 토끼를 모두 잡기 위해 본 연구는 StarGAN v2를 제안한다. 개선을 위해 StarGAN 에서의 도메인 라벨을 **특정 도메인의 다양한 스타일을 표현할 수 있는 domain specific style code**로 변경한다. 이를 위해 본 연구는 `mapping network` & `style encoder` 2개의 모듈을 제안한다.

1. `mapping network` : 랜덤 가우시안 노이즈를 스타일 코드로 전환하는 방법을 학습

2. `style encoder` : 주어진 참조 이미지로부터 스타일 코드를 추출하는 방법을 학습 

다중 도메인의 케이스에서 위의 두 모듈은 모두 다수의 특정 도메인의 **style code**를 출력하게 되고, 이러한 스타일 코드를 활용하는 것이 다양한 도메인의 다양한 스타일을 가진 이미지를 합성하는데 있어 성공적이었다. (다중 도메인이기 때문에 두 모듈 모두 출력 지점이 여러 개 있고, 각각 특정 도메인의 스타일 코드를 제공함)

<br/>

## 2 StarGAN v2

<br/>

### 2.1 Proposed framework

<br/>

본 연구의 목표는 이미지 x에 대응되는 도메인 y의 다양한 이미지들을 생성하는 단일 생성자를 학습시키는 것. 이를 위해 학습된 도메인 별 style 공간에서 **domain specific style vectors**를 생성하고, 해당 style을 생성자가 반영할 수 있도록한다.

![image](https://user-images.githubusercontent.com/44194558/139212388-9ad98a74-c996-422f-b58a-9c670ba6d78d.png)

<br/>

**Generator**

생성자는 입력 이미지 x와 style code s를 입력으로 받아 style code를 반영하는 출력 이미지 G(x, s)로 전환한다. 여기서 s는 mapping network F 또는 style encdoer E를 통해 제공되며, s를 생성자의 입력으로 제시하기 위해 AdaIn의 방법을 활용했다. style code s는 특정 도메인 y의 스타일을 표현하고 있기 때문에 생성자에 직접적으로 도메인 라벨 y를 제공할 필요는 없어졌다.

<br/>

**Mapping network**

Mapping network F는 **latent code z와 도메인 y를 입력으로 받아 생성자에 공급할 style code s=Fy(z)를 생성**한다. F는 모든 가능한 도메인에 대해 style code를 생성할 수 있도록 MLP로 구성되었다 (multiple output brance는 모든 도메인의 style code를 제공). F는 latent vector, 도메인에 대한 랜덤 샘플링을 통해 다양한 style code를 생성할 수 있다. 본 논문의 multi task architecture는 F가 모든 도메인의 style representation을 효과적으로 학습할 수 있도록 한다.

<br/>

**Style encoder**

Style encoder E는 **이미지 x와 원본 도메인 y를 입력으로 받아 생성자에 공급할 style code s=Ey(z)를 생성**한다. F와 마찬가지로 multi task 학습용이며, 서로 다른 참조 이미지를 사용하여 다양한 style code를 생성한다. 생성자는 style s를 참조 이미지 x에 반영하여 결과 이미지를 합성할 수 있다 (참조 이미지의 스타일을 반영하는 출력 이미지를 합성).

<br/>

**Discriminator**

판별자도 multi task용으로 여러 개의 출력 지점으로 이루어져 있다. 각 도메인 y에 대해 입력 이미지가 진짜인지 가짜인지 판별하는 이진 분류를 수행한다.

<br/>

### 2.2 Training objectives

<br/>

**Adversarial objective**

학습 중에 latent code z와 타깃 도메인 y_wave를 임의로 샘플링한 후, mapping network를 통해 타깃 style code s_wave를 생성한다. 생성자는 이미지 x와 s_wave를 입력으로 받아 다음의 손실 함수를 통해 G(x, s_wave)를 출력하는 방법을 학습한다.

![image](https://user-images.githubusercontent.com/44194558/139226713-3da45e3a-9e3d-4618-a982-52c2b0b29aa6.png)

Mapping network F는 타깃 도메인 y_wave에 있을 법한 style code s_wave를 만들 수 있도록, 생성자는 생성된 s_wave를 활용하여 도메인 y_wave의 진짜 이미지와 구별이 불가능한 생성 이미지 G(x, s_wave)를 생성할 수 있도록 학습된다.

<br/>

**Stytle reconstruction**

생성자가 이미지 G(x, s_wave)를 생성할 때 생성도니 style code s_wave를 이용하도록 강제하기 위해 style reconstruction loss를 활용한다.

![image](https://user-images.githubusercontent.com/44194558/139238784-89936381-c028-407a-b471-223f981fdc04.png)

단일 인코더 E가 다중 도메인의 다양한 이미지 결과물을 생성할 수 있도록 학습한다. 테스트 단계에서 학습된 E는 생성자가 입력 이미지를 참조 이미지의 스타일을 반영하여 변환할 수 있도록 한다.

<br/>

**Style diversification**

생성자가 다양한 이미지들을 생성할 수 있도록 diversity sensitive loss를 활용해 정규화한다.

![image](https://user-images.githubusercontent.com/44194558/139239556-6f3ec88c-76df-47ce-8128-aae0483a1cbf.png)

타깃 style code인 s_wave1, s_wave2는 mapping network F에서 2개의 random latent codes z1, z2를 입력으로 받아 생성된다.

![image](https://user-images.githubusercontent.com/44194558/139239822-3f53d90d-122a-4a06-8eea-614d644e11a6.png)

정규화 식을 최대화 함으로써 생성자가 이미지 공간을 더 탐색하고, 의미 있는 style의 특징을 발견하여 다양한 이미지를 생성하게 한다. 이 목적함수는 최적화 지점이 없기 때문에 선형적으로 가중치를 0으로 줄여가며(decay) 학습을 진행했다.

<br/>

**Preserving source characteristics**

생성된 이미지 G(x, s_wave)가 입력 이미지의 domain-invariant 특징을 유지할 수 있도록 하기 위해 cycle consistency loss를 활용한다. (생성된 style code를 사용하여 생성된 이미지가 입력 이미지에서 포즈나 배경 같이 도메인과 무관한 특징을 적절하게 유지할 수 있도록 함)


![image](https://user-images.githubusercontent.com/44194558/139241632-14a44c7a-9763-4010-b905-b546a59dfed5.png)

  * x : 입력 이미지 
  * y : 입력 이미지의 원래 도메인
  * y_wave : 타깃 도메인 (입력 이미지 x를 타깃 도메인의 이미지로 바꾸려고 함)
  * s_wave : 타깃 도메인에 대한 스타일 벡터 (F를 통해 잠재 벡터 z를 변형)
  * s_hat : 입력 이미지 x에서 도메인 y에 대해 추출한 스타일 벡터

s_hat=Ey(x)은 입력 이미지 x에 대해 추정된 style code이고, y는 입력 이미지의 원래 도메인이다. 생성자가 추정된 style code s_hat을 이용하여 입력 이미지 x를 잘 복원하도록 함으로써, 생성자는 입력 이미지 x의 특징을 유지한 채 스타일만 변경할 수 있도록 학습한다. (타깃 도메인 y_wave로 변환한 이미지를 다시 원래 도메인 y로 되돌려, 가장 처음의 입력 이미지 x와 유사해지도록 강제하는 방식)

<br/>

**Full objective**

![image](https://user-images.githubusercontent.com/44194558/139242114-efe96271-522a-4c7d-957a-b6e386b5e3f3.png)

<br/>

## 3. Experiments

StarGAN에 하나씩 component를 추가하며 실험을 진행

![image](https://user-images.githubusercontent.com/44194558/139243016-715ff547-0682-4558-8f03-7356da3de77d.png)

![image](https://user-images.githubusercontent.com/44194558/139243261-bac0f950-f29d-4036-b3b0-e0db93703ecd.png)

소스 이미지의 포즈, identity (인종 같이 굵직한 특징들)를 참조 이미지의 헤어 스타일, 메이크업, 나이 특징을 조합해 만든 이미지들

![image](https://user-images.githubusercontent.com/44194558/139243396-d9e4e97f-efd3-483d-9905-d1b3aac4c663.png)

![image](https://user-images.githubusercontent.com/44194558/139243805-03bb8349-15d3-42e9-b6ba-da77ae0e4419.png)

<br/>

## 4. Discussion

StarGAN v2의 성공 요인은 다음과 같다.

1. Style code가 multi head mapping network를 통해 각 도메인 별로 독립적으로 생성된다. 이를 통해 생성자는 style code를 활용하는 방식에만 초점을 맞출 수 있고, domain specific 정보는 mapping network가 담당한다.

2. 본 연구의 Style space는 StyleGAN의 영향을 받아 학습된 변환을 통해 생성되기 때문에, 고정된 분포를 사용하는 것에 비해 모델에 유연성을 제공한다.

3. 여러 도메인의 데이터를 완벽하게 활용한다. 각 모듈은 도메인과 무관한 (domain-invariant) 특징들을 학습하여 정규화 되게 하고, 아직 보지 못한 샘플에 대해 보다 나은 일반화를 가능하게 한다.

<br/>


![image](https://user-images.githubusercontent.com/44194558/139244897-f44ff56f-162d-4a0f-9f3f-3dfe82a8abf4.png)

![image](https://user-images.githubusercontent.com/44194558/139245056-55b186e8-ebee-4889-8461-337a1c408a13.png)












