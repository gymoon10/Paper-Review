# StarGAN : Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation

<br/>

## Abstract

<br/>

최근의 연구들은 Image-to-Image translation에 있어 뛰어난 성과를 이루었지만, 기존의 접근법들은 이미지 도메인들의 모든 pair 각각에 대해 서로 다른 독립적인 모델들을 구축해야 하기 때문에 다수의 도메인을 처리하는데 있어 확장성의 한계가 있다. (머리, 성별, 나이 등 다양한 도메인 간 변환을 수행할 때 각각의 변환당 독립적인 생성자를 구축해야 한다)

이러한 한계를 극복하기 위해 본 연구는 **하나의 모델을 사용하여 여러 도메인에 대해 Image-to-Image translation을 가능케 하는 StarGAN**을 제안한다. 

`Such a uniﬁed model architecture of StarGAN allows simultaneous training of multiple datasets with different domains within a single network.`

이와 같은 StarGAN의 특성이 기존 모델들에 비해 이미지의 translation을 유연하게 만들어 보다 우수한 퀄리티의 이미지를 생성하며, 뒷 부분에서 열굴의 표정과 특징 변화 태스크를 통해 StarGAN의 우수성을 보인다.

<br/>

## 1. Introduction

<br/>

Image-to-Image Translation은 웃는 얼굴을 찡그린 얼굴로 바꾸는 것 처럼, 주어진 이미지의 특정한 모습을 다른 모습으로 변환하는 것으로 2개의 다른 도메인에서 학습 데이터가 주어졌을 때, 이미지를 한 도메인에서 다른 도메인으로 전환하는 방법을 학습한다. 몇 가지 용어들의 의미는 다음과 같다.

* `attribute` : 이미지에 존재하는 의미있는 특징들.  ex) 머리 색, 성별, 나이 등

* `attribute value` : attribute의 특정한 값. ex) 머리 색 - 흑발/금발/갈색 머리, 성별 - 남자/여자

* `domain` : 같은 attribute value를 공유하는 이미지들의 집합.  ex) 여성의 이미지들이 하나의 domain을 구성  

다수의 이미지 데이터셋들은 라벨링된 attribute들을 포함하는 경우가 있다. 예를 들어 CelebA는 Facial attributes와 관련된 40개의 라벨들을 포함하고, RaFD는 행복, 분노, 슬픔 등을 포함하는 8개의 라벨들을 가진다. 이러한 세팅들이 다음과 같이 **multi-domain image-to-image translation**을 가능하게 한다.

![image](https://user-images.githubusercontent.com/44194558/139184607-5c228cc4-ddf2-410e-a04e-21b4e2596de3.png)

하지만 기존의 모델들은 아래와 같이 k개의 도메인들 간에 존재하는 K(K-1)개의 모든 mapping을 학습해야 하기 때문에 multi-domain image translation에 있어 비효율적인 측면이 있다. (K(K-1)개의 서로 다른 생성자들을 학습해야 함) 

![image](https://user-images.githubusercontent.com/44194558/139184981-84f47dd5-2c96-441c-b482-daae2d4fa67e.png)

기존의 모델들은 모든 도메인의 이미지들에 존재하는 공통된 특성들이 (e.g 얼굴 모양) 존재함에도 불구하고, 각각의 생성자는 전체 데이터셋을 온전히 활용하지 못한 채 K개의 도메인 중 2개 도메인만을 대상으로 학습하기 때문에 비효율성이 야기된다. 그리고 이런 비효율성이 생성된 이미지의 퀄리티를 낮추게된다.

`They are incapable of jointly training domains from different datasets because each dataset is partially labeled`

이에 대한 해결책으로 제시된 것이 다중 도메인간의 mapping을 학습하는 StarGAN이며, **모든 가능한 도메인들 사이의 mapping을 하나의 생성자를 통해 학습**한다. StarGAN의 아이디어는 고정된 translation을 (e.g black-to-blond) 학습하지 않고, **단일 생성자가 이미지와 도메인 정보를 모두 입력으로 받아 유연하게 이미지를 알맞은 도메인으로 전환**하는 방법을 학습한다. 도메인 정보는 binary나 one-hot-vector의 형식을 이용해 표현한다. 학습하는 동안 랜덤하게 타겟 도메인 라벨을 만들어내고 모델이 유연하게 이미지를 타겟 도메인으로 변환하도록 학습시킨다. 이를 통해 도메인 라벨을 제어할 수 있고, 이미지를 원하는 도메인으로 변환시킬 수 있다.

또한 **mask vector**를 도메인 라벨에 추가하여 서로 다른 데이터셋들의 도메인들 간에 joint training이 가능하도록 하였다. 해당 방법은 모델이 모르는 라벨은 무시하고, 특정 데이터셋의 라벨들에 초점을 맞출 수 있게 한다. 이러한 방식으로 모델은 얼굴 표정 합성 같은 태스크를 효과적으로 수행할 수 있다.

<br/>

## 2. Related Work

<br/>

**GAN**

Adversairal loss를 사용.

<br/>


**Conditional GANs**

![image](https://user-images.githubusercontent.com/44194558/139194388-95cbde5d-493f-44a1-b12c-baedf73bebf6.png)

![image](https://user-images.githubusercontent.com/44194558/139194463-4e6bbef7-efb0-4599-8842-28da2f6be51a.png)

CGAN의 기존 연구들은 생성자와 판별자에 클래스 정보를 입력으로 넣어 특정 클래스에 conditioning된 이미지를 생성할 수 있도록 했고, 주어진 텍스트 설명에 제일 적합한 특정 이미지를 생성하는 최근의 연구들도 있다. 이러한 아이디어는 domain transfer, 이미지의 초해상도, 사진 편집 등의 영역에 성공적으로 적용되었다. 본 연구에서는 conditional한 도메인 정보를 이용하여 이미지를 다양한 타겟 도메인으로 전환할 수 있는 새로운 GAN 프레임워크를 제안한다.

<br/>

**Image-to-Image Translation**

![image](https://user-images.githubusercontent.com/44194558/139194606-c603d76a-a369-4c2a-a72d-a9583b21b02d.png)

 * 학습 과정에서 이미지를 조건으로 입력하는 CGAN의 유형

CGAN을 지도학습 방식으로 활용하는 pix2pix는 adversarial loss에 L1 loss를 결합하기 때문에 paired 데이터를 필요로한다. **데이터 pair**를 얻는 수고를 경감하기 위해 제안된 UNIT은 2개의 생성자가 도메인간의 결합 분포를 학습하기 위해 가중치를 공유하는 방식을 사용한다. CycleGAN은 입력 이미지와 전환된 이미지의 key attribute를 보존하기 위해 cycle consistency loss를 활용한다.


![image](https://user-images.githubusercontent.com/44194558/139194831-b7c04d50-1466-496a-9f0e-dc50cf39fb88.png)

 * paired data for colorization (서로 다른 도메인의 데이터를 한 쌍으로)
 * unpaired 데이터에 대해서는 CycleGAN을 이용해 해결함

<br/>

하지만 위의 프레임워크들은 2개 도메인간의 관계만 학습할 수 있기 때문에 확장성 측면에서 한계가 있다. 본 연구가 제안하는 프레임워크는 **하나의 모델만을 사용하여 다중 도메인들간의 관계들을 학습**할 수 있다.

<br/>

## 3. Star Generative Adversarial Networks

`A framework to address multi-domain image-to-image translation within a single dataset`

![image](https://user-images.githubusercontent.com/44194558/139192960-ee13c684-31d2-4fed-8ba7-973319527070.png)

<br/>

### 3.1 Multi-Domain Image-to-Image Translation

<br/>

목표는 여러 도메인간의 mapping을 학습하는 단일 생성자를 학습시키는 것으로, 이를 위해 입력 이미지 x를 타깃 도메인 라벨 c에 conditioning하여 출력 이미지 y로 전환시킨다. 

![image](https://user-images.githubusercontent.com/44194558/139188119-e87d55b0-c358-4330-bd6a-4b06f7e0e939.png)

 타깃 도메인 라벨 c를 랜덤하게 생성하여 생성자가 입력 이미지를 유연하게 전환하는 방법을 학습할 수 있도록 한다. 또한 보조 분류기를 통해 단일 판별자가 도메인 라벨들과 입력 이미지에 대해 확률 분포를 만들어낼 수 있도록 하여, 다중 도메인을 통제할 수 있게 한다.

 ![image](https://user-images.githubusercontent.com/44194558/139188166-dcff254b-1711-42a7-8bfe-450aee510f09.png)

   * D_src : 판별자에 주어진 소스 이미지들의 확률 분포 
   * D_cls : 도메인 라벨에 대한 확률 분포

<br/>

 **Adversarial Loss**

 ![image](https://user-images.githubusercontent.com/44194558/139188422-d10e612d-9b5b-4881-9860-3e04ce6eccfd.png)

 생성자는 타깃 도메인 라벨 c에 conditioned된 가짜 이미지를 생성하고, 판별자는 입력으로 들어온 이미지에 대한 진짜/가짜 여부를 판별한다. 표준 GAN과 동일하게 생성자는 위의 손실 함수를 최소화 하고자 하고, 판별자는 최대화 하고자 한다.

 <br/>

 **Domain Classification Loss**

 주어진 입력 이미지 x와 타깃 도메인 라벨 c에 대해 x를 타깃 도메인 c로 적절하게 분류가 가능한 출력 이미지 y로 변환하는 것이 목표. 이러한 조건을 만족시키기 위해 판별자에 보조 분류기를 추가하여 도메인에 대한 분류 작업을 수행한다. 즉 손실을 두 가지로 나눌 수 있는데,

 1. 진짜 이미지들에 대한 도메인 분류 손실 (판별자 학습)

![image](https://user-images.githubusercontent.com/44194558/139189438-ff076df2-c61b-4407-88a2-54f7dd2d68d9.png)

판별자는 위 손실 함수를 최소화하여, 진짜 이미지 x를 원래의 도메인 c'로 분류하는 방법을 학습. (D_cls가 1에 가까워지도록)

2. 가짜 이미지들에 대한 도인 분류 손실 (생성자 학습)

![image](https://user-images.githubusercontent.com/44194558/139189473-c659605b-2c30-48e7-9773-584bc2541111.png)

생성자는 위 손실 함수를 최소화하여, 타깃 도메인 c로 분류될 수 있도록 이미지를 생성하는 법을 학습. (D_cls가 1에 가까워지도록)

<br/>

**Reconstruction Loss**

생성자는 진짜 같고, 타깃 도메인으로 적절하게 분류될 수 있는 이미지를 생성하기 위해 adversarial loss, domain classification loss를 최소화한다. 하지만, 위의 손실 함수 값을 최소화 한다고 해서 변환된 이미지가 입력 이미지의 내용을 보존하는 것이 보장되지는 못한다. (**이미지의 content는 유지한 채 style만 변화시키길 원함**) 즉, 입력 이미지의 컨텐츠를 보존하지 않고, 완전히 다른 형상으로 변환된 이미지도 판별자를 충분히 속일 수 있기 때문에 reconstruction loss를 활용하여 생성된 이미지가 원래 도메인 라벨과 함께 생성자에 입력으로 주어지면 원래의 입력 이미지와 같은 형상으로 복원할 수 있도록 한다.

`changing only the domain-related part of the inputs.`

이와 같은 문제를 해결하기 위해 생성자에 cycle consistency loss를 활용한다.

![image](https://user-images.githubusercontent.com/44194558/139191087-7ca436e5-2b97-48b3-9fb6-25fc43696c3b.png)

변환된 이미지 G(x, c)와 원래의 도메인 라벨 c'를 입력으로 하여, 원래 이미지 x를 다시 생성해내도록 (reconstruct) 한다. L1-norm을 사용하고, 단일 생성자는 encoder-decoder의 역할과 유사하게 원본 이미지 x를 타깃 도메인 c로 변환할 때, 변환된 이미지 G(x, c)를 원래 이미지 x로 reconstruct할 때 총 2 번 사용된다.

<br/>

**Full Objective**

![image](https://user-images.githubusercontent.com/44194558/139191561-ea13be40-59b2-4144-9b01-d271cf5dd7f7.png)

lambda는 하이퍼 파라미터. (lambda_cls=1, lambda_rec=10 사용)

<br/>

### 3.2 Training with Multiple Datasets

<br/>

StarGAN의 중요한 장점은 다른 라벨들을 가지고 있는 서로 다른 데이터셋들을 동시에 처리하여, 테스트 단계에서는 모든 라벨들을 제어할 수 있다는 점이다. 서로 다른 데이터셋들 로부터 학습을 진행할 때 문제가 되는 점은 각 데이터셋의 라벨 정보가 부분적으로만 알려져 있다는 점이다 (A 데이터셋은 B 데이터 셋의 라벨 정보를 공유하지 못함).

**Mask Vector**

위와 같은 문제를 해결하기 위해 mask vector m을 사용하여, 특정화되지 않은 라벨들은 무시하고 특정 데이터 셋에 존재하는 라벨에 초점을 맞출 수 있도록 한다. 예를 들어 CelebA 데이터셋의 경우 행복 같은 라벨은 무시하고, 가지고 있는 머리색 라벨에 초점을 맞추도록 할 수 있다.

![image](https://user-images.githubusercontent.com/44194558/139192223-392012e4-dc61-485e-b541-b3278a1cc37b.png)

![image](https://user-images.githubusercontent.com/44194558/139195273-4415e9de-ee1d-41ef-b294-4ed27cb97a79.png)

 * 리스트가 아니라 concatenation
 * c_i : i 번째 데이터셋의 라벨에 대한 벡터
 * 본 연구는 CelebA, RaFD를 사용했으므로 n=2

**Training Strategy**

![image](https://user-images.githubusercontent.com/44194558/139195351-625daee7-881d-4027-b2b2-a3bcd3d5f4f6.png)

위의 도메인 라벨을 생성자의 입력으로 사용. mask vector를 통해 알려지지 않은 라벨은 무시하고, 확실히 알려진 라벨에 초점을 맞추어 학습하게 된다. 판별자의 보조 분류기는 모든 라벨들에 대해 확률 분포를 생성한다. 그 후 판별자는 classification error만 최소화하게 됨. 예를 들어 CelebA에 대해서는 판별자가 CelebA의 attribute들에 대해서만 classification error를 최소화하고, RaFD에 대해서는 RaFD의 attribute들에 대해서만 classification error를 최소화한다.

이러한 세팅을 통해 판별자는 다양한 데이터셋을 번갈아가며 학습함으로써 모든 데이터셋에 관한 discriminative feature들을 학습하게 되고, 생성자도 모든 라벨들을 제어할 수 있게 된다.

<br/>

## 4. Imprementation

<br/>

**Improved GAN Training**

학습 과정을 안정화시키고, 보다 좋은 퀄리티의 이미지를 생성하기 위해 Eq(1)을 gradient penalty와 Wasserstein GAN의 목적 함수로 대체.

![image](https://user-images.githubusercontent.com/44194558/139193433-db1e93e7-ea05-4273-89ec-7a8ae4b6db41.png)

**Network Architecture**

생성자 네트워크는 2개의 Convolutional layer로 구성됨 - stride=2 for down-sampling, 6 residual blocks, 2 transposed CNN with stride=2 for up-sampling

생성자에는 instance normalization을 사용하고, 판별자에는 사용하지 않음.

<br/>

## 5. Experiments

![image](https://user-images.githubusercontent.com/44194558/139193919-e4971ea4-16b6-449f-9f73-0c20d45041bd.png)

<br/>

![image](https://user-images.githubusercontent.com/44194558/139193979-d783d062-75ca-46f4-a4f4-4b0f35d5cb68.png)

<br/>

![image](https://user-images.githubusercontent.com/44194558/139194044-77f2679b-a1be-4b3a-a5fd-599a851aaaf6.png)



<br/>

참고 : https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/lecture_notes/StarGAN.pdf























