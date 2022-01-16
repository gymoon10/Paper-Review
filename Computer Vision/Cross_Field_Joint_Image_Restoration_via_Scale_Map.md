# Cross-Field Joint Image Restoration via Scale Map

<br/>

## Abstract

가시광선 색상, 적외선, 플래쉬 등 다양한 조건에서 갭쳐된 이미지들은 noise와 기타 artifact들을 제거하는데 효과적으로 사용될 수 있다. 따라서 본 연구는 서로 다른 filed에서 촬영된 이미지들을 활용하는 2 image restoration 프레임워크를 제안한다.  ex) noisy color 이미지, dark flashed 근적외선 이미지

해당 프레임워크에서 중요한 점은 structure divergence를 처리하고, 두 이미지에서 공통적으로 유용하게 활용될 수 있는 에지와 smooth transition을 찾는 것이다. 

`We introduce a scale map as a competent representation to explicitly model derivative-level conﬁdence and propose new functions and a numerical solver to effectively infer it following new structural observations.`

<br/>

## 1. Introduction

저조도 환경에서 촬영된 이미지는 노이즈가 많아 품질이 낮고, 플래쉬를 사용하는 경우는 원치 않는 명암이나, 톤의 변화를 야기한다. 따라서 최근에는 다양한 imaging device를 통해 서로 다른 환경에서 촬영된 이미지들을 활용하는 컴퓨터 비전 solution들이 제안되어 왔다.

예를 들어, NIR 이미지는 플래쉬에 의한 노이즈가 적기 때문에 노이즈가 많은 색상 이미지를 복원하는데 guide로써 활용될 수 있다. 실제로 다양한 분야에서 image fusion 방식들이 활용되고 있고, 본 연구는 2-image high quality restoration 문제를 효율적으로 해결하고자 한다.

아래와 같이 RGB, NIR 이미지는 detail distribution이나 intensity formation에 있어 큰 차이를 보이며, 픽셀들간에 존재하는 structure inconsistency는 4가지로 분류될 수 있다.

![image](https://user-images.githubusercontent.com/44194558/149651410-4b56bebb-c0df-4dd6-ad28-3f60f95054a6.png)

1. Gradient Magnitude Variation : 위 그림 (c)의 1행은 문자 D가 적외선과 가시광선의 빛 반사 차이로 인해 서로 다른 명암비를 가지는 것을 보인다. 

2. Gradient Direction Divergence : 2 행에서 edge 그래디언트가 서로 다른 방향을 가짐으로써 structural deviation을 야기한다.

3. Gradient Loss : 마지막 행에서 문자들이 NIR 이미지에서 사라졌다.

4. Shadow and Highlight by Flash : NIR 이미지에 플래쉬를 사용하면, 다른 이미지에는 나타나지 않은 그림자와 빛이 생긴다.


위와 같은 이슈들은 서로 다른 유형의 이미지들간에 존재하는 inherent discrepancy of structures로 인해 발생하며, 이를 cross-field problem으로 정의한다. 이러한 이슈들을 다루는 알고리즘들을 cross-field image restoration이라고 부른다. 이때 Simple joint image filtering은 약한 에지를 blur 처리할 수 있고, 직접적으로 guidance gradient를 노이즈 field에 transfer하는 것은 unnatural appearance를 야기하는 문제가 있다.

따라서 본 연구는 Scale map을 활용한 새로운 프레임워크를 제안한다. Scale map은 이미지들간의 structure discrepancy를 포착하고, 통계와 수치적으로 명확한 의미를 가진다. 그러므로 본 연구는 최적의 scale map (adaptive smoothing, edge preservation, guidance strength manipulation)을 구성하는 함수를 설계한다. 앞서 언급한 4가지의 이슈들을 해결할 수 있고, robust function approximation, problem decomposition을 통해 수백 회 이상의 iteration이 필요한 gradient descent방식을 대체할 수 있는 효율적인 (5회 이내의 pass면 충분히 수렴) solver를 제안한다.

<br/>

## 2. Modeling and Formulation

입력으로 noisy RGB 이미지 I0와 guidance 이미지 G를 (다른 imaging device로 촬영) 입력으로 받는다. G와 I0의 채널수는 같지 않아도 되며, 픽셀 값은 [0, 1]로 정규화된다. 우리의 목적은 I0로부터 structure를 보존하면서 noise를 제거한 이미지를 복원하는 것이고, 각각의 색상 채널을 개별적으로 처리한다.

`We introduce an auxiliary map s with the same size as G, which is key to our method, to adapt structure of G to that of I∗ – the ground truth noise-free image.`

s map은 다음과 같은 condition에서 정의되며, 아래의 그림은 Fig 1의 cross field example에 대한 최적의 s map.


![image](https://user-images.githubusercontent.com/44194558/149651987-a703b4dc-3701-4621-b6e4-b3d0403fbe08.png)

 - 그래디언트는 x, y축 각각 계산됨
 - si는 두 이미지간에 서로 대응되는 그래디언트의 차이를 측정
    - `s is a ratio map between the guidance and latent images.`


![image](https://user-images.githubusercontent.com/44194558/149652096-356efd97-0095-48a6-97a4-a4a44be12abc.png)

Scale map s의 속성은  delta G와 delta I*간의 structure discrepancy에 의해 설명된다 (아래 그림 참고).

![image](https://user-images.githubusercontent.com/44194558/149652182-6313789f-2a3d-4e3e-aead-7caf74ef96e4.png)

 - (b) : 에지 근방에서 그래디언트의 방향이 반대 
 - (c) : s value는 NIR 이미지에서 에지가 존재하는 부분들에서 변동이 존재

s에서 음수값은 에지가 두 이미지에서 모두 존재한다는 것을 의미한다 (그래디언트의 방향은 서로 반대). 

Guidance 이미지 G에 플래쉬로 인한 추가적인 그림자나 빛이 존재하면 (delta I*에는 없는) si=0이 되어 그러한 요인들을 무시할 수 있도록 함 (일종의 mask 역할).

Guidance 에지가 존재하지 않는 경우 delta G=0이 되고, 이때 si는 아무 값이나 취할 수 있으나 0으로 설정하는 것이 local smoothness 측면에서 권장된다.

<br/>

![image](https://user-images.githubusercontent.com/44194558/149652503-d9a22b6f-b1d3-4b93-9e6e-322b52e9f73b.png)

 - delta I, s가 unknown이므로 ill-posed
 - data term expression을 통해 variation을 취함

`Our ﬁnal scale map adapts the gradients of G to match I0 with noise removed.`

<br/>

### 2.1 Data Term about s

Final cost : ![image](https://user-images.githubusercontent.com/44194558/149652671-87fc9f7e-690a-4fe4-ac14-731ec4dbd3b5.png)

  - s를 통해 변환한 G의 structure가 I의 structure와 얼마나 유사한지

해당 비용을 안정화하기 위해 normalization 적용. Delata G에 의한 unexpected scaling effect제거 효과.

![image](https://user-images.githubusercontent.com/44194558/149652709-da554e98-61f3-47c2-b9a7-4548f0f86cbe.png)

분모가 0이 되는 상황과, outlier를 제거하기 위해 다음과 같이 data term 정의

![image](https://user-images.githubusercontent.com/44194558/149652732-7480a18c-81bc-43ba-8b78-06e9f5c73f06.png)

 - x축, y축으로 sclae map s가 I의 structure를 충분히 반영해야 함

<br/>

### 2.2 Data Term for I

![image](https://user-images.githubusercontent.com/44194558/149652862-b3ca26ef-67d1-4efc-9fbe-4571e4e315ca.png)

 - Noisy 이미지 I0와 구조적으로 유사하면서(특히 명백한 edge 쪽에서) 노이즈가 제거된 restoration result를 요구함
 - Robust function p가 I0의 noise를 일부 제거하는 효과가 있음

<br/>

### 2.3 Regularization Term

`Our regularization term is deﬁned with anisotropic gradient tensors [13, 4]. It is based on the fact that s values are similar locally only in certain directions. Our anisotropic tensor scheme preserves sharp edges according to gradient directions of G.`

Anisotropic(비등방성) 텐서는 다음과 같이 표현. 해당 텐서에 대한 고윳값 분해를 통해 delta s에 대한 regularization을 표현.

![image](https://user-images.githubusercontent.com/44194558/149653097-ed00619f-ae41-4a8d-aeca-16897bc31c72.png)

![image](https://user-images.githubusercontent.com/44194558/149653169-ea6614f1-070b-4a78-83ee-fa859511389c.png)

 - 고윳값에 의해 각 방향마다 smoothing penalty가 다르게 적용됨

<br/>

### 2.4 Final Objective Function

s map과 복원된 이미지 I를 추정하는 최종 목적 함수는 다음과 같음

![image](https://user-images.githubusercontent.com/44194558/149653206-b472a85b-f5d8-45b6-9589-d6294eb0f475.png)

 - Non-convex
 - Naive gradient descent로는 최적화가 힘들기 때문에 iterative method 제안

<br/>

### 3. Numerical Solution

`To solve the non-convex function E(s, I) deﬁned in Eq.(14), we employ the iterative reweighted least squares (IRLS), which make it possible to convert the original problem to a few corresponding linear systems without losing generality.`

- Robust function approximation

![image](https://user-images.githubusercontent.com/44194558/149653940-0624e220-416f-4233-be29-445ee6624e76.png)

- Express final objective function as vector form

![image](https://user-images.githubusercontent.com/44194558/149653953-727d61a9-8866-46f9-bcb4-06adbbf114a0.png)

<br/>

![image](https://user-images.githubusercontent.com/44194558/149653999-8c342314-e60e-4aea-b02f-17728d1656b5.png)

<br/>

## 4. Experiments


![image](https://user-images.githubusercontent.com/44194558/149654119-8520b50c-b31f-4e07-9c73-33fe411328a2.png)

 - `Reversed gradients for the letter “D” are corrected with the negative values in the resulting scale map s.` 


![image](https://user-images.githubusercontent.com/44194558/149654162-120addfb-711c-4f38-b010-02545626e9ed.png)

 - `Our estimated s map shown in (c) contains large values along object boundaries, and has close-to-zero values for highlight and shadow.`


![image](https://user-images.githubusercontent.com/44194558/149654192-cb9e5667-f377-423e-8043-30a9dde16312.png)

 - ` Fig. 7 gives comparisons with BM3D [5] and the method of [21], which do not handle gradient variation.`


![image](https://user-images.githubusercontent.com/44194558/149654260-84043f9d-f59d-4ba4-8aec-35418ce99bce.png)

 - `our recovered structures are sharp. Gradient reversion in input images also happens in this conﬁguration due to strong ﬂash. Without handling it, it is hard to preserve these sharp edges as gradients averaging to nearly zeros are commonly resulted in.`

<br/>

## 5. Conclusion

서로 다른 도메인의 이미지들간의 structural discrepancy를 고려하여 noisy 이미지를 복원하는 새로운 기법.

 - Scale map s가 structural discrepancy에 대한 정보를 encoding된 형태로 보유

목적 함수와 최적화 방식은 다른 도메인의 이미지를 guidance로 효과적으로 사용하며 중요한 디테일과 에지를 잘 보존한다.






