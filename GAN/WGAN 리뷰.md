## 1. Introduction

본 연구는 비지도 학습에 관심을 가지며 '확률 분포를 학습'하는 것의 의미를 구체적으로 살펴본다. 이에 대한 고전적인 정답은  확률 밀도 함수를 학습하는 것이고, 주어진 데이터에 대한 가능도를 최대화 하는 파라미터를 찾는 것이다.  

![image](https://user-images.githubusercontent.com/44194558/136524525-d914cdc2-929b-4fad-8f69-3f6329c501a9.png)

즉 실제 데이터의 분포와 추정량인 parameterized density의 KL divergence를 최소화 하는 것으로, 실제 데이터의 분포와 특정 파라미터에 의해 표현되는 분포 사이의 거리를 최소화하여, 실제 데이터의 분포와 유사한 분포를 생성하는 파라미터를 찾자는 의미.

KL divergence는 분포들의 집합 {P}에서 정의된 하나의 거리와 같고, MLE 문제를 잘 해결하기 위해서는 KL divervgence가 0으로 잘 수렴해주어야 하는데(실제 데이터 분포와 유사한 분포를 만드는 파라미터를 찾는데) 두 분포의 support가 겹치지 않으면 KL diververgence가 발산하기 때문에 계산이 쉽지 않음. 특히 이미지 생성 문제에서 이미지들이 고차원에 분포하기 때문에 이러한 현상들이 더 빈번함.

  > Support : 어떤 함수가 존재할 때 함수값이 0이 되게 하는 정의역의 집합 (x=-1 for f(x)=x+1)

이와 같은 문제에 대한 전형적인 해결책이 모델 분포에 노이즈를 추가하는 방식. 사실상 모든 생성 모델들이 noise component를 포함하는 이유이고, 모든 example들을 커버하기 위해 상대적으로 높은 대역폭을 가지는 gaussian noise가 많이 사용됨.

![image](https://user-images.githubusercontent.com/44194558/136526352-4dac2ef9-1aa5-4294-bfd8-e23632866701.png)

  * 좌 : 두 분포의 support가 겹치지 않아 MLE 불가능  /  우 : noise가 추가되어 이미지의 분포가 넓어짐, MLE 해결 가능 but 성능 저하

-> 생성되는 샘플의 품질을 저하시키고 뿌옇게 만듬

최근 논문들의 경향을 보면 모델에 더해지는 noise의 최적의 표준 편차는 생성된 이미지에 대해 각 픽셀당 0.1인데, 이는 상당히 높은 양의 noise이기 때문에 논문에서 모델이 생성한 샘플을 제시할 때는 likelihood를 보고하는 noise를 추가하지 않음.

-> 즉, noise의 추가는 MLE 방식을 잘 작동하게 만들기 위한 수단으로 문제에 대해서는 부정확함. 


존재하지 않은 실제 데이터의 분포를 추정하는 대신(이미 정답을 알고 있는 상태가 됨) 고정된 분포 P(z)를 가지는 변수 Z를 정의하고, 일종의 신경망인 parametric function g_theta : Z -> X에 통과시킴으로서 어떤 분포 P_theta를 따르는 샘플을 만들게 한다. theta를 다르게 함으로써 분포를 변화시키고, 실제 데이터 분포에 가깝게 만들 수 있다. 이러한 방식은 두 가지 측면에서 유용함.

  1. 확률 밀도와는 다르게 저차원의 manifold에 국한된 분포를 표현할 수 있음. 
  2. 쉽게 샘플을 생성하는 능력이 밀도의 정확한 값을 아는 것보다 유용함

VAE, GAN이 이러한 접근법으로 잘 알려져있음.  VAE는 example들에 대한 approximate likelihoo에 초점을 두며, noise를 조작할 필요가 있으며 GAN은 목적 함수 정의에 있어 융통성을 제공하고,  Jensen Shannon, F-divergence, exotic combination을 포함. **GAN을 학습 시키는 것은 까다롭고 불안정한 것(수렴에 실패)으로 잘 알려져 있음.**

본 논문은 모델의 분포와 실제 분포가 얼마나 가까운지 측정하는 다양한 방법에 관심을 가지며, 거리나 분산을 정의하는 방법들에 주목한다. 본 연구에서 정의하는 distance의 핵심은 **확률 분포 sequence의 수렴**에 미치는 영향이다. 특정 확률 분포 sequence는 ![image](https://user-images.githubusercontent.com/44194558/136537542-b54727f2-4743-4997-ae08-a58d23d2de18.png) 가 0이 되면 수렴하는데, 이 때 **p가 얼마나 정확하게 정의되었는지**가 중요한 이슈이다. **확률 분포의 수렴은 분포 간의 거리를 계산하는 방법에 크게 의존**한다는 점이 필수적인 내용.

![image](https://user-images.githubusercontent.com/44194558/136539061-76e9aed5-d243-4cc7-baa0-2d70a56437d9.png)

 * Pg(x)=0이지만 Pr(x)가 0이 아니면 발산.

![image](https://user-images.githubusercontent.com/44194558/136539198-2c22ca7f-15a8-49cc-9c44-2b8ddefdbcba.png)

  * support가 겹치지 않으면 KL이 양으로 발산 -> 모델이 의미 없는 gradient를 생성해 학습이 잘 진행되지 않음


해당 연구의 주된 contribution은

1. 분포를 학습하는 과정에서 주로 사용되는 probability distance, divergence와의 비교를 통해 **Earth Mover distance**가 어떤 방식으로 작동하는지에 대한 이론적 분석을 제공

2. EM distance에 대한 효율적이고 합리적인 approximation을 최소화하는 WGAN 정의하고 이에 대응되는 최적화 문제가 타당하는 것을 증명

3. GAN의 학습에서 발생하는 주요 문제를 WGAN이 경험적으로 해결한다는 것을 입증. 
   - 생성자와 판별자 사이의 조심스러운 균형(Min-Max)을 유지하는 것이 요구 X
   - 판별자를 최적 수준까지 학습시키면서 EM distance를 연속적으로 추정할 수 있음

<br/>

## 2. Different Distances

**Notation 정리**
  * X : compact metric set (이미지 [0, 1]^d의 공간과 동일한)
  * compact : 경계가 있고, 경계를 포함하는 집합
  * metric : distance
  * Sigma : X의 모든 Borel subset
  * Borel subset : X내에서 측정 가능한 집합들 (확률 분포로 확률 값이 계산될 수 있는 집합, 연속 함수의 기댓값을 계산하기 위한 최소 조건) 
  * Prob(X) : X에 정의된 확률 분포 공간

Divergence에 대한 지표로는 다음과 같은 것들이 있음

![image](https://user-images.githubusercontent.com/44194558/136540808-8c0c95cf-0f45-4598-9e02-235c77b7a3a9.png)

![image](https://user-images.githubusercontent.com/44194558/136540741-7bb996d2-99b5-4024-8ea9-26ae4813549c.png)

  * Pr, Pg는 연속형으로 가정
  * 비대칭, 발산의 위험 (두 분포의 support가 겹치지 않을 때)

![image](https://user-images.githubusercontent.com/44194558/136541707-c467b5bc-d9f9-4874-b78b-eb7897928132.png)

  * JS는 대칭
  * 두 분포의 support가 겹치지 않아도 발산 x (log2로 고정) -> 얼마나 먼지에 대한 정보를 줄 수 없음
  * 
![image](https://user-images.githubusercontent.com/44194558/136541770-64103703-0241-4b32-bd93-0d8c4e38c205.png)

  * r(x, y) : 분포 Pr을 Pg로 변형시키기 위해 얼마나 많은 **질량**을 x에서 y로 운송해야 하는지
  * EM : 최적 운송의 cost   


다음의 예시는 간단한 확률 분포의 sequence가 EM distance를 사용할 땐 명백하게 수렴하고, 나머지 기준 하에서는 수렴하지 않는 것을 잘 보여줌.

![image](https://user-images.githubusercontent.com/44194558/136542666-6171c082-80e0-439c-9f00-ccfa72284412.png)

  * theta가 0에 가까울수록 작은 값을 산출해야 좋은 metric
  * theta가 0이 아니라면 두 분포는 겹치지 않게 됨
     * 기존의 metrifc들이 의미 없는 값을 출력함

EM(좌)과 JS(우)를 비교하면

![image](https://user-images.githubusercontent.com/44194558/136543566-c7e0c351-7f34-46ed-9259-3fbdd66a89f7.png)

  * JS는 theta=0일 때를 제외하면 constant gradient(0) -> 기울기 소실, 학습의 불안정성
  * EM은 두 분포의 support가 겹치지 않아도 학습에 유의미한 gradient 출력


EM distance (Wasserstein distance)가 파라미터 theta에 대해 연속형 loss function인지 증명

![image](https://user-images.githubusercontent.com/44194558/136544063-2e1babfb-ec85-45cd-9449-0d3cd6bb2ac2.png)
  
  * **기술적인 제약(립시츠)**을 만족하며 실제 분포의 기댓값과 생성된 분포의 기댓값 사이 거리를 최소화 해야함
      * 어떤 함수가 립시츠 상수라는 값보다 항상 변화율이 작아야 함
      * 변화율을 ths보다 작게하여 gradient가 과하게 커지는 현상 방지 (유용한 gradient를 얻기 위해)


Topology의 strength 측면에서 KL > JS > TV > EM 순

![image](https://user-images.githubusercontent.com/44194558/136544932-3ad36183-2ced-40c9-8ad9-ddcfa718d2a1.png)

KL, JS, TV는 저차원의 manifold상에서 support가 낮은 분포를 학습할 때는 sensible한(유의미한 gradient를 제공하는) cost fucntion X, EM은 그러한 환경에서 덜 민감 (유의미한 gradient 제공)

참고 : https://haawron.tistory.com/21

<br/>

## 3. Wasserstein GAN

W distance(EM distance)가 기존의 JS보다 GAN 학습에 있어 보다 유용한 loss이지만 수식 계산이 굉장히 힘듬 (intractable).

Wasserstein 손실은 아래와 같이 표현됨

![image](https://user-images.githubusercontent.com/44194558/136545612-2deb9125-b9e6-4a6c-b918-8fdae88d0521.png)

  * r은 샘플링으로 해결해도, joint distribution 공간은 탐색하기 힘들고 최솟값에 대한 보장을 하기 힘듬

위 수식을 어떤 K에 대해 K-립시츠 조건을 만족하는 parameterized family of function을 가지고 있으면 다음과 같이 변경할 수 있음

![image](https://user-images.githubusercontent.com/44194558/136546394-540f7579-02c6-4ad7-9ff4-3a8cce705bdb.png)

  * f_w : 판별자 역할 수행 (critic)
  * critic은 EM거리를 추정, f_w의 파라미터를 바꿔가며 진짜 분포와 생성된 분포의 차이의 최댓값을 측정
    * 이동해야 할 확률 질량의 양을 최대화하는 f_w를 찾음으로써 생성자를 최대로 어렵게 만듬  

생성자의 손실 함수는 아래 식을 minimize하는 식으로 정의됨

![image](https://user-images.githubusercontent.com/44194558/136547011-b4d5ec27-6104-438c-9f04-0488739152b8.png)

  * 실제 분포의 기댓값과 생성된 분포의 기댓값 사이의 거리를 최소화


위와 같은 과정을 통해 Equation 2에 대한 최대화 문제를 해결하는 함수 f를 찾는 문제에 도달함. 이에 대한 근사를 위해 compact space에 놓여있는 f_w의 파라미터 w를 학습시켜야 함. 그리고 위에 주어진 생성자의 손실 함수를 통해 역전파를 수행.

파라미터 W는 gradient 업데이트 이후 가중치들을 특정 범위의 fixed box에 고정시킴으로써  ex) W=[-0.01, 0.01] 어떤 경계 안에 들어올 수 있게 할 수 있고, weight clipping을 통해 립시츠 제약 조건을 강제할 수 있음.

clipping 상수는 일반 머신러닝에서 학습률처럼 동작하는 하이퍼 파라미터이므로 이에 대한 적절한 튜닝이 필요함. (너무 크면 critic을 최적 수준까지 학습시키기 어렵고, 작으면 기울기 소실의 위험) 

WGAN의 전체적인 절차는 아래와 같음

![image](https://user-images.githubusercontent.com/44194558/136549851-1e8ca82e-9573-4d71-934f-b8db35623d49.png)

EM distance는 미분 가능하고 연속적이기 때문에 critic을 최적 수준까지 학습시킬 수 있음. 

 -> 우리가 critic을 더 많이 훈련시킬수록 보다 유용한 Wasserstein gradient를 얻을 수 있음.

이와 달리 JS는 판별자가 더 나은 gradient를 얻을 수록, JS의 locally saturated성질 때문에 trud gradient는 0이 되어 기울기 소실 문제가 발생한다.

<br/>

**Mode collapse 측면에서 WGAN의 장점**

 Mode collapse comes from the fact that the optimal generator for a **ﬁxed** discriminator is a **sum of deltas on the points the discriminator assigns the highest values**, as observed by [4] and highlighted in [11].

![image](https://user-images.githubusercontent.com/44194558/136549964-a3f73829-3a98-4533-9a52-d26feaaaaa78.png)

**판별자의 학습이 앞서 나가게 되어** (생성자와 판별자의 학습 속도의 균형이 깨짐), 판별기의 함수가 step fucntion으로 수렴하게 된 상황 -> gradient vanishing, saturate
 
 판별자는 계속 실제 데이터를 보기 때문에 생성자의 학습이 고정되어도 지속적인 학습이 가능하여, 나중엔 생성자의 가짜 이미지들을 완벽히 구분할 수 있게 됨
 
WGAN은 립시츠 제약 조건으로 인해 critic f_w가 step function이 되지 않아 유의미한 gradient 정보를 지속적으로 줄 수 있음 (non-saturate)
  * 가중치를 제한하기 때문에(clipping) 함수를 선형으로 제한

WGAN 덕분에 판별자의 학습이 앞서나가는 것을 신경쓰지 않아도 되고, 둘의 학습 속도를 조절하기 위해 두 모델의 구조를 다르게 하고 여러 파라미터들을 찾는 수고를 덜 수 있음

<br/>

## 4. Experiments

기존 GAN과 비교했을 때 2가지 장점은

1. 생성자의 수렴, 샘플 품질과 상관관계가 있는 유의미한 loss metric (기존 GAN은 수렴해도 mode collpase 발생)
2. 최적화 프로세스의 향상된 안정성


**EM distance 추정치가 생성된 샘플의 품질과 얼마나 상관관계가 있는 지** 

![image](https://user-images.githubusercontent.com/44194558/136552174-fc76c766-11a2-4d96-98a5-b5f07ddf01d0.png)

  * 1, 2 :학습이 진행됨에 따라 loss가 감소하고, 샘플 품질이 향상됨
  * 3 : 학습 실패, loss가 일정하기 때문에 샘플 품질도 일정함

위와는 반대로 JS distance에 대한 GAN estimation의 변화는 아래와 같음

![image](https://user-images.githubusercontent.com/44194558/136552281-721382d4-bcc6-4e9b-a2ec-8996e521109d.png)

이 양은 샘플 품질과는 상관이 없음

![image](https://user-images.githubusercontent.com/44194558/136552907-dd72c76d-3540-41ea-b7c6-06d11a729fd7.png)

  * 품질이 좋아져도 JSD가 특정 범위의 값으로 고정되어 이미지 품질에 대한 정보를 제공하지 못함
  * 품질이 향상돼도 JSD는 오히려 증가한 케이스도 있음


**향상된 안정성**

WGAN의 장점 중 하나는 critic을 최적 수준까지 학습하도록 한다는 점인데, critic이 완전히 학습된 이후 다른 신경망인 생성가 학습될 수 있는 loss를 제공함.

-> 생성자와 판별자의 capacity를 적절하게 조절할 필요가 없어짐 (critic의 성능이 올라갈수록 생성자가 학습하는데 보다 유용한 gradient를 제공함)

특히 생성자의 구조를 변경시켜도 WGAN이 훨씬 robust함.

![image](https://user-images.githubusercontent.com/44194558/136553737-2c7f4205-be34-403e-968b-770eaa2919df.png)

