
# Abstract

 네트워크의 노드 및 에지에 대한 예측에 있어 학습 알고리즘에 사용되는 특성 공학(feature engineering)에 대한 세심한 주의가 필요하다.
Representation learning에 대한 최근의 연구는 예측에 사용되는 유의미한 특성들을 스스로 학습하는 방식으로 발전해왔다.(기존의 통계 분석, 머신 러닝에서는 유의미한 feature들을 분석가가 도메인 지식 등을 활용하여 정의해야 했지만, 딥러닝은 데이터에 내재된 중요한 특성들을 입력 데이터로부터 스스로 학습함)


 하지만 위와 같은 방식도 네트워크에서 관찰되는 연결 패턴의 다양성을 포착하기엔 부족한 면이 있다. (not expressive enough)
본 연구에서 제안하는 node2vec은 네트워크의 노드에 대한 연속적인 특징 표현을 학습하는 프레임워크로 다음과 같은 특징들이 있음
- 노드를 저차원의 특성 공간으로 매핑함
- 네트워크의 노드가 매핑되는 저차원의 특성 공간은 네트워크 상에서의 특정 노드의 이웃을 보존하는 가능성을 최대화하는 특성을 가짐
- 노드의 연결 관계, 이웃 관계같은 구조적 특성을 충분히 표현할 수 있는 저차원의 공간으로 매핑 (PCA와 유사한 측면 o)
   
 기존의 네트워크 연구들은 네트워크 이웃의 정의에 있어 엄격한(rigid) 기준을 세웠지만, 본 연구는 네트워크 이웃에 대한 유연한(flexible)개념을 정의하여 보다 풍부한 representation learning을 가능하게 한다.
- 본 연구는 biased random walk procedure를 통해 보다 다양한 이웃 탐색을 가능하게 함
    
    
다항 분류 태스크와 링크 예측에 있어 기존 연구들보다 효율적, 복잡한 네트워크에서 보다 효율적인 representation learning을 가능하게 함

<br/>

# 1. Introduction

노드와 에지에 대한 예측은 네트워크 분석에 있어 중요한 태스크이다. 

> ex.1) 노드의 class 예측 작업 
 - 소셜 네트워크에서의 사용자 관심사 예측 & 단백질간 상호 예측 네트워크에서 특정 단백질의 기능적 라벨을 예측

> ex.2) 링크 예측
  - 유전자 간의 새로운 상호 작용 발견 & 소셜 네트워크 상에서 특정 친구의 실제 친구 예측


지도 학습 머신러닝 알고리즘은 유용한 독립 변수들을 필요로하고, 연구자는 노드와 에지에 대한 벡터 표현을 구축해야 한다. 하지만 연구자의 노력과 수고에도 그렇게 구축된 특성들은 일반화가 매우 어려움. (task specific하여 다른 문제 상황에 적용이 힘듬)


위와 같은 방식의 대안으로 최적화 문제에 대한 해결을 통해 feature representation을 학습하는 방식이 있지만, 최적화의 대상이 되는 목적함수를 정의하기 어려울 때가 있고, 계산 효율성과 정확도 사이의 trade off가 존재함.


현재 기술로는 네트워크의 비지도 학습에 대한 목적 함수 정의와 최적화 모두 어려운 측면이 있음. 
1. 기존의 선형, 비선형 차원 축소 기술을 사용해(PCA등) 네트워크의 representative matrix를 (data representation을 최대한 보존하는 식으로)저차원으로 매핑하는 과정은 계산 복잡도가 높고, 성능도 좋지 않음
2. 기존의 차원 축소 방식은 행렬에 대한 고윳값 분해를 포함하기 때문에, 실제 현실을 표현 하는 네트워크에 있어서는 계산이 복잡하고, 그러한 결과로 산출되는 latent representation의 예측 성능도 좋지 않음

이에 대한 대안으로 노드의 local neighborhoods를 최대한 보존하는 목표를 설계함.
- 해당 목표는 SGD를 통해 효율적으로 최적화가 가능
- 기존 연구들은 네트워크 이웃에 대한 엄격한(rigid) 정의를 기반으로 하는 알고리즘이기 때문에, 네트워크의 고유한 연결관계 패턴을 잘 포착하지 못하는 단점이 있음
  

네트워크의 특성인 homophily, structural equivalence를 반영하기 위해서는 노드에 대한 유연한 정의가 필요함.
- Homophily :  nodes in networks could be organized based on communities they belong to
- Structural equivalence : the organization could be based on the structural roles of nodes in the network
   
위의 두가지 규칙을 반영하는 노드 표현 학습을 위해서는 유연한 알고리즘을 허용하는 것이 중요함
- 동일한 네트워크 커뮤니티의 노드를 서로 밀접하게 표현하는 능력 (homophily)
- 유사한 역할을 공유하는 노드들이 비슷한 임베딩 표현을 갖게하는 능력 (structural equivalence)

<br/>

#### Present work
본 연구는 준 지도학습 알고리즘은 node2vec를 제안. 해당 알고리즘은 네트워크로부터 다른 태스크에도 일반화가 가능한(scalable) 표현을 학습하며, SGD를 통해 그래프 기반의 목적 함수를 최대화한다.

node2vec은 d 차원의 특성 공간에서 노드의 네트워크 이웃을 보존할 가능성을 최대화 하는 feature representation을 학습하는 것이 목적이며, 2차 random walk를 통해 샘플 네트워크 이웃을 생성한다.

본 연구의 주요 contribution은 노드의 네트워크 이웃에 대한 유연한 개념 정의에 있다. (flexible notion of a node's network neighborhood)
- 네트워크에서의 노드의 역할(structural equivalence) 및 노드가 속한 커뮤니티를(homophily) 기반으로 노드에 대한 표현을 학습함
- random walk를 통해 노드의 다양한 네트워크 이웃을 효율적으로 탐색함
- 탐색하고자 하는 representation space를 매개 변수를 통해 통제할 수 있음. (네트워크 이웃에 대한 엄격한 정의 하에서는 불가능)
- 직관적이고 해석가능한 parameter


개별 노드들의 feature representation들은 노드의 쌍(pair)인 에지로 확장이 가능하다. 에지에 대한 표현을 학습하기 위해서는
1. 간단한 이항 연산자를 이용하여 개별 노드의 학습된 특징 표현을 구성
2. 위와 같은 compositionality가 노드와 에지를 포함하는 예측 작업을 가능하게 함


본 연구의 실험은 노드에 대한 다중 레이블 분류 작업 및 노드의 쌍(에지)가 주어졌을 때 에지의 존재 여부를 예측하는데(서로 다른 노드들이 연결되었는지) 중점을 두고 있음.

nodevec의 주요 단계들은 병렬화가 가능하며 대규모 네트워크로의 확장이 가능하다는 장점이 있음.

본 연구의 contribution을 다시 설명하자면
1. 효율적이고 확장 가능한 네트워크 표현 학습 알고리즘 node2vec 제안
2. 해당 알고리즘은 네트워크 이론에서 확립된 원칙들에 잘 부합. structural equivalence, homophily를 반영하는 유용한 표현들을 학습하는데 유연성을 제공함
3. 이웃 보존 목표를 기반으로 node2vec을 포함한 표현 학습 알고리즘을 에지에 대한 태스크로 확장이 가능
4. 실제 데이터 셋에 대한 다중 레이블 분류, 링크 예측을 통해 node2vec의 성능을 경험적으로 평가


<br/>


# 2.Related Work
특성 공학은 머신러닝의 주요 화두였으며, 네트워크에서 노드에 대한 유용한 특성을 생성하는 기존의 방식들은 기술자의 수작업을 필요로 했었음.

이와 반대로 본 연구의 목적은 그러한 특성 공학의 작업을 완전히 자동화 하는 것.

비지도 학습에서는 인접행렬, 라플라시안 등 그래프에 대한 행렬 표현의 특성들을 활용했었음. 이러한 방법들은 선형대수학 관점에서 차원 축소로 이해할 수 있고, PCA나 IsoMap같은 선형 및 비선형의 차원 축소 기법들이 많이 제안되어 왔다. 하지만 이런 방법들은 계산 및 성능 측면에서 문제점이 존재한다.
1. 데이터 행렬에 대한 고유값 분해 연산은 고비용이며, 비용을 줄이기 위해서는 성능의 일정 부분을 포기해야 함 (trade off)
2. 네트워크의 다양한 표현들 (homophily, structural equivalence)을 학습하기에 부족함. 네트워크에서 관찰되는 다양한 패턴들에 대해 robust하지 않음.
3. 네트워크와 예측 작업의 관련성에 대한 가정을 필요로함. 해당 가정들은 특정 태스크엔 적합할 수 있어도 일반화하기는 힘듬.


NLP 분야의 최근 발전들은 단어와 같은 discrete objects에 대한 표현 학습의 새로운 방향을 제시했다. 특히 중심 단어로부터 주변 단어를 예측하는 Skip-gram 모델을 살펴 보면 - Skip-gram model aims to learn **continuous feature representations for words** by optimizing a **neighborhood preserving** likelihood objective.

해당 모델은 다음과 같은 가정을한다.
1. 비슷한 의미를 가진 단어들은 비슷한 맥락에서 나옴. (homophily : 비슷한 기능을 하는 노드는 같은 커뮤니티에 속함)
2. similar words tend to appear in similar word neighborhoods (structural equivalence : similar nodes tend to appear in similar network neighborhood)


Skip gram 모델에 영향을 받아 네트워크를 document (ordered sequence of words) 관점에서 표현하는 방식들이 제안되어 옴.
네트워크를 ordered sequence of nodes로 변환.
노드에 대한 다양한 샘플링 전략이 존재하지만 (샘플링에 따라 학습되는 표현들이 달라질 수 있음), 모든 예측 작업과 네트워크에 공통적으로 적용될 수 있는 만능 전략은 없음 - 기존 연구들은 샘플링 전략 측면에서 네트워크 샘플링에 **유연성**을 제공하는 것에 실패함
=> **Our algorithm node2vec overcomes this limitation by designing a flexible objective that is not tied to a particular sampling strategy and provides parameters to tune the explored search space**

<br/>

# 3. Feature Learning Framework
네트워크는 그래프 G = (V, E)로 표현

![](2021-09-30-20-51-40.png)  mapping function from nodes to feature representation, V x D size matrix

![](2021-09-30-20-53-19.png)   network neighborhood of node, u : every source node of V, S : sampling strategy

Skip-gram 아키텍쳐를 네트워크에 적용. node u의 feature representation이 주어졌을 때 network neighborhood Ns(u)를 관찰할 확률을 최대화하는 f를 찾는 최적화 문제

![](2021-09-30-20-57-41.png) - Eq1

해당 최적화 문제를 tractable(미분등의 계산이 가능)하게 만들기 위해 2개의 가정이 필요함


1. 조건부 독립. feature representation이 주어졌을 때 특정 neighborhood node를 관찰할 가능도는 다른 neighborhood를 관찰하는 사건에 독립.
![](2021-09-30-21-01-38.png)

2. Symmetry in feature space. 소스 노드와 neighborhood node는 feature space에서 서로에게 symmetric한 effect를 가짐. 따라서 모든 소스 노드 - neighborhood node의 pair에 대한 조건부 가능도는, 두 노드의 feature들의 내적에 의해 모수화되는 softmax에 의해 표현됨. 

    ![](2021-09-30-21-04-48.png)

    소스 노드 u의 feature representation이 주어졌을 때 가능한 neighborhood node n_i의 가능도

    소스 노드 u의 feature representation 벡터 f(u)와 가능한 neighborhood node n_i의 feature representation 벡터의 내적을 계산


위의 가정을 반영하면 처음의 목적함수 Eq1는 다음과 같이 표현

![](2021-09-30-21-07-55.png) - Eq 2

![](2021-09-30-21-08-33.png)   per node partition function

\=> 해당 함수는 대규모 네트워크에서 계산이 어렵기 때문에 negative sampling을 활용한 근사치를 계산함

SGD를 통해 Eq2의 모델 parameter를 정의하는 f를 최적화

위와 같이 Skip-gram을 베이스로 하는 feature learning 방법론을 제안할 수 있고, 텍스트의 linear nature를 고려하면 이웃의 개념은 연속적인 단어들에 대한 슬라이딩 윈도우로 정의될 수 있다. (sliding window over consecutive words) 

하지만 네트워크는 텍스트와 달리 linear하지 않기때문에 이웃에 대한 richer notion이 필요하다. 이러한 문제를 해결하기 위해 주어진 소스 노드 u에 대해 많은 수의 이웃들을 랜덤하게 샘플하는 방식을 사용한다.

Ns(u) are not restricted to just immediate neighbors but can have vastly different structures depending on the sampling strategy S. 

## 3.1 Classic Search Strategies

소스 노드의 이웃을 샘플링하는 문제를 local search 관점에서 이해할 수 있음.

![](2021-09-30-21-26-50.png)

Fig 1 : 주어진 소스 노드 u가 있을 때 이웃 Ns(u)를 생성하는 과정. Sampling strategy S를 공정하게 비교하기 위해 Ns의 크기를 k로 고정하고, 노드 u에 대한 다양한 이웃 집합을 생성. k개 노드로 구성된 이웃 집합 Ns(u)를 생성하는 sampling strategy에는 2종류가 있음.

1. BFS : 너비 우선 탐색 -> Neighborhood Ns is restricted to nodes which are immediate neighbors of the source (소스 노드에 인접한 모든 노드들을 우선적으로 탐색)
   
2. DFS : 깊이 우선 탐색 -> Neighborhood consists of nodes sequentially sampled at increasing distances from the source node
(소스 노드에서 시작해서 다음 분기로 넘어가기 전에 해당 분기를 완벽하게 탐색, 미로를 탐색할 때 한 방향으로 갈 수 있을 때 까지 가다가 더 이상 갈 수 없으면 다시 가장 가까운 갈림길로 돌아와서 탐색 진행)

BFS, DFS 보충 : https://gmlwjd9405.github.io/2018/08/14/algorithm-dfs.html


BFS, DFS모두 학습된 표현을 탐색해나가는 극단적인 방안들. (extrem scenarios in terms of the search space they explore leading to interesting implications on the learned representation)

노드에 대한 예측 작업은 두 가지 유사성에 의해 충돌될 때가 있음.

1. homophily 가정에서 서로 연관성이 높은 노드들은 유사한 네트워크 군집이나 커뮤니티에 속하도록 임베딩 되어야 함

2. structural equivalence 측면에서 유사한 구조적 역할을 가진 노드들은 서로 비슷하게 임베딩 되어야함 

  * 1과 다르게 2는 노드들의 연결성(connectivity)를 중요시하지 않음 => 서로 멀리 떨어진 노드여도 유사한 구조적 역할을 담당할 수 있음

BFS는 structural equivalence, DFS는 homophily를 표현함
- BFS는 근처의 노드들을 우선적으로 탐색하기 때문에 노드의 이웃들에 대한 microsopic view 제공. 그래프의 일부분(소스 노드 근처) 만을 탐색하는 경향이 있어 variance 작음
 
- DFS는 네트워크의 다양한 범위를 탐색하고, 소스 노드 u로부터 멀리 떨어진 노드까지 탐색이 가능함. 이웃들에 대한 macro view 제공. 
   
   - 노드간의 dependency와 그 속성을 추론하는데 효과적이지만 K만을 탐색할 수 있다는 제약이 존재하고, variance가 큼
   - 보다 멀리(깊게) 탐색함으로써 보다 복잡한 dependency를 표현할 수 있으나 표현력이 떨어질 수 있음 


## 3.2 node2vec

A flexible neighborhood sampling strategy which allows us to **smoothly interpolate between BFS and DFS**

Biase random walk를 통해 DFS, BFS 방식 모두를 활용하여 이웃을 탐색하는 것이 가능함.

<br/>

### 3.2.1 Random Walks

고정된 길이 만큼의 random walk를 수행하게 되고, i 시점에서 생성되는 이웃 노드는 

![](2021-09-30-21-52-24.png)

C i : i th node in the random walk, starting with C 0 = u

transition probability btw nodes v and x

<br/>

### 3.2.2 Search bias a

BFS, DFS 처럼 극단적으로 어느 한 쪽(homophily, structural equivalence)에 특화된 방식이 아닌 두 가지 방식을 잘 융합하여 실제 네트워크의 특성을 잘 반영할 수 있도록 해야 함.

Our random walks should accommodate for the fact that these notions of equivalence are not competing or exclusive,
and real-world networks commonly exhibit a mixture of both

파라미터 p, q를 통해 random walk를 제어 (2nd order)

![](2021-09-30-21-58-41.png)

![](2021-09-30-21-58-52.png)

d : t, x의 최단 거리 (0, 1, 2 중 하나)

t -> v -> x (현재 위치는 v, v와 x의 전이 확률을 계산)

![](2021-09-30-22-00-03.png)

파라미터 p, q를 통해 random walk의 속도를 제어하고, BFS와 DFS의 interpolation을 가능하게 함
Parameters p and q control how fast the walk explores and leaves the neighborhood of starting node u. In particular, the parameters allow our search procedure to (approximately) interpolate between BFS and DFS and thereby reflect an **affinity** for different notions of node equivalences


**Return parameter p**

P를 높은 값으로 설정할 수록 이미 방문한 노드를 샘플링할 가능성이 줄어듬. P가 작을수록 random walk를 backtrack하게 만들어 random walk의 과정을 출발 노드 u에 가깝게 만듬.


**In out parameter q**

q > 1이면 BFS와 유사하며 출발 노드에 대한 local view 정보를 얻음 (inward), q < 1이면 DFS와 유사하지만 무작정 깊게 탐색하지는 않음 (outward exploration)


**Benefits of random walks**

BFS, DFS만을 사용하는 것보다 많은 이점이 있음 - 시간 복잡도, 공간 복잡도 측면에서 효율적


### 3.2.3 The node2vec algorithm

![](2021-10-01-15-02-07.png)

출발 노드 u를 결정하는 것에 대한 implicit bias 존재하며, 모든 노드들에 대한 표현을 학습하는 것이 목적이기 때문에 해당 bias를 고정된 l 길이에 대한 r회 random walk를 수행하여 상쇄함.

random walk의 매 step마다 2차 마르코프 체인의 전이확률(v -> x)에 의해 샘플링이 수행됨. 전이확률은 precompute되는 사항이기 때문에 randomwalk는 o(1)의 시간복잡도를 갖는 효율적인 과정이된다.

전이확률 계산, ranom walk simulation, SGD를 활용한 최적화는 sequential하게 수행되며 병렬화가 가능하고, node2vec의 확장성에 기여한다.

<br/>

# 3.3 Learning edge features

node2vec 알고리즘은 네트워크의 노드들에 대한 풍부한 표현의 학습을 가능하게 하는 준 지도학습 방법. 노드-노드의 pair에 대한 예측이 필요할 때가 있다. (link prediction

random walk 과정은 기본적으로 네트워크의 노드들의 연결 구조에 기반하고 있기 때문에, 개별 노드들의 feature representation들에 대한 bootstraping 접근법을 사용하여 노드와 노드의 pair에 적용.

g(u, v) = f(u) o f(v) ,  o : binary operater  g : pair (u, v)의 representation



# 4.1 Case study : Les Miserables Network

노드는 소설 레미제라블의 등장인물, 에지는 등장인물들의 동시출현 (co-appearing)
네트워크는 77개의 노드, 254개의 에지로 표현, feature representation의 차원 d=16
node2vec을 사용하여 feature representation을 학습하고 k-means를 활용하여 군집화를 수행, 시각화


1. p=1, q=0.5
   
등장인물에 대한 군집, 커뮤니티를 색깔로 구별하고 서브 플랏에서 등장인물끼리 얼마나 상호작용 하는지를 표현. 등장인물간의 에지는 동시출현에 기반한다는 점에서 homophily를 표현하고 있음. 

2. p=1, q=2

파란색 노드는 소설의 서브 플랏에서 일종의 bridge 역할을 하는 등장인물들을 표현.
노란색 노드는 상호작용이 적은 주변 인물을 표현. 


![](2021-10-01-15-26-41.png)



# 5. Discussion And Conclusion

본 연구를 통해 탐색 기반의 최적화를 통한 네트워크 feature learning 알고리즘을 제안. 해당 알고리즘은 exploration-exploitation의 trade off를 효과적으로 해결하는 이점이 있음. (파라미터 p, q를 통한 interpolation으로 BFS, DFS의 장점을 적절하게 활용)

예측에 있어 학습된 표현에 대한 interpretability를 제공. (BFS는 homophily를 설명, DFS는 structural equivalence를 표현)

node2vec의 이웃 탐색 전략은 flexible('random' walk), controllable함. 

link prediction에서도 높은 성능을 보임




