# Momentum Contrast for Unsupervised Visual Representation Learning

<br/>

## Abstract

NLP 분야에서 GPT, BERT 등을 활용한 unsupervised learning이 우수한 성과를 보이는 것에 비해 컴퓨터 비전 분야에서는 아직 미약한 단계이다. 이러한 격차는 서로의 signal space에서의 차이에 기인한다. 언어의 특성 중 하나인 한정된 벡터 공간을 딕셔너리로 만들 수 있는 것에 비해, 컴퓨터 비전 분야에서의 신호는 연속적이며, 인간의 언어처럼 체계적이지 않고, 벡터 공간의 차원이 굉장히 크다는 문제가 있다.

 - 언어 : discrete signal spaces (words, sub-words) <-> 비전 : continuous & high dimensional spaces

하지만 최근에 Contrastive Learning을 통해 컴퓨터 비전 분야에서도 supervised learning 못지 않은 성과를 달성 하고 있으며, 그 중 하나가 MoCo이다.

<br/>

**Contrastive Learning**

![image](https://user-images.githubusercontent.com/44194558/154008433-c07fc93c-6094-4802-a84a-5e1d6858165f.png)

이미지 데이터셋 내에서 instance 단위로 similarity 학습 (유사한 이미지들은 같은 공간에 mapping, 다른 이미지들은 서로 다른 공간에 mapping).

레이블 정보 없이 관측치(instance) 레벨에서 학습, 공유하는 representation, semantic structure가 존재하기 때문에, 관측치 사이의 유사성을 기반으로, 관측치 끼리 서로 구분하도록 학습한다면 레이블 정보 없이도 유용한 representation을 학습할 수 있음.

<br/>

`From a perspective on contrastive learning [29] as dictionary look-up, we build a dynamic dictionary with a queue and a moving-averaged encoder. This enables building a large and consistent dic- tionary on-the-ﬂy that facilitates contrastive unsupervised learning.`

MoCo는 ImageNet 분류 문제에서 뛰어난 성능을 보이고, 다양한 downstream task에서 유용하게 활용될 수 있는 representation을 제공한다.

<br/>

## Appendix 

**참고** : 

https://www.youtube.com/watch?v=wyBzB9iRveI&t=471s

https://www.youtube.com/watch?v=2Undxq7jlsA

<br/>

![image](https://user-images.githubusercontent.com/44194558/154006700-5a3f4c88-80fa-4cc7-b1c4-74aee1457881.png)

**SimCLR (end-to-end)**

![image](https://user-images.githubusercontent.com/44194558/154007681-4b8491fd-7152-461f-a45f-9f4f80277410.png)


현재 mini-batch에 존재하는 샘플들을 dictionary(negative samples)로 활용. Query, key가 동일한 encoder를 거치지만, dictionary 크기가 batch-size에 국한된다는 단점 존재.

 - 유용한 visual representation 학습을 위해서는 많은 수의 negative sample이 필요하기 때문에 batch-size가 커야 함. 실제로 4000 이상의 batch를 사용. (computational limit)

<br/>

**Instance Discrimination (Memory bank)**

![image](https://user-images.githubusercontent.com/44194558/154009497-657d1c3b-72cc-41ef-a035-d3d8cd3c7aa5.png)

 - Anchor : 현재 분류하고자 하는 이미지
 
 - Positive : Anchor의 과거 정보
 
 - Negative : 나머지 이미지들

모든 데이터셋의 embedding 정보를 보유하고 있는 memory bank를 만들고, mini-batch만큼 샘플링하는 방식을 통해 computational limit을 극복하고, 많은 양의 negative sample들을 활용할 수 있음.

하지만, encoder가 계속 학습되어가는 데 비해 memory bank에서 추출하는 negative sample들에 대한 feature, representation은 갱신되지 않기 때문에 inconsistency의 문제가 발생하고, 데이터 샘플 별로 학습에 기여하는 정도가 다름.

<br/>

**MoCo**

`Building large and consistent dictionaries for unsupervised learning with a contrastive loss.`

1. Negative representation을 저장하는 **queue**

2. Key encoder의 **momentum update** 


<br/>

## 1. Introduction

최근의 연구들은 contrastive loss를 활용한 unsupervised visual representation learning을 제안해왔고, 이러한 방식들은 **dynamic dictionary**를 구축하는 것으로 간주될 수 있다.

사전의 keys (tokens)들은 data(images or patches)로부터 샘플링되어, encoder network를 통과한다. Unsupervised learning은 dictionary look-up을 통해 encoder를 학습시킨다. 인코딩된 query는 matching key와는 유사하고, 다른 key들과는 달라야 한다. 이 학습은 contrastive loss를 줄이는 방향으로 수행된다.

![image](https://user-images.githubusercontent.com/44194558/154006381-8a85429d-7dff-4c6e-9214-f006bb1f6add.png)

 - 이미지 x가 존재할 때, x는 2개의 augmentation에 의해 query, matching key로 분류. 각각은 encoder, momentum encoder를 거쳐 feature를 산출하고, 서로 같은 이미지에서 도출된 query, matching key의 contrastive loss는 최소화하면서, 다른 이미지들과의 contrastive loss는 최대화해야함.
 
Dictionary를 데이터 샘플들에 대한 **queue**의 형태로 간주하고, 서서히 변동시킴 (현재 mini batch의 encoding된 feature를 enque, 과거의 mini batch의 feature는 deque). Encoder가 갱신됨에 따라 과거의 representation은 더 이상 consistent하지 않기 때문.

이 때 consistency 유지를 위해 **momentum based moving average**를 사용 (momentum update). 

![image](https://user-images.githubusercontent.com/44194558/154011432-970601f4-e89e-480a-9985-e5cc79d6c2f7.png)


 - ` Moreover, as the dictionary keys come from the preceding several mini-batches, a slowly progressing key encoder, implemented as a momentum-based moving average of the query encoder, is proposed to maintain consistency.`

<br/>

## 2. Related Work

### Loss functions

`Contrastive losses [29] measure the similarities of sample pairs in a representation space. Instead of matching an input to a ﬁxed target, in contrastive loss formulations the target can vary on-the-ﬂy during training and can be deﬁned in terms of the data representation computed by a network [29].`

 - Representation space에서 sample pair들의 유사성을 측정. 고정된 타겟, 입력을 매칭하는 대신에 타겟이 학습되는 동안에도 바뀔 수 있고, 네트워크로부터 계산된 data representation의 관점에서도 정의될 수 있음.

<br/>

## 3. Method

### 3.1 Contrastive Learning as Dictionary Look-up

`Contrastive learning [29], and its recent developments, can be thought of as training an encoder for a dictionary look-up task, as described next.`

![image](https://user-images.githubusercontent.com/44194558/154013199-d4bd5c04-256e-4b7b-9fa8-7a5a747997d3.png)

 - 내적을 통해 유사도 계산
 
 - 분모 : K개의 negative sample과 하나의 positive(matchin key) 샘플들의 합
 
 - **q를 k+로 분류하려고 노력하는 (K+1)-way softmax-based classifier의 logloss**  

<br>

### 3.2 Momentum Contrast

이미지와 같은 고차원, 연속적인 입려겨에 대해 discrete한 사전을 구축하는 방식. Key들이 랜덤하게 샘플링되고, key encoder도 학습 동안 서서히 변동된다는 점에서 **dynamic**.

`Our hypothesis is that good features can be learned by a large dictionary that covers a rich set of negative samples, while the encoder for the dictionary keys is kept as consistent as possible despite its evolution.`

<br/>

#### Dictionary as a queue

Dictionary를 데이터 샘플들의 queue로 유지하여, 바로 앞의 mini batch에서 encoding된 key를 재사용할 수 있음. Queue를 도입함으로써 dictionary의 크기가 mini-batch 크기로부터 분리됨. 이에 따라 dictionary의 크기도 일반적인 minibatch보다 크기가 클 수 있고, 하이퍼 파라미터 조정을 통해 유연하고 독립적으로 설정될 수 있음.

Dictionary의 샘플들은 점진적으로 바뀜 (현재 mini-batchh가 enqueue 되면서 가장 과거의 mini-batch가 dequeue됨). Queue안의 가장 오래된 mini-batch를 삭제함으로써 consistency를 유지. Dictionary는 항상 모든 데이터의 샘플된 subset을 표현하고 있으며, 유지 측면에서 메모리 효율도 괜찮음.

<br/>

#### Momentum update

Queue를 사용하여 dictionary를 크게 만들 때, 역전파 과정에서 gradient는 queue안의 모든 샘플들에 대해 전파되기 때문에 key encoder를 갱신하는 것이 어려워진다. 

Key encoder의 파라미터 theta_k는 다음과 같이 업데이트

![image](https://user-images.githubusercontent.com/44194558/154016667-c8dfe4bd-746d-4853-b20c-7c1af7e4f40f.png)

 - 역전파를 통해 theta_q만 업데이트됨
 - Momentum coefficient m을 통해 theta_k는 theta_q보다 서서히 갱신됨

`As a result, though the keys in the queue are encoded by different encoders (in different mini-batches), the difference among these encoders can be made small.`

<br/>

### Relations to previous mechanisms

![image](https://user-images.githubusercontent.com/44194558/154017590-4577a6b4-ed46-4c6a-a5f1-f863b5942837.png)

`End to end` : dictionary 크기가 mini-batchh 크기에 의존.

`Memory bank`  
- 각 mini-batch에서의 dictionary는 bank로부터 랜덤하게 샘플링됨 (역전파 필요 x) 
- 샘플링된 key는 과거 epoch에 걸쳐 여러 개의, 다른 단계의 encoder에 대한 것이므로 consistency 떨어짐

<br/>

### 3.3 Pretext Task

Instance discrimination task를 사용. Query, key가 같은 이미지로부터 생성되었으면 positive pair, 아니면 negative. Positive pair는 동일한 이미지에 대해 랜덤한 augmentation을 적용한 결과 중 2개의 랜덤한 view를 선택 (서로 다른 2가지 버전의 random augmentation을 동일한 이미지에 적용).

Encoder는 아무 CNN 네트워크나 가능.

![image](https://user-images.githubusercontent.com/44194558/154018289-138a2f1e-636e-4800-be83-5e588ab07b0c.png)

<br/>

#### 1. Query & Key Encoding

![image](https://user-images.githubusercontent.com/44194558/154019308-8640f30f-308f-468c-b5e6-158d8b50fc9d.png)

 - N : batch size / c : dim 

<br/>

#### 2. 유사도 계산

Positive pair

![image](https://user-images.githubusercontent.com/44194558/154019575-e1844f47-ee8c-43f7-b2a2-0cb1fe7f4e16.png)

 - logit

Negative pair

![image](https://user-images.githubusercontent.com/44194558/154019532-a333a924-d407-4e4d-859b-32144e0f8e92.png)

 - Queue(dictionary)의 모든 요소들을 negative로 간주 (Queue에는 K개의 샘플 존재)
 - Negative의 dim=NxK

Concat

![image](https://user-images.githubusercontent.com/44194558/154020035-982497a2-2cca-483c-bcfc-a3be8b8bf263.png)

<br/>

#### 3. Loss

![image](https://user-images.githubusercontent.com/44194558/154020262-787c28e8-2874-4137-898e-b4a188267a32.png)

이 loss를 encoder로 backpropagation
