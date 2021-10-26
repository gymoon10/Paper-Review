# End-to-end Sequence Laebling via Bi-directional LSTM-CNNs-CRF

<br/>

## Abstract

이전의 전통적인 **Sequence labeling**은 많은 양의 task specific한 사전 지식과 수작업에 기반한 특성 공학(feature engineering) 및 전처리에 의존해야 했지만, 본 연구는 양방향 LSTM, CNN, CRF를 사용하여 **자동으로** 단어 및 문자 레벨의 유용한 표현들을 학습하는 새로운 신경망 네트워크를 소개한다. 본 연구가 제안하는 시스템은 특성 공학이 필요 없는 end to end 모델이기 때문에 다양한 범위의 sequence labeling 태스크에 적용될 수 있다. POS tagging과 NER 분야에 대해 sequence labeling의 성능을 평가했고 SOTA 성능을 얻었다.

<br/>

## 1. Introduction

전통적으로 성능이 높았던 sequence labeling 모델들은 HMM(Hidden Markov Model), CRF(Conditional Random Fields) 같이 수작업에 기반한 피쳐들에 크게 의존하는 **통계적 선형 모델**이었다. 예를 들어 영어 POS tagging에 있어 높은 성능을 위해서는 정교하게 디자인된 단어 철자에 대한 특성들이 요구되지만, 이와 같은 task specific한 지식들은 비용이 많이 요구될 뿐 아니라 sequence labeling 태스크가 다른 태스크나 도메인에 일반적으로 적용되는 것을 어렵게 한다.

최근에는 워드 임베딩을 통해 단어에 대한 분산 표현을 입력으로 받는 **비선형 신경망 네트워크** 가 NLP 분야에 성공적으로 적용되고 활용되었다. 고정된 길이를 갖는 window 맥락을 활용하여 독립적으로 단어에 대한 라벨을 예측 하는 feed forward 모델, RNN, LSTM, GRU 등이 시퀀스 데이터를 처리하는데 사용되어 왔고, RNN 베이스의 몇 몇 모델들은 sequence labeling task에 적용되어 왔다.  

하지만 워드 임베딩을 통해 단어에 대한 분산 표현을 입력으로 사용하는 신경망 시스템조차 수작업에 의존하는 특성 공학 작업을 완벽히 대체하지 못했다. 따라서 본 연구는 **전처리와 수작업을 필요로 하지 않는 end to end sequence labeling model**을 제안한다.

`It is a truly end- to-end model requiring no task-speciﬁc resources, feature engineering, or data pre-processing be- yond pre-trained word embeddings on unlabeled corpora.` 

그렇기 때문에 본 연구가 제안하는 모델은 다양한 언어와 도메인의 sequence labeling 태스크에 일반적으로 적용될 수 있고, 모델링 단계는 다음과 같다.

1. CNN : Encode character-level information of a word into its character-level representation
2. BLSTM : 문자와 단어 레벨의 표현들을 결합하여 양방향 LSTM의 입력으로 넣고, 개별 단어들에 대한 문맥 정보를 학습
3. Sequential CRF : 주변의 맥락 정보를 파악하여 전체 문장의 라벨을 예측. (jointly decode labels for the whole sentence)

<br/>

## 2. Neural Network Architecture

### 2.1 CNN for Character-level Representation

CNN은 단어를 구성하는 문자들로부터 형태학적(morphological) 정보를 추출하고, 그 정보들을 neural representation으로 인코딩하는 효과적인 방법이다. CNN은 주어진 단어로부터 문자 레벨의 표현을 추출하고, 문자 임베딩을 거치기 전에 드롭 아웃을 적용시킨다. 

![image](https://user-images.githubusercontent.com/44194558/138804939-4ec9ff88-b19b-4147-9504-d065c4769d16.png)

<br/>

### 2.2 Bi-directional LSTM

#### 2.2.1 LSTM Unit

RNN은 그래프의 사이클을 토오해 시간 변동성을 포착하는 강력한 연결 모델이지만 기울기 소실등의 문제가 있기 때문에 LSTM을 사용한다.

![image](https://user-images.githubusercontent.com/44194558/138805503-317ebbc6-7f3a-476f-99a4-c68c4a27157c.png)

![image](https://user-images.githubusercontent.com/44194558/138805582-66ada83e-f904-47fd-a1a8-1add7e7905d1.png)

  * X_t : t 시점에서의 입력 벡터 (워드 임베딩)
  * h_t : t 시점과 이전 시점들의 모든 유용한 정보를 저장하고 있는 은닉 벡터 (output vector)
  * U : 입력에 대한 가중치 행렬
  * W : 은닉 상태 h_t에 대한 가중치 행렬
  * b : 편향

#### 2.2.2 BLSTM

Sequence labeling 태스크에 있어 양방향의(과거+미래) 문맥 정보를 활용하는 것이 권장되지만 LSTM의 h_t는 t 시점과, 그 이전 시점들의 정보만 저장하고 있기 때문에 양방향 LSTM을 활용한다. LSTM 2개를 사용하여 주어진 시퀀스를 forward, backward 방향으로 각각 입력으로 넣은 후 각 방향의 hidden state들을 결합하여 최종 아웃풋으로 사용.

<br/>

### 2.3 CRF

Sequnece labelig에 있어 독립적으로 디코딩하는 것이 아니라 주변의 정보를 활용하여 주변과의 상관관계를 파악하는 것이 권장된다.

'It is beneﬁcial to consider the **correlations between labels in neighborhoods** and **jointly decode the best chain of labels** for a given input sentence.'

  ex) 형용사 뒤에는 동사보다 명사가 많이 등장

<br/>

**문장 z가 주어졌을 때 label sequence y에 대한 조건부 확률**

![image](https://user-images.githubusercontent.com/44194558/138806562-037c7140-1edb-4226-aa8c-045302ccc025.png)

  * n개의 단어로 구성된 문장 Z (Input Sequence) : ![image](https://user-images.githubusercontent.com/44194558/138816031-b0f38fc7-d70f-4e67-a402-5df51de2c188.png)
  * Z에 대한 label sequence : ![image](https://user-images.githubusercontent.com/44194558/138816155-babde0d8-92b1-4e85-8d15-ac67ef49563f.png)
  * Y(z) : z에 대해 모든 가능한 label sequence들의 집합


CRF의 학습은 아래의 로그 가능도를 최대화하는 방식으로 이루어짐. (Maximum conditional likelihood estimation for a training set {(z, y)})

![image](https://user-images.githubusercontent.com/44194558/138816643-89278d2a-eacb-4ce4-9474-87f289784c53.png)

디코딩은 최대의 조건부 확률을 갖는 label sequence y*를 찾는 식으로 이루어짐.

![image](https://user-images.githubusercontent.com/44194558/138816902-c5111f7e-9af8-4be7-84a0-903cb7f6227e.png)

<br/>

### 2.4 BLSTM-CNNs-CRF

BLSTM의 output vector를 CRF 레이어의 입력으로 하여 네트워크를 구축. 문장을 구성하는 개별 단어에 대해

1. CNN을 사용하여 문자 레벨의 표현을 학습 (문자 레벨의 임베딩을 입력으로 받음)
 
2. 문자 레벨의 표현 벡터가 단어 임베딩 벡터와 결합되어 BLSTM에 입력으로 주어짐

3. BLSTM의 출력 벡터가 CRF 계층의 입력으로 들어가 최적의 laebl sequence를 디코딩함. **jointly decode**

* BLSTM의 입력, 출력에 드롭 아웃을 적용시키는 것이 성능을 유의미하게 개선시킴


![image](https://user-images.githubusercontent.com/44194558/138817491-7e586b7b-a849-4607-8e66-7d5d6f90b191.png)


<br/>

## 3. Network Training

Theano library, GeForce GTX TITAN X GPU를 사용하여 모델을 학습


### 3.1 Parameter Initialization

**Word Embeddings**

사전 학습된 GLOVE (100 차원), Senna (50 차원), Word2Vec (300 차원) 에 대해 실험을 진행함.

**Character Embeddings**

30 dim uniformed sampled vector from range [-sqrt(3 / dim), sqrt(3 / dim)]

**Weight Matrices and Bias Vectors**

Randomly initialized with uniform samples from [-sqrt(6/r+c), sqrt(6/r+c)]

<br/>

### 3.2 Optimization Algorithm

SGD with batch size=10 & momentum=0.9, lr=0.01(POS Tagging), 0.015(NER)

학습률은 매 에포크 마다 0.5의 decay rate로 업데이트됨.

기울기 폭발을 방지하기 위하여 gradient clipping 기법을 사용. (gradient norm을 5보다 작게 유지)

보다 나은 성능과 과적합 방지를 위해 조기 종료, 드롭 아웃, initial embedding에 대한 미세 조정(역전파를 통해 그래디언트를 업데이트하는 과정에서 embedding도 같이 업데이트)을 활용.

![image](https://user-images.githubusercontent.com/44194558/138819066-4fb8a6a9-913c-4d31-aeac-d91735523517.png)

<br/>

## 4.Experiments

### 4.1 Data Set

![image](https://user-images.githubusercontent.com/44194558/138819530-e22c75a9-cbef-40c2-b70e-d7b8c860994a.png)


### 4.2 Main Results

![image](https://user-images.githubusercontent.com/44194558/138819589-b8ed89da-a97d-4f6d-a76b-ac46ada3aa97.png)


### 4.3 Comparison with Previous Work

#### 4.3.1 POS Tagging

![image](https://user-images.githubusercontent.com/44194558/138819718-19b3256b-2117-4823-9038-1c4d37aa0478.png)


#### 4.3.2 NER

![image](https://user-images.githubusercontent.com/44194558/138819802-cb795057-0bcf-4515-bf1c-be915449b0e7.png)


#### 4.4 Word Embedding

![image](https://user-images.githubusercontent.com/44194558/138819863-9124c557-4574-4183-bd7c-d347aa3c8992.png)


#### 4.5 Effect of Dropout

![image](https://user-images.githubusercontent.com/44194558/138819929-8e4249d0-ed27-4da5-b5ab-ab00b2f52cbf.png)


#### 4.6 OOV Error Analysis

![image](https://user-images.githubusercontent.com/44194558/138820128-0a5fb82b-dfb6-4494-99e4-8541bbfc14dc.png)

  * IV : In Vocabulary
  * OOTV : 임베딩은 되었지만 학습 세트에 없는 단어
  * OOEV : 학습 세트에는 있지만 임베딩이 안됨
  * OOBV : 학습 세트에도 없고 임베딩도 안됨

Joint decoding에 CRF를 추가함으로써 OOBV 문제에 보다 효과적으로 대응할 수 있게 됨