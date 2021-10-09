## Abstract

방대한 양의 라벨링 되지 않은 text corpora에 비해, 특정 태스크 학습을 위해 필요한 라벨링된 데이터는 많지 않기 때문에 모델이 적절한 작업을 수행하기 어려운 점이 존재함.


그래서 GPT-1은 라벨링 되지 않은 text corpus에서 언어 모델에 대한 **Generative Pre-Training**과 각 태스크에 적합한 **discriminative fine tuning**을 통해 학습을 개선함. 기존의 방법들과는 달리 fine tuning 중가에 **task aware input transformer**를 사용하여 모델 아키텍쳐의 변경을 최소화 하면서 효과적인 전이 학습을 가능하게 한다.

<br/>

## 1. Introduction

raw text(unlabeled text)로부터 효과적으로 학습하는 능력은 NLP에 있어 지도학습에 대한 의존을 줄이는데 매우 중요하다. 대부분의 딥러닝 방법론들은 manually labeled data를 필요로 하기 때문에, 다양한 도메인에 적용될 수 있는 능력이 제한되기도 한다. 또한 비지도 학습 방식을 사용하여 representation을 학습하는 것이 지도 학습의 성능을 개선시키는데 큰 도움이 되기도 한다.

하지만 unlabeled text로부터 단어 수준 이상의 정보를 얻는 것은 매우 어려운데 그 이유는 다음과 같다.

   1. 단어 수준 이상의 정보를 얻기가 쉽지 않음
  2. 어떤 최적화 목적 함수(optimization objective)가 전이 학습에 용이한 text representation을 학습하는데 효과적인 지 알 수 없음
  3. 학습된 representation을 특정 태스크에 효과적으로 전달하는 일치된 의견이 없음
  4. 위와 같은 불확실성은 semi supervised learning을 어렵게 만듬

따라서 본 연구에서는 **unsupervised pre training과 supervised fine tuning을 결합한 semi-supervise 접근을 사용**하며 다양한 종류의 태스크에 약간의 fine tuning만으로도 전이가 가능한 범용(generalized) representation을 학습하는 것을 목표로 한다.

GPT-1은 다음과 같은 과정을 거친다.

  1. unlabeled data에서 language modeling objective를 사용하여 신경망에 대한 초기 파라미터들을 학습
  2. 그 이후 supervised objective에 해당하는 target task에 적용하며 학습된 파라미터들을 조정 (adapt)

본 모델은 텍스트 데이터의 long term dependency에 강인하고, 기존 순환 신경망에 비해 구조화된 메모리를 사용하는 transformer의 구조를 적용함. 이러한 transformer를 적용시켰기 때문에 다양한 태스크에 대해 강인한 전이 학습 performance를 보장할 수 있고, 전이 학습 도중에는 task specific input adaptation을 활용한다. 

(This model choice provides us with a **more structured memory for handling long-term dependencies** in text, compared to alternatives like recurrent networks, resulting in **robust transfer performance** across diverse tasks. During transfer, we utilize **task-speciﬁc input adaptations** derived from traversal-style approaches [52], which **process structured text input as a single contiguous sequence of tokens**.)

위와 같은 adaptation 덕분에 사전 학습된 모델의 아키텍쳐에 대한 변경을 최소화하며 효과적인 미세 조정을 가능하게 한다.

<br/>

## 2. Related Work


### Semi-supervised learning for NLP

기존의 접근법들은 unlabeled data로부터 단어나 구문 단위의 통계량을 추출하고, 해당 정보를 지도학습의 feature로써 활용했다. 최근 몇 년간의 연구에서는 unlabeled corpora에서 학습된 워드 임베딩 결과가 다양한 태스크에 있어 유용하게 사용된다는 것이 밝혀졌다. 하지만 이러한 접근법과는 달리 본 연구는 단어 수준의 정보를 넘어선 **higher level semantics**를 학습하는 것이 목표이다.

최근의 접근법들은 unlabeled data로부터 word level semantic 이상의 것들을 학습하는 것에 주목해왔고, unlabeled corpus에서 학습되는 구나 문장 단위의 임베딩이 다양한 태스크에 적합한 벡터 표현으로 encode하는데 사용되어 왔다.

<br/>

### Unsupervised pre-training

unsupervised pre training은 semi supervised learning의 한 종류로, 좋은 초기 환경을 제공하는 것을 목적으로 한다. (ﬁnd a good initialization point instead of modifying the supervised learning objective)

본 연구와 가장 유사한 방식으로는 language modeling objective로 사전 학습을 진행하고, 지도 학습의 target task로 미세 조정하는 방식이다. 기존의 방식은 LSTM을 사용하였기 때문에 short range에 대한 예측 성능에 있어 한계가 있었는데, 본 연구는 transformer 구조를 활용하여 보다 긴 의존성을 학습할 수 있다는 장점이 있음.

이외에도 사전 학습된 언어, 번역 모델의 hidden representation을 feature로써 사용하는 다양한 방식들이 있지만 ex) ELMO, CoVe 다양한 태스크에 대해 서로 다른 많은 수의 파라미터를 사용하고 학습해야 한다는 단점이 있다.

<br/>

### Auxiliary training objectives

보조적인 unsupervised training objective를 추가하는 것도 semi supervised learning의 변형 중 하나. 과거에는 POS-tagging, chunking, NER등을 이용하여 semantiv role labeling의 성능을 향상시킨 사례가 있고, 최근에는 보조적인 language modeling objective를 target task objective에 추가하여 sequence labeling task에 대한 성능을 향상시켰다. 본 연구도 보조적인 objective를 활용함.

<br/>

## 3. Framework

학습은 크게 2가지 단계로 구성되는데, 우선 대용량의 텍스트를 기반으로 language model을 학습하고, 미세 조정 단계에서 레이블된 데이터를 기반으로 학습을 진행한다.

<br/>

### 3.1 Unsupervised pre-training

레이블링 되지 않은 데이터에 대해 다음과 같은 가능도를 최대화 하는 방향으로 언어 모델링을 학습함

![image](https://user-images.githubusercontent.com/44194558/136647607-32d837fc-64ab-43bb-8af4-07767c598068.png)


![image](https://user-images.githubusercontent.com/44194558/136647422-45157667-5c6e-4a94-b9b2-81a01aa9f1a2.png)  - Eq1

  * K : window size
  * 조건부 확률은 파라미터 theta에 대해서 계산됨
  * 모든 파라미터는 SGD를 통해 학습

본 연구는 언어 모델에 **multi layer transformer decoder**를 사용. 해당 모델은 모든 입력 토큰에 대해 multi headed self attention을 수행하고, 해당 결과를 position wise feedforward layer의 입력으로 제공하여 토큰들에 대한 output distribution P(u)을 생성한다. 

![image](https://user-images.githubusercontent.com/44194558/136647541-901956b6-1990-4c67-a452-3d9b88577e59.png)

  * U : 토큰들에 대한 context vector
  * W_e : token embedding matrix
  * W_p : position embedding matrix


<br/>

### 3.2 Supervised fine-tuning

Eq1을 통해 사전 학습을 완료하면, 학습된 파라미터는 target task에 맞게 미세 조정된다. 해당 과정에서는 레이블링된 데이터셋 C를 사용한다고 가정하며 m개의 입력과 레이블 y로 구성됨.
해당 입력들은 사전 학습된 모델의 입력으로 제공되고, 마지막 transformer block을 통해 최종적으로 출력된다. 해당 최종 출력은  W_y를 파라미터로 갖는 linear output layer의 입력으로 제공되어 레이블 y를 예측하게 됨.

![image](https://user-images.githubusercontent.com/44194558/136647841-57fe4181-e1a2-47c1-b7a5-e847fb7b4598.png)

언어 모델을 미세 조정하는 과정에서 보조 objective를 사용했을 때 지도학습 모델의 일반화 가능성을 향상시키고, 빠르게 수렴할 수 있게 한다는 장점이 있었음. 다음과 같은 방식으로 최적화를 진행한다.

![image](https://user-images.githubusercontent.com/44194558/136647928-0fd0d7c5-f1ac-4e9e-b060-59870c996130.png)

미세 조정 단계에서 추가적으로 필요한 파라미터는 linear output layer를 구성하는 파라미터 W_y와 delimiter token을 위한 임베딩 뿐이다.

<br/>

### 3.3 Task-specific input transformers

![image](https://user-images.githubusercontent.com/44194558/136648004-33283650-5532-419a-b245-47275a9a58f4.png)

특정 태스크는 ordered sentence pairs같이 구조화된 입력을 제공해야 하는 경우가 있다. ex) Question-Answering, textual entailment

이와 같은 경우에는 contiguous text를 바탕으로 학습한 사전 학습된 모델과 불일치하는 면이 있기 때문에 task specific cusomization이 필요하게 된다. 이때 구조화된 입력을 ordered sequence로 변환하여 사전 학습된 모델이 처리할 수 있도록 한다. 모든 변환은 시작과 끝을 알려주는 토큰 s, e를 사용하기 때문에 task에 따라 모델의 아키텍쳐를 변경할 필요가 없게된다. 

  * **Textual entailment** : premise p와 hypothesis h를 delimeter token $로 구분
   
  * **Similarity** : 두 입력 문장의 순서에 의미가 없기 때문에 모든 경우의 수를 고려. A&B, B&A 두 결과에 element wise addition을 통해 최종 representation을 생성
  
  * **Question & Answering, Commonsense Reasoning** : context document z, question q, 답변 집합 a를 입력으로 제공. 모든 가능한 답변들을 본문, 질문과 하나씩 연결해서 [z ; q ; a_k]의 representation을 생성. 개별 답변들에 대한 representation은 독립적으로 모델의 입력으로 제공되고 softmax를 통해 결과를 예측하게 됨.


<br/>

## 4. Experiment

Book corpus dataset을 이용하여 언어 모델을 학습을 진행함. 상대적으로 긴 문장들이 포함되어 있어 long range information을 학습하는데 적합함.

모델은 기본적으로 transformer구조를 따름. **12 layer decoder only transformer**를 **masked self attention head**를 이용하여 구현함.
(12개의 attention head, 768차원의 hidden states)

BPE(Byte Pair Encoding), L2 규제, GELU 활성화 함수 사용

**BPE** : 단어를 문자(character) 단위로 나누어 subword생성, OOV 문제를 해결함

참고 :  https://wikidocs.net/22592


![image](https://user-images.githubusercontent.com/44194558/136648601-70d08b35-e840-4d10-839b-913cb76998d0.png)


<br/>

## Appendix

GPT는 transformer의 decoder부분만 사용하는 언어 모델. GPT 1, 2, 3의 모델 구조는 동일하고 학습 데이터의 양과 decoder block의 개수에서 차이가 난다.

GPT는 forward 방향으로 학습을 진행함

![image](https://user-images.githubusercontent.com/44194558/136648742-f2e5f00f-6505-47c5-a7ec-3c4817deb4a8.png)

GPT의 seq2seq학습 과정은
  1. 문장이 BPE를 통해 토큰으로 분리
  2. 토큰으로 분리된 문장이 embedding matrix를 통해 임베딩 벡터로 변환된 후, decoder block에 입력

     ![image](https://user-images.githubusercontent.com/44194558/136648859-120f9d3a-b670-40ba-9d97-5cbbfececef2.png)

  3. decoder block을 거친 후 softmax를 거쳐 가장 확률이 높은 단어를 선택

     ![image](https://user-images.githubusercontent.com/44194558/136648882-bb5cb3c2-68d1-49fc-ba98-6440f4f577a6.png) 

학습에서는 s 단어를 입력했을 때의 최종 출력(예측)과 실제 정답간의 cross entropy loss를 최소화 하는 방식으로 진행됨.

추론에서는 s 단어를 입력했을 때 나오는 출력을 다음 step의 입력으로 넣어 반복적으로 단어가 출력되게 함 (generative)
