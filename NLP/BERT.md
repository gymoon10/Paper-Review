# Abstract
본 연구는 새로운 language representation model BERT를 제안. BERT는 모든 레이어에서 왼쪽(forward) 및 오른쪽(backward) 맥락 모두에 공동으로 conditioning하여, labeling되지 않은(비지도 학습) 텍스트로부터 bidirectional representation을 사전 학습한다.

사전학습된 BERT모델은 추가적인 하나의 아웃풋 레이어를 추가함으로서 미세조정되고, 이를 통해 질문-답변, 언어 추론 등과 같은 다양한 태스크를 수행할 수 있다.

<br/>

# 1. Introduction
자연어 처리 태스크에서 사전 학습된 언어 모델은 굉장히 유용하게 적용되어 왔음. 사전 학습된 language representation을 예측이나 분류등의 downstream task에 적용하는 대표적인 전략은 feature based와 fine tuning이다.

ELMO같은 feature based 접근법은 사전학습된 표현을 추가적인 피쳐로 사용함으로서 task specific한 모델 아키텍쳐를 사용한다.

OpenAI GPT같은 fine tuning 접근법은 최소한의 task specific한 파라미터를 필요로하고, 모든 사전학습된 파라미터에 대한 미세조정을 통해 주어진 downstream task에 적합하게 학습된다. 

두 접근법은 사전 학습에 있어 일반적인(다른 task에 일반화가 가능한) language representation을 학습하는데 있어 방향성이 없는 language model을 사용한다는 공통점이 있음.

본 연구는 현재의 기술이 사전학습된 representation의 유용성을 제한한다고 주장한다. (특히 미세조정에 있어서)
특히 표준 언어 모델의 undirectional한 특성은 사전학습에 사용될 수 있는 모델 아키텍쳐를 제한하는 단점이 있다. 

예를 들어 GPT의 경우 left to right(forward) 아키텍쳐를 사용하기 때문에 모든 토큰들은 Transformer의 self-attention계층에서 이전 시점의 토큰들만 참고할 수 있다. 이러한 제한은 문장 단위의 태스크에 있어 sub-optimal할 뿐 아니라 질문-답변 태스크 같이 양방향의 문맥 정보를 학습해야 하는 경우 성능 저하로 이어질 수 있다.

본 연구는 미세 조정 베이스의 접근법인 Bidirectional Encoder Representations from Transformers를 제안함. BERT는 **Masked Language Model**을 통해 undirectionality 제약을 해결. MLM은 입력의 토큰들 일부를 랜덤하게 마스킹하고, 문맥 정보를 바탕으로 마스킹된 토큰을 정확히 예측하도록 학습이 된다. MLM은 이를 통해 정방향과 역방향의 문맥 정보를 통합한 representation을 학습함. 또한 **next sentence prediction**을 사용하여 텍스트-텍스트 pair의 representation 역시 학습할 수 있도록 한다.

본 연구의 contribution은 다음과 같다.

1. MLM을 통해 deep bidirectional representation을 학습

2. BERT는 최초의 미세 조정 베이스의 representation 모델로 대규모의 토큰, 문장 단위의 태스크에서 다른 task specific 아키텍쳐들보다 SOTA 성능을 보임.

<br/>

### Additional Summary
BERT는 특정 분야에 국한된 기술이 아닌, 모든 자연어 처리 분야에서 좋은 성능을 보이는 범용 language model.

자연어 처리 모델의 성능은 데이터가 충분히 많다면 단어의 의미를 잘 반영하는 벡터로 표현하는 embedding이 큰 영향을 미침.

-> 이 embedding 과정에서 BERT를 사용, 특정 과제를 하기 전 **사전 훈련된 embedding**을 통해 특정 과제의 성능을 향상시킬 수 있는 모델
(대량의 코퍼스를 encoder가 embedding하고, 이를 전이하여 fine tuning을 한 다음 특정 태스크를 수행)

<br/>

# 2. Related Work

## 2.1 Unsupervised Feature-base Approaches

일반적으로 활용 가능한 단어의 representation을 학습하는 것이 자연어 처리 분야의 주된 관심사였음. 사전 학습된 word embedding은 최근 NLP 트렌드의 핵심적인 분야이고, 혁신적인 성능 향상을 보여줬음. (improvements over embeddings from scratch)

word embedding 벡터에 대한 사전학습을 위해 left to right 언어 모델이 사용되어 왔고, 세부적으로 문장과 단락에 대한 임베딩으로 일반화 되었다. 해당 모델은 문장에 대한 표현을 학습하기 위해 이전의 결과 (prior work)들을 활용했다.

ELMO의 경우 context sensitive 피쳐를 left to right, right to left 모든 방향에서 추출하고 그 둘을 concatenating함으로서 개별 토큰들에 대한 최종적인 문맥 표현을 생성함. 

![image](https://user-images.githubusercontent.com/44194558/135740945-00b2b255-a4a1-4a8d-bd51-c9192cb1d0de.png)

<br/>

## 2.2 Unsupervised Fine-tuning Approaches

contextual token representation을 생성하는 인코더는 라벨링되지 않은 텍스트로부터 사전 학습되고 supervised downstream task에 미세조정된다. 이러한 방식은 비교적 적은 수의 파라미터들만 학습되면 된다는 장점이 있음.  

![image](https://user-images.githubusercontent.com/44194558/135741126-3df5572e-6e83-49b4-95c5-c0e7bc584dbe.png)

![image](https://user-images.githubusercontent.com/44194558/135741219-87030b79-1924-4983-bf66-99a09f623de4.png)

<br/>

## 2.3 Transfer Learning from Supervised Data

대규모의 데이터셋에 대한 지도 학습의 전이가 효과적. 컴퓨터 비전 분야에서도 ImageNet으로 사전 학습된 모델을 미세조정하는 것 같이, 대규모 데이터로부터 사전 학습된 모델을 활용한 전이 학습이 갖는 중요성이 주목 받고 있음. 

<br/>

# 3. BERT

사전 학습 과정에서 라벨링되지 않은 데이터를 바탕으로 훈련되고, downstream task의 라벨링된 데이터를 바탕으로 사전 학습된 파라미터들을 미세조정함. 각각의 downstream task들은 처음에는 모두 똑같은 사전 학습된 가중치를 공유하지만, 최종적으로는 고유한 미세 조정된 모델을 가진다. BERT는 서로 다른 태스크들에 대해서 공통된 아키텍쳐를 가진다. 

<br/>

## Model Architecture

Multi-layer bidirectional Transformer encoder -> GPT와 달리 bidirectional self attention 사용

![image](https://user-images.githubusercontent.com/44194558/135741856-3d2fa584-0718-48ef-aed1-82d343a48c0e.png)

<br/>

## Input/Output Representations

![image](https://user-images.githubusercontent.com/44194558/135742826-1481bb49-4f3b-4c58-8d91-2b0accf5bbe0.png)

![image](https://user-images.githubusercontent.com/44194558/135742851-fd136759-f1b8-43ca-985f-196eb8db23e6.png)

![image](https://user-images.githubusercontent.com/44194558/135743026-20ee0d1b-eced-4cad-b1d9-7be141ec0b4e.png)

위 3가지 임베딩을 합치고, layer 정규화와 드롭아웃을 적용하여 입력으로 사용. BERT는 이미 총 3.3억개의 단어로 구성된 대용량 코퍼스를 정제하고 임베딩하여 학습시킨 모델.


**Sentence** : Arbitrary sapn of contiguous text (실제 언어적인 문장 개념과는 다름)

**Sequence** : Input token sequence to BERT (단일 문장일수도, 문장-문장 pair일수도 있음)

<br/>

### Contextual Embedding
문맥을 반영한 임베딩을 사용. d_model=768이기 때문에 모든 단어들은 768차원의 임베딩 벡터가 되어 BERT의 입력으로 사용됨.

![image](https://user-images.githubusercontent.com/44194558/135800152-01ee2cd2-c80e-4d72-82d7-06f70a6d5212.png)

내부적인 연산(attention등)을 거쳐 각 단어 토큰에 대해 768차원의 hidden state 벡터 출력.

![image](https://user-images.githubusercontent.com/44194558/135800242-edbad6a3-162f-4b31-9115-1b63617c3d0a.png)

BERT의 연산을 거친 출력 임베딩은 **문장의 문맥을 모두 참고한 문맥**을 반영한 임베딩. [CLS], love 모두 문장의 모든 단어 벡터들을 참고한 후에 문맥 정보를 가진 벡터가 됨. 화살표는 모든 단어를 참고하고 있다는 것을 의미.

![image](https://user-images.githubusercontent.com/44194558/135800415-355b8809-159f-46e3-9c61-a53af3e42e5d.png)

**하나의 단어가 모든 단어를 참고**하는 연산은 BERT의 모든 layer에서 수행됨. 트랜스포머 인코더를 12개 쌓은 BERT는 각 layer마다 내부적으로 **multi head self attention**, **position wise feed forward** 연산을 수행.

<br/>

### WordPiece Embedding

30000개의 토큰 단어들에 대해 **WordPiece embedding** 수행 

  -  bag of words model은 단어 개수 만큼의 차원을 지닌 벡터 공간을 이용, 차원의 저주를 방지하기 위해 제한된 개수의 단어를 이용하기 때문에 OOV(미등록 단어) 문제 발생
  -  WordPiece는 단어를 표현할 수 있는 subword units인 **글자(character)**로 모든 단어를 표현
  -  자주 등장하면서 가장 긴 길이의 subword를 하나의 단위로 만들어 OOV 문제를 해결
     - 공연, 개막공연의 빈도수가 같으면 공연이나 개막은 유닛으로 이용하지 않음 (개막공연을 나타내기 위한 부분들이기 때문)
  -  언어에 상관없이 모두 적용이 가능, 해당 언어의 특징을 반영한 토크나이저를 만들 필요가 없음
  - 참고 : https://lovit.github.io/nlp/2018/04/02/wpm/

<br/>

### Position Embedding

단어의 위치 정보를 표현. 기본적인 positional encoding은 사인, 코사인 함수를 이용하여 위치에 따라 다른 값을 가지는 행렬을 만들어 이를 단어 벡터와 더하는 방식. BERT는 position embedding을 학습을 통해서 얻음.

![image](https://user-images.githubusercontent.com/44194558/135800693-a928d83d-e81a-4589-bb01-94f7da38b7c1.png)

실질적 입력인 WordPiece Embedding + Position Embedding

실제 BERT는 문장의 최대 길이를 512로 설정하기 때문에 총 512개의 position embedding 벡터가 학습됨. 

<br/>

### Segment Embedding

서로 다른 2개의 문장을 구분. **BERT에서의 문장은 실제 우리가 알고있는 언어적 문장과는 다름**

> Ex. Question and Answering
> 
> [질문, 본문(paragraph)] 두 종류의 텍스트를 입력으로 받으면 segment embedding, [SEP]을 통해 구분. 이때 1개의 본문은 실제로는 다수의 문장으로 구성될 수 있음.

> **BERT가 말하는 2개의 문장은 실제로는 두 종류의 텍스트 혹은 두 개의 문서일 수 있음**

![image](https://user-images.githubusercontent.com/44194558/135801144-a8bd834e-3b55-42ad-9392-eba30217f556.png)



## 3.1 Pre-training BERT

BERT를 사전 학습 시키기위해 left to right나 right to left아키텍쳐를 사용하지 않고, 2개의 비지도 학습을 통해 사전학습 시킴.

언어의 특성을 잘 학습하도록 MLM, NSP를 사용함.

### Task 1 : Masked LM

![image](https://user-images.githubusercontent.com/44194558/135742864-c420698a-a06a-457b-8455-dce2da058505.png)

![image](https://user-images.githubusercontent.com/44194558/135801577-598a9744-3989-4503-a11b-d28d99ccbbb0.png)

 * dog 토큰을 [MASK]로 변경, BERT 모델은 원래 단어인 dog를 제대로 예측할 수 있도록 학습
 * Masking된 위치의 출력층 벡터만 예측과 학습에 사용 (다른 위치의 벡터들은 사용 x)

표준적인 조건부 언어 모델은 left to right나 right to left를 통해서만 학습될 수 있다. 양방향에 대한 조건부 모델의 경우 multi layered context를 통해 타깃 단어를 단순하게 보고 예측할 수 있다. (allow each word to indirectly **see itsef**)

Deep bidirectional representation을 학습하기 위해 입력 토큰의 일부분을 랜덤하게 masking한다. 마스크 토큰의 최종 hidden state 벡터 출력이 소프트맥스 함수를 통해 원래의 단어를 잘 예측할 수 있도록 학습이 되어야 함. Denoising auto encoder처럼 전체 입력을 재구성하는 것이 아닌, 마스킹된 부분만을 예측한다. 

하지만 mask 토큰은 미세 조정 단계에서는 등장하지 않기 때문에 사전 학습과 미세 조정 사이의 mismatch가 발생하는 문제가 있다. 이러한 문제를 해소하기 위해 모든 마스크 토큰의 80%만을 실제로 마스킹함.

![image](https://user-images.githubusercontent.com/44194558/135742883-af7ce7e3-23d6-4d3b-9d26-a0e83ecd47cf.png)

![image](https://user-images.githubusercontent.com/44194558/135801750-9fd9a7df-84c0-4c2f-8d99-7685ab5cbca7.png)

* dog 토큰은 [MASK]로 변경
* he 토큰은 랜덤한 단어인 king으로 변경
* play 토큰은 변경되지 않았지만 예측에 사용
* BERT는 어떤 단어가 변경되었는지 모르므로 원래 단어를 잘 예측할 수 있도록 학습되어야 함

### Task 2 : Next Sentence Prediction

QA(Question Answering) 혹은 NLI(Natural Language Inference)는 두 문장들의 relationship을 이해할 필요성이 있지만, 언어 모델로는 학습하기 어려운 면이 있음. **문장들의 관계**를 학습하기 위해 일종의 이진 분류인 NSP를 수행함. 

두 개의 문장을 준 후에 서로 이어지는지 아닌지를 맞추는 방식으로 훈련.

![image](https://user-images.githubusercontent.com/44194558/135742974-6bf5959e-0918-4cac-8222-976c6a55e590.png)

![image](https://user-images.githubusercontent.com/44194558/135802014-e80a4273-ae0e-47c2-87c7-6cc6952f8c64.png)

* 두 문장이 실제로 이어지는지 아닌지를 [CLS] 토큰의 최종 출력층을 활용하여 이진 분류 문제를 풀도록 함. (분류 문제를 풀기 위해 추가된 특별 토큰)

**BERT는 MLM, NSP의 loss를 합하여 학습이 동시에 이루어질 수 있게 함.**

참고 : https://wikidocs.net/115055

<br/>

## 3.2 Fine-tuning BERT

사전 학습된 BERT에 우리가 풀고자 하는 downstream task의 데이터를 추가적으로 학습시켜 테스트. 태스크를 실질적으로 해결하기 위해 BERT를 사용하는 단계.


![image](https://user-images.githubusercontent.com/44194558/135743877-f2f24c19-cf8c-4081-851a-8430de06f8c2.png)

Transformer의 self attention 메커니즘은 BERT가 다양한 downstream task들을 간단하게 모델링할 수 있게 해줌. 입력의 개수에 따라 알맞게 하나의 sequence로 생성해서 모델의 입력으로 제공함. 두 문장이 입력으로 제공된 경우 하나의 sequence로 생성하고, 두 문장 사이의 self attention도 수행.

 text pair를 인코딩하는 기존의 방식들은 문장들에 bidirectional cross attention을 적용시키전에 문장들을 개별적으로 인코딩하였음. BERT는 self attention을 활용하여 두 문장들을 개별적으로 인코딩하는 과정을 통합함. 통합된 텍스트(concatenated)에 대한 self attention은 두 문장들간의 효과적인 bidirectional cross attention을 가능하게 함.

개별 태스크에 적합한 입력과 출력을 모델에 입력으로 제공해서 파라미터들을 해당 태스크에 맞게 end-to-end로 갱신. Token representation은 token level task(sequence tagging, question-answering)의 입력으로 사용되고 [CLS]토큰은 분류 목적으로 사용됨.

fine tuning은 상대적으로 적은 비용으로 수행이 가능하며 fine tuned된 모델은 pre trained된 모델과 구조적인 차이가 거의 없음

<br/>

**Single Sentence Classification**

![image](https://user-images.githubusercontent.com/44194558/135744032-cb2c0e04-cada-4164-a9d3-6776123f20f8.png)

![image](https://user-images.githubusercontent.com/44194558/135802302-e4079467-a233-483a-8348-d35670dbe969.png)

분류 문제를 위한 특별 토큰인 [CLS] 위치의 출력층에 Dense/FC layer를 추가하여 분류에 대한 예측 수행.

**Single Sentence Tagging**

ex) 품사 태깅, 객체 태깅

![image](https://user-images.githubusercontent.com/44194558/135744050-a733ef58-ac5a-4b4b-92b9-24ebd6fe1145.png)

![image](https://user-images.githubusercontent.com/44194558/135802449-f8e7470c-02f1-4a7c-9814-f9a24b2af771.png)

각 토큰의 최종 출력층에 추가적인 Dense layer를 추가하여 분류에 대한 예측 수행.

<br/>

**Sentence Pair Classification**

텍스트의 쌍을 입력으로 받는 태스크 수행.  ex) 자연어 추론

![image](https://user-images.githubusercontent.com/44194558/135744064-6d3beb59-a2fb-4176-b453-7b77604f2001.png)

![image](https://user-images.githubusercontent.com/44194558/135802625-c790b46a-9364-4252-a36a-e517e16b8806.png)


<br/>

**Question and Answering**

질문, 본문 2개의 텍스트 쌍을 입력. 질문과 본문을 입력받으면, 본문의 일부분을 추출해서 질문에 답변.

> Q : "강우가 떨어지도록 영향을 주는 것은?"
> 
> A : "기상학에서 강우는 대기 수증기가 응결되어 중력의 영향을 받고 떨어지는 것을 의미합니다. 강우의 주요 형태는 이슬비, 비, 진눈깨비, 눈, 싸락눈 및 우박이 있습니다."
> 
> 정답 : 중력 

![image](https://user-images.githubusercontent.com/44194558/135744084-04907c29-ec63-40d3-aed8-7d443708e659.png)

![image](https://user-images.githubusercontent.com/44194558/135802768-d3575b68-81b9-4ce3-b4e6-888a76ddfeff.png)

<br/>

![image](https://user-images.githubusercontent.com/44194558/135744153-67c7edd4-8ec4-4c41-9502-0cc03a713755.png)