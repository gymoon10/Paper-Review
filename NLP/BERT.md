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

30000개의 토큰 단어들에 대해 **WordPiece embedding** 수행 

  -  bag of words model은 단어 개수 만큼의 차원을 지닌 벡터 공간을 이용, 차원의 저주를 방지하기 위해 제한된 개수의 단어를 이용하기 때문에 OOV(미등록 단어) 문제 발생
  -  WordPiece는 단어를 표현할 수 있는 subword units인 **글자(character)**로 모든 단어를 표현
  -  자주 등장하면서 가장 긴 길이의 subword를 하나의 단위로 만들어 OOV 문제를 해결
     - 공연, 개막공연의 빈도수가 같으면 공연이나 개막은 유닛으로 이용하지 않음 (개막공연을 나타내기 위한 부분들이기 때문)
  -  언어에 상관없이 모두 적용이 가능, 해당 언어의 특징을 반영한 토크나이저를 만들 필요가 없음
  - 참고 : https://lovit.github.io/nlp/2018/04/02/wpm/

<br/>

## 3.1 Pre-training BERT

BERT를 사전 학습 시키기위해 left to right나 right to left아키텍쳐를 사용하지 않고, 2개의 비지도 학습을 통해 사전학습 시킴.

언어의 특성을 잘 학습하도록 MLM, NSP를 사용함.

### Task 1 : Masked LM

![image](https://user-images.githubusercontent.com/44194558/135742864-c420698a-a06a-457b-8455-dce2da058505.png)

표준적인 조건부 언어 모델은 left to right나 right to left를 통해서만 학습될 수 있다. 양방향에 대한 조건부 모델의 경우 multi layered context를 통해 타깃 단어를 단순하게 보고 예측할 수 있다. (allow each word to indirectly **see itsef**)

Deep bidirectional representation을 학습하기 위해 입력 토큰의 일부분을 랜덤하게 masking한다. 마스크 토큰의 최종 hidden state 벡터 출력이 소프트맥스 함수를 통해 원래의 단어를 잘 예측할 수 있도록 학습이 되어야 함. Denoising auto encoder처럼 전체 입력을 재구성하는 것이 아닌, 마스킹된 부분만을 예측한다. 

하지만 mask 토큰은 미세 조정 단계에서는 등장하지 않기 때문에 사전 학습과 미세 조정 사이의 mismatch가 발생하는 문제가 있다. 이러한 문제를 해소하기 위해 모든 마스크 토큰의 80%만을 실제로 마스킹함.

![image](https://user-images.githubusercontent.com/44194558/135742883-af7ce7e3-23d6-4d3b-9d26-a0e83ecd47cf.png)

### Task 2 : Next Sentence Prediction

QA(Question Answering) 혹은 NLI(Natural Language Inference)는 두 문장들의 relationship을 이해할 필요성이 있지만, 언어 모델로는 학습하기 어려운 면이 있음. 문장들의 관계를 학습하기 위해 일종의 이진 분류인 NSP를 수행함. 

![image](https://user-images.githubusercontent.com/44194558/135742974-6bf5959e-0918-4cac-8222-976c6a55e590.png)