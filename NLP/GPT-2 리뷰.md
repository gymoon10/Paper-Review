## Abstract

Question & Answering, 기계 번역 등의 NLP 태스크는 전통적으로 task specific 데이터에 대한 지도학습을 통해 훈련되었지만, GPT-2는 이러한 supervision없이 언어 모델만을 사용해 해당 태스크들을 해결하고자 하는 모델이다. 모델의 크기는 **zero-shot task trnasfer**의 성능을 결정짓는데 굉장히 중요한 요소인데, 본 연구가 제안하는 가장 큰 모델인 GPT-2는 1.5B개의 파라미터를 가지는 trnasformer로 WebText 데이터에 대해 학습이 완전히 이루어지지 않았는데도 불구하고 NLP 태스크 8개 중 7개 분야에서 SOTA 성능을 달성했다.

<br/>

## 1. Introduction

머신러닝 시스템은 대규모의 데이터셋, 높은 용량의 모델, 지도 학습 3가지 조합으로 학습을 가속해왔다. 하지만 이런 시스템은 task specific할 뿐 아니라 data의 분포에도 민감하다는 문제가 있다. 즉 현재의 시스템은 **competent generalists**가 아닌 narrow expert라고 특징지을 수 있는데, 본 연구는 다양한 태스크에 적용가능한 **general system**을 구축하여, 최종적으로는 학습 데이터를 일일히 라벨링하는 작업을 필요없게 만드는 것이다. 

본 연구는 sigle task training on single domain datasets 경향이 현재 시스템에 있어 **generalization**역량을 부족하게 만든다고 생각한다. 즉 현재의 모델 아키텍쳐를 바탕으로 다양한 범위와 도메인의 태스크에 대해 일반화가 가능한 **robust system**을 구축하는 것이 목적이다.

Multi task learning은 다양한 도메인과 태스크를 동시에 수행하며 학습하는 방법으로 일반화 성능을 위한 탁월한 방법론이지만, 현재의 머신러닝 모델은 충분한 수준의 일반화를 위해 방대한 양의 training example을 필요로하기 때문에 최대한 많은 양질의 데이터를 확보하고, 합리적인 목적 함수를 정의하는 것이 굉장히 어려운 작업이 된다.

현재 NLP 분야에서 가장 성능이 좋은 모델들은 BERT, GPT-1같은 pre trained language model인데, transformer의 self attention을 사용하기 때문에 task specific한 구조를 필요로 하지 않는다. 특히 대부분의 downstream task에 있어 하나의 output layer만 추가하면 되는 수준의 **유연한  형태의 transfer**가 가능함. 하지만 이러한 방법들도 미세 조정 단계에서는 여전히 지도학습을 필요로 한다는 단점이 있음.

본 연구는 **general mehods of transfer**의 경향을 잇는 연구로, 언어 모델이 downstream 태스크를 **zero shot setting** (매개 변수나 모델 아키텍쳐 수정 x)을 통해 수행할 수 있다는 것을 밝힌다. 즉 지도학습 형태의 미세 조정없이 zero shot downstream task가 가능한 범용성 있는 언어 모델의 일반화 능력을 입증한다.

<br/>

## 2. Approach

본 연구의 핵심은 언어 모델이다. 언어 모델링은 문장 ![image](https://user-images.githubusercontent.com/44194558/136651981-c48e09df-28a7-4be5-b22d-de081d6a9ea0.png) 의 가변 길이를 갖는 토큰 sequence ![image](https://user-images.githubusercontent.com/44194558/136651993-12d2830d-99fc-4d4b-8177-66f8dd2ecf68.png) 를 사용하여 문장 x의 비지도 분포인 p(x)를 추정하는 방법이다.

언어는 자연스러운 서순과 맥락이 있기 때문에 문장이라는 결합 확률 분포는 조건부 확률의 연쇄 법칙으로 분해되어 모델링된다(joint probabilities over symbols as the product of conditional probabilities). 이러한 접근법은 p(x)에 대한 추정에 있어 tractable한 샘플링을 가능하게 할 뿐 아니라 ![image](https://user-images.githubusercontent.com/44194558/136652040-2524e606-b003-4bab-b669-35b8c6320059.png) 형태의 복잡한 조건부 확률도 다룰 수가 있다. 최근에는 transformer의 self attention 구조 덕분에 이러한 조건부 확률을 계산할 수 있는 모델의 표현력이 크게 상승되었다.

single task 수행을 위한 학습은 p(output | input)를 추정하는 확률 프레임워크로 표현될 수 있다. 본 연구가 지향하는 일반적인 시스템은 많은 태스크들을 수행할 수 있어야 하기 때문에 (입력이 똑같은 상황에서도) p(output | input, task)와 같은 형태로 모델링을 한다. 이와 같은 **task conditioning**이 아키텍쳐 레벨에서 구현되는 것은 'task specific encoders and decoders' 논문을, 알고리즘 레벨에서 구현되는 것은 'the inner and outer loop optimization framework of MAML'을 참고하는것이 좋다. 하지만 언어는 특정 task, 입력, 출력을 토큰들의 sequence로 specify하는데 유연한 방안을 제공한다.
  > Ex 1 ) 번역 태스크 : (프랑스어로 번역하세요, 영어 텍스트, 프랑스어 텍스트) 형태의 sequence
  > Ex 2 ) 독해 능력 태스크 : (질문에 답하세요, document, 질문, 답변) 형태의 sequence

언어 모델링은 출력에 대한 예측을 수행하는 supervision 없이도 태스크에 대해 학습할 수 있다. 지도 학습의 global minimum은 비지도 학습의 global minimum과 차이가 없기 때문에 (Since the supervised objective is the the same as the unsupervised objective but only evaluated on a subset of the sequence, the global minimum of the unsupervised objective is also the global minimum of the supervised objective.) 비지도 학습의 목적 함수를 수렴하는 단계까지 최적화시킬 수 있는지가 중요하다. 즉, 비지도 학습의 목적을 가지고 학습을 진행하면 지도 학습의 목적에 대해서도 만족할 수 있다. 

While it is a large step from the well-posed setup described above to the messiness of “language in the wild”, Weston (2016) argues, in the context of dialog, for the need to **develop systems capable of learning from natural language directly** and demonstrated a proof of concept - reward signal없이 (지도 학습 x) foward prediction만을 활용하여 QA 태스크를 학습. 비슷한 예로 인터넷은 상호작용 커뮤니케이션을 필요로 하지 않는 (지도학습을 필요로 하지 않는) 풍부한 양의 정보를 가지고 있다. 
따라서 본 연구는 대용량의 데이터를 바탕으로하는 언어 모델이 자연어 시퀀스에 명시된 특정 태스크에 대해 보다 정확한 예측을 할 수 있도록 학습할 수 있을 것이라고 가정한다.
만약 언어 모델이 이러한 것을 할 수 있다면 보다 효과적으로 비지도 멀티 태스크에 대한 학습을 수행할 수 있을 것이고, 본 연구는 **zero shot setting**을 기반으로 한 다양한 종류의 태스크를 통해 이러한 사실을 검증하고자 함.

<br/>

### 2.1 Training Dataset

GPT-2의 가장 큰 목적은 **미세 조정없이 비지도 사전 학습만을 통해 zero shot으로 downstream task를 진행할 수 있는 일반적인 언어 모델을 개발**하는 것이다. 따라서 대부분의 기존 연구들이 언어 모델을 single domain 텍스트에 대해 훈련 시켰던 것과 달리 본 연구는 다양한 도메인과 맥락을 가진 텍스트 내용을 수집하기 위해 가능한 많고 다양한 데이터를 구축하고자 한다 (diverse and nearly unlimited). 본 연구는 기존의 web scrap data의 품질이 떨어진다고 판단하여 직접 만든 WebText 데이터를 사용함. 품질을 보장하기 위해 해당 데이터는 사람이 직접 필터링을 수행했고, 레딧으로부터 최소 3개의 평가를 받은 외부 링크만을 사용하여 수집, 추가적인 중복 제거, 위키피디아 같은 대중적 문서를 제외하는 과정을 거침.

![image](https://user-images.githubusercontent.com/44194558/136654217-a5a06ada-36b1-4bca-bd7f-19b0ef622dba.png)

위와 같은 40GB의 텍스트로 구성된 WebText 데이터셋은 **품질, 크기, 다양성**을 동시에 고려한 데이터셋

<br/>

### 2.2 Input representation

언어 모델은 어떠한 문자열에 대해서도 확률을 계산할 수 있어야 하기 때문에, 현재의 대용량 언어 모델들은 lower casing, 토큰화의 전처리 과정을 거친다. 본 연구는 GPT-1과 마찬가지로 **BPE** 방식을 사용함. BPE는 subword 기반의 인코딩 방법으로 문자 단위로 단어를 분해하여 vocabulary를 생성하고, 반복을 통해 빈도수가 높은 문자의 pair을 지속적으로 vocabulary에 추가하는 방식. BPE는 자주 등장하는 토큰들의 시퀀스와 그렇지 않은 토큰 시퀀스의 문자 수준 입력을 잘 보간(interpolation)할 수 있다는 장점이 있고, OOV에 대해서도 합리적인 토큰화가 가능함.

유니코드 시퀀스에서 사용되는 BPE는 13만 개 이상의 매우 큰 vocabulary를 필요로 하는데 비해, Byte 수준의 BPE는 256개의 vocabulary만을 사용하기 때문에 본 연구는 BPE를 Byte 수준의 문자열에 적용하는 시도를 함. 하지만 Byte 수준의 BPE는 dog., dog?, dog! 같이 유의미하지 않은 variation을 추가하는 경향이 있어 한정된 vocabulary를 최적으로 사용하지 못하는 단점이 있기 때문에 본 연구는 문자 수준 이상의 병합을 막아 **vocbulary 공간을 최적으로 활용하며 input representation을 구성**할 수 있도록 했음.

<br/>

### 2.3 Model

Transformer를 활용한 언어 모델을 사용. 몇 가지의 변경 사항은 다음과 같음
  1. Layer normalization이 attention과 feedforward의 출력층이 아닌 입력층에 적용
  2. residual layer의 파라미터를 초기화 시에 1 / N으로 scale weight를 적용 (N : residual layer 개수)
  3. context 길이를 1024로 batch size는 512

![image](https://user-images.githubusercontent.com/44194558/136654587-a1d93440-566c-463f-87b9-df821a8af4d7.png)

Layer norm의 위치의 경우 해당 계층을 feedforward의 출력부에 두는 것보다 입력부에 두는 것이 학습 시 layer별 gradient의 정도가 상대적으로 고른 편이라고 함

<br/>

## 3. Experiments

BPE를 사용하기 때문에 UNK 토큰이 발생할 가능성은 40조 바이트 중 26번 정도로 매우 낮음. 즉, 대부분의 단어가 생성 가능하고 BPE를 통해 역 토큰화도 가능함.

WebText로 학습된 GPT-2 모델은 전반적인 도메인과 데이터셋에 잘 맞는 모델임을 7개의 태스크에 대한 실험을 통해 입증. 한 개의 태스크를 제외하고는 zero shot learning이 상당히 효과적임.

![image](https://user-images.githubusercontent.com/44194558/136654742-d788192a-59ae-4993-8fcd-be8541792ffb.png)

책에 빠진 개체명이나 명사에 대해 모델이 예측하도록 한 실험. 명사 예측에 대해서는 93.3%, 일반 명사에 대해서는 89.1%의 정확도를 보였음.

![image](https://user-images.githubusercontent.com/44194558/136654755-08a12790-66eb-4605-8079-90190f3bc46f.png)

긴 문장 (long range dependency)에 대해서도 모델링을 효과적으로 수행하는지 실험. 최소 50개의 단어 위치에서 마지막 단어를 예측하는 실험

![image](https://user-images.githubusercontent.com/44194558/136654824-05c4e7e8-ffc3-4a84-8c6a-7ab6764beee2.png)

보다 나은 요약문을 생성함

![image](https://user-images.githubusercontent.com/44194558/136654857-816ce0e4-e343-422b-9fe8-d319e266f105.png)

입력된 텍스트 질문 정보 안에서 정답을 출력하는 단문 단답의 문제도 잘 해결함.

![image](https://user-images.githubusercontent.com/44194558/136654925-713099d4-0380-403c-b9ef-3b95b3c04474.png)

<br/>

## 4. Generalization vs Memorization

훈련 데이터와 테스트 데이터의 과도한 중복은 모델의 memorization을 유도하고 일반화 성능을 왜곡할 수 있음. 이런 현상은 본 연구에서 만든 WebText 데이터에서도 충분히 나탈 수 있음 (학습한 데이터 양이 방대하기 때문). 본 연구는 GPT-2가 단순히 데이터 학습으로 기억된 정보만을 가지고 답을 하는 것이 아님을 입증함.

본 연구는 overlap의 기준을 8 grams overlap 기준으로 중복되는 정도를 판단함. 이를 통해 본 연구가 학습한 WebText가 common text라고 할 수 있는 PTB나 WikiText보다 overlap되는 정도가 낮음을 보임.

![image](https://user-images.githubusercontent.com/44194558/136655104-99f5da73-8ac3-4bd5-adbf-d068bd2de23a.png)

다음과 같이 학습, 테스트 데이터에서의 성능은 유사하며 모델 크기에 따라 동시에 성능이 증가하고 있기 때문에 모델의 성능 개선은 memorization으로 인한 것이 아니며, 아직도 개선될 여지가 있음을 보임 (underfitting)

![image](https://user-images.githubusercontent.com/44194558/136655308-dac46a83-3d0d-4708-8077-9569ad2d1f2a.png)

GPT-2 is still **underﬁtting on WebText** in many ways.


<br/>

## Appendix

학습 목적, 개념 : Task Conditioning, Zero Shot Learning, Zero Shot Task Transfer (fine tuning X)

데이터 : WebText, 8백만 개 이상의 문서에서 추출된 40GB 정도의 텍스트 데이터

모델 아키텍쳐 : GPT-1보다 매개변수가 10배 많음

Task Conditioning을 통해 모델은 다른 작업에 대해 동일한 입력에 대해서도 다른 출력을 생성할 것으로 예상함. 언어 모델에 대한 task conditioning은 예제 또는 자연어 명령을 제공하여 수행.

Zero shot learning은 예시가 전혀 제공되지 않고 모델이 주어진 지침에 따라 작업을 이해하는 zero shot task trnasfer의 특별한 케이스
GPT-1이 미세 조정을 위해 단어 순서를 조정하는 것과 달리, GPT-2의 입력은 모델이 작업의 특성을 이해하고 답변을 제공할 것으로 기대하는 형식으로 제공됨. (모델이 영어를 프랑스어로 번역하는 작업임을 이해하고 태스크 수행)

