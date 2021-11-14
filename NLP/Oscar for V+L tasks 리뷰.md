# Oscar : Object-Semantics Aligned Pre-training for Vision-Language Tasks

<br/>

## Abstract

<br/>

컴퓨터 비전과 자연어 처리 태스크에서 이미지-텍스트 쌍의 **cross-modal representations**를 학습하는 대용량의 사전 학습 기법이 널리 사용되고 있다. 기존 방법들은 단순하게 이미지 영역의 특징과, 텍스트의 특징을 결합하고 self-attention 기법을 사용하여 **image-text semantic alignments** (이미지와 텍스트의 의미론적인 대응관계)를 학습하게 했다.

Oscar는 기존의 방법들과 달리 **검출된 이미지의 object tags를 anchor points로 사용하여 이미지와 텍스트의 대응관계를 보다 효율적으로 학습**하게 한다. 본 연구의 아이디어는 이미지에서 명백한 (salient) 객체들은 정확하게 검출하는 것이 가능하고, 이미지와 쌍을 이루는 텍스트에서도 언급이 된다는 관찰에 기인한다. Oscar 모델은 6.5M의 텍스트-이미지 쌍을 대상으로 사전 학습하고, downstream task에서 미세 조정을 수행하여 vision-language understanding과 generation task에서 SOTA 성능을 얻었다.

`Learning of cross-modal representations can be signiﬁcantly improved by introducing object tags detected in images as anchor points to ease the learning of semantic alignments between images and texts.` 

<br/>

## 1. Introduction

<br/>

**Cross-modal representation**을 학습하는 것은 이미지 캡셔닝이나 image-text retrieval같은 V+L (Vision-Language) 태스크의 기본이라고 할 수 있다. VLP (Vision Language Pre-training)에 대한 최신 연구들에 의하면 대용량의 이미지-텍스트 쌍으로부터 일반화가 가능한 표현들을 학습하는 것이 가능하고, task-specific한 데이터에 대해 미세 조정을 수행 함으로써 높은 성능을 얻을 수 있다. 

이와 같은 VLP 모델들은 **multi-layer Transformers**에 기반하고 있다. 모델을 사전학습 하기 위해 기존의 방법들은 이미지 피쳐와 텍스트 피쳐를 단순히 결합하여 (cocatenate) 모델의 입력으로 제공하고, self-attention 기법을 사용하여 이미지 영역들과 텍스트 간의 의미론적 대응관계 (semantic alignments)를 학습할 수 있도록 했다. 하지만 이미지와 텍스트 간의 명백한 대응 관계 정보의 부족은 **alignment modeling**이 약한 지도 학습 태스크가 되게 하였다. 또한 이미지에 대한 visual region들은 오버 샘플링 되거나, 잡음이 많고 불명확한 측면이 있어 V+L 태스크를 보다 어렵게 만들었다.

본 연구는 이미지와 텍스트 간의 의미론적 대응관계의 학습을 쉽게 하기 위해 이미지에서 검출된 object-tags를 anchor point로 도입하여 cross-model representation의 학습을 향상시킬 수 있음을 보인다. 훈련 샘플은 **word sequence, object tags, image region features**로 구성된다. 이 아이디어는 이미지내의 명백한 객체들은 정확하게 검출하는 것이 가능하고, 이미지와 쌍을 이루는 텍스트에서도 언급이 된다는 관찰에 기인한다. Oscar 모델은 대용량의 V+L 데이터셋에 사전 학습 되어있고, 7개의 V+L understanding과 generation task에 대해 미세 조정되고 평가 되었다.

Alignment modeling을 위한 **anchor point**의 사용은 자연어 처리 태스크에서 많이 사용된 기법이지만 VLP 태스크에 대해서는 본 연구가 최초이다. 기존 연구들도 V+L 태스크에서 객체나 image tag를 사용하는 경우가 있었으나 이미지-텍스트의 alignment를 보다 잘 학습하는 목적이 아닌, 이미지 영역들에 대한 feature representation을 향상 시키는 것이 목적이었다.

본 연구의 주된 기여는 다음과 같다.

1. V+L understanding, generation task를 위한 generic image-text representation을 학습하는 강력한 VLP 방법인 Oscar 제안

2. 기존의 여러 V+L 벤치마크의 성능을 크게 능가

3. 여러 실험을 통해 Cross-model representation의 학습과 downstream task를 위해 object tags를 anchor point로 사용하는 것의 이점을 입증 

<br/>

## 2. Background

<br/>

학습 데이터는 이미지-텍스트 쌍으로 구성된다. 사전 학습의 목적은 이미지-텍스트 쌍의 cross-modal representation을 self-supervised 방식으로 학습하는 것이다. 학습된 표현들은 미세 조정을 통해 다양한 downstream 태스크에 적용될 수 있다. 

VLP는 일반적으로 이미지와, 텍스트 각각에 대한 고유한 embedding을 기반으로 **cross-modal contextualized representations** 을 학습하기 위해 multi-layer self-attention Transformers를 사용한다. VLP의 성능은 입력으로 input singular embeddings의 퀄리티에 의존한다.

`VLP typically employs multi-layer self-attention Transformers [39] to learn cross-modal contextualized representations, based on the singular embedding of each modality. Hence, the success of VLP fundamentally relies on the quality of the input singular embeddings.`

기존의 VLP 방법들은 이미지에 대한 visual features, 텍스트에 대한 word embeddings를 입력으로 제공하고, self attention 기법을 활용하여 이미지-텍스트의 alignments를 학습하고 cross-modal contextualized representation을 생성한다. 직관적이고 효율적이지만 이러한 방법들은 다음과 같은 문제를 내포하고 있다.

**1. Ambiguity**

Visual region features는 보통 이미지에서 오버 샘플링된 영역들에서 추출되기 때문에 서로 다른 객체들에 대한 이미지 내의 영역들이 겹칠 수 있다.

`The visual region features are usually extracted from over-sampled regions [2] via Faster R-CNN object detectors [28], which inevitably results in overlaps among image regions at diﬀerent positions.`

![image](https://user-images.githubusercontent.com/44194558/141670994-feb77d32-1563-4fff-a346-62289a12a979.png)



**2. Lack of grounding**

VLP는 이미지의 regions/object와 텍스트의 단어/문장 사이의 명시적으로 라벨링된 alignments가 존재하지 않기 때문에 기본적으로 weakly supervised learning의 문제점을 가지고 있다. 하지만 Oscar에서는 아래와 같이 명백한 (salient) 객체인 dog, couch가 이미지, 텍스트에 모두 표시되어 있기 때문에 이미지 영역과 텍스트 유닛 간의 semantic alignments의 학습에서 anchor point로 사용될 수 있다.

`Salient objects such as dog and couch are presented in both image and its paired text as in Fig. 2(a), and can be used as anchor points for learning semantic alignments between image regions and textual units.`

![image](https://user-images.githubusercontent.com/44194558/141670994-feb77d32-1563-4fff-a346-62289a12a979.png)

- 이미지의 dog, couch 객체가 텍스트에도 표시되어 있음

<br/>

![image](https://user-images.githubusercontent.com/44194558/141671214-62deb948-b911-40d7-a577-6a1a169d80ab.png)

- Object tags (dog, couch)가 anchor point로 사용되어 image region과 word embeddings를 대응시키고 있다.

<br/>

## 3. Oscar Pre-training

<br/>

![image](https://user-images.githubusercontent.com/44194558/141671373-0eaf4440-bc68-4fe6-81d2-fdc75dbab4d2.png)

사람들은 여러 채널들을 통해 세상을 이해한다. 이 때 몇 몇 채널들은 잡음이 있거나 불완전할 수 있지만, 중요한 요인들은 여러 채널들 사이에서 공유되기 때문에 충분히 인지 가능하다. Oscar는 이 아이디어를 기반으로 한다. 이에 따라 Oscar는 의미론적 단계에서 channel(modality) invariant한 표현들을 학습하게된다.  

`Humans perceive the world through many channels. Even though any individual channel might be incomplete or noisy, important factors are still perceivable since they tend to be shared among multiple channels. With this motivation, we propose a new VLP method Oscar to learn representations that capture channel-invariant (or modality-invariant) factors at the semantic level.`

<br/>

### Input

<br/>

Oscar는 입력인 이미지-텍스트 쌍을 word-tag-image (w, q, v)로 표현한다.

- w : text's word embeddings sequence
- q : 이미지에서 검출된 object tags의 word embedding sequence
- v : set of region vectors of the image

Oscar는 기존 VLP와 달리 이미지-텍스트 alignment 학습을 쉽게 하기 위한 anchor point q를 사용한다. 이는 이미지에서 중요한 객체가 (object tags) 텍스트에도 표현되어 있는 경우가 빈번하다는 관찰로부터 기인한 아이디어이다. 텍스에서 q와 w 사이의 alignment는 사전 학습된 BERT 모델을 사용함으로써 상대적으로 쉽게 식별할 수 있기 때문에, 텍스트에서 의미론적으로 관련이 높은 단어가 query가 될 때 object tags가 검출된 이미지의 영역들은 다른 영역들에 비해 높은 attention 가중치를 갖는 경향이 있다. 

`Since the alignments between q and w, both in text, are relatively easy to identiﬁed by using pre- trained BERT models [6], which are used as initialization for VLP in Oscar, the image regions from which the object tags are detected are likely to have higher attention weights than other regions, when queried by the semantically related words in the text.`

이 과정이 alignment를 학습하는 과정이고 아래와(b) 같이 나타낼 수 있다. 이러한 과정은 이미지 객체에 대한 보다 근본적인 표현을 학습하는데 필요한 것으로, vision space에서 애매하게 표현된 객체들을 (dog, couch) language space에서 고유한 개체로 표현할 수 있게 된다(c).

![image](https://user-images.githubusercontent.com/44194558/141673094-20cf33ff-0536-4870-b853-f2b3600266bc.png)    &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;   ![image](https://user-images.githubusercontent.com/44194558/141673250-1ff079f1-79f2-4ef0-9395-6ff4a83e2b00.png)

입력인 v와 q는 다음과 같은 과정으로 생성된다. 객체들에 대한 K개의 영역들 (보통 오버 샘플링되고 노이즈 많음)과 함께 주어진 이미지에 대해 Faster R-CNN 알고리즘을 활용하여 각 영역들에 대한 고정된 P 차원의 visual semantics를 추출한다. 


![image](https://user-images.githubusercontent.com/44194558/141673376-9768ba2d-17bd-4d89-8aba-a0eff81329ad.png)

- v' : region feature with P-dim (P=2048)
- z : region position with R-dim (R= 4 or 6)

v'와 z를 결합하여 position-sensitive region feature vector를 생성하고, 해당 객체에 대한 텍스트의 word embedding과 동일한 차원을 가질 수 있도록 linear projection을 적용시켜 v를 생성한다. 동일한 Faster R-CNN이 object tags를 검출하는데 사용되고, q는 object tags에 대한 word embedding sequence이다.

<br/>

### Pre-Training Objective

<br/>

Oscar의 입력은 다음과 같이 2 가지 관점으로 이해할 수 있고, 2개의 뷰를 사용하여 사전 학습의 objective를 설정할 수 있다.


![image](https://user-images.githubusercontent.com/44194558/141673615-87732331-54ef-4ee9-8989-c016a1fa6ba3.png)

- x : 텍스트와 이미지 간의 representation을 구별하기 위한 modality view
- x' : 2개의 다른 semantic space를 구별하기 위한 dictionary view


**A Dictionary View : Masked Token Loss (MTL)**

Object tags와 단어 토큰은 같은 linguistic semantic space를 공유하지만 image region features는 visual semantic space에 놓여있다. 본 연구는 이산 token sequence 
![image](https://user-images.githubusercontent.com/44194558/141675588-d5d603a6-7e2e-4bab-84b9-33f156e866f7.png) 를 정의하고 사전 학습을 위해 MTL을 적용한다. 매 iteration마다 h의 입력 토큰의 15%를 랜덤하게 마스킹하여 [MASK] 토큰으로 대체한다. 훈련의 목표는 negative log likelihood를 최소화하는 것으로 주변 토큰 h_i와 모든 image features v를 기반으로 마스킹된 토큰을 예측하는 것이다.

![image](https://user-images.githubusercontent.com/44194558/141675684-69512acc-7711-4b78-92b1-f82499a90e10.png)

이는 BERT의 Masked Language Model과 유사하며, 마스킹된 단어/태그는 주변 맥락과 추가적인 이미지 정보를 고려하여 복원되어야 한다.  

> 보충 : dictionary view에 대해
> 
> `A semantic space can be viewed a vector space deﬁned by a dictionary, which maps an input to a vector representation in the semantic space. For example, BERT can be viewed as a dictionary that deﬁnes a linguistic semantic space. BERT maps an input word or word sequence into a feature vector in the semantic space.`

<br/>

**A Modality View : Contrastive Loss**

이미지의 modality를 ![image](https://user-images.githubusercontent.com/44194558/141675827-a304081c-e7f4-495d-b6a5-e877d4ec3b69.png)로 표현하고, w는 언어 modality로 간주한다. q의 각 object tag의 50%를 데이터셋 D에서 랜덤하게 샘플링한 다른 tag sequence로 대체하여, `polluted image representations`를 샘플링한다. 특수 토큰인 [CLS]의 encoder output이 (h', w)에 대한 fused vision-language representation이기 때문에, 이 위에 FC를 쌓아 이미지-텍스트 쌍이 원본 이미지의 representation을 포함하는 지(1), 오염된 representation을 포함하는 지 예측하는 분류기를 생성한다.


![image](https://user-images.githubusercontent.com/44194558/141676155-5817d837-44d1-4d53-bdb5-673203abf43d.png)

Cross-modal 사전 학습 과정에서 object tags를 이미지에 대한 프록시로 사용함으로써 BERT의 word embedding space를 조정한다. Oscar의 pre-training objective는 다음과 같다.

![image](https://user-images.githubusercontent.com/44194558/141677039-47d8fbdf-6baf-4124-910a-bcbf4103b90a.png)

<br/>

### Pre-training Corpus

![image](https://user-images.githubusercontent.com/44194558/141677251-074592e2-3511-415b-9dce-56a994744ce6.png)

<br/>

### Imprementation Details

BERT base, large의 파라미터로 초기화된 Oscar_B (H=768), Oscar_L (H=1024)를 사전 학습 시켰다. Image region features가 BERT와 동일한 입력 embedding 크기를 가질 수 있도록 행렬 W를 사용하여 linear projection을 수행했다. 즉 훈련 가능한 파라미터는 BERT기반의 파라미터와 W이다. 옵티마이저로 Adam W를 사용했고, 이산 토큰인 h와 region features v의 시퀀스 길이는 각각 35, 50이다.

<br/>

## 4. Adapting to V+L Tasks

<br/>

사전 학습된 모델을 7개 유형의 (5 for understanding 2 for generation) downstream V+L 태스크에 적용시켰다. 

**Image-Text Retrieval**

이미지와 텍스트의 joint representation과 큰 연관성이 있다. 서브 태스크로 image retrieval, text retrieval가 있다. Aligned된 이미지-텍스트 쌍에 대해, unaligned된 이미지-텍스트 쌍을 만들기 위해 랜덤하게 샘플링한다. [CLS] 토큰이 최종 representation이 이진 분류기에 입력으로 제공되어 주어진 쌍이 aligned인지 unaligend인지 판별한다. 실험 결과 이진 분류 손실을 사용하는 것이 보다 나은 성능을 보였다.

**Image Captioning**

주어진 이미지에 대해 적절한 텍스트 description을 생성하는 것이 목적이다. 문장의 생성을 위해 seq2seq objective를 이용하여 Oscar를 미세 조정한다. 입력 샘플들은 triples : (image region features, 이미지 캡션, object tags)로 변환되어 사전 훈련된다. 캡션 토큰의 15%를 랜덤하게 마스킹하고 마스킹된 토큰을 잘 예측하도록 학습을 진행한다.  VLP와 유사하게 캡션 토큰이 이전 시점들의 토큰들만 고려할 수 있도록 self-attention mask가 사용된다 (uni-directional generation process). 모든 캡션 토큰들은 image regions에 대해서는 full attention을 적용한다. 

Inference 과정에서는 image regions, object tags, [CLS] 특수 토큰을 입력으로 제공하여 인코딩하고, 모델은 [MASK] 토큰에 가능도를 고려하여 생성할 단어들을 feeding하는 것으로 문장 생성을 시작한다. 현재 시점의 입력 시퀀스의 [MASK] 토큰은 샘플링된 단어로 대체되고, 다음 시점의 입력 시퀀스의 [MASK] 토큰에서 또 다시 단어를 예측하게 된다. 이러한 과정이 [STOP] 토큰을 만날 때까지 반복되게 된다.

**VQA**

이미지를 기반으로 자연어 질의에 모델이 적절하게 응답하는 것을 목표로한다. 이미지와 질문이 주어졌을 때 multi choice list에서 알맞은 정답을 선택하는 태스크이다. 본 연구에서는 MS COCO 데이터셋을 사용하여 실험을 진행하였다. 개별 질문에 대해 모델은 질문에 상응하는 응답을 3129개의 답변들로 구성된 shared set에서 선택하게 된다.

VQA 태스크에 대해 미세 조정을 수행할 때 (주어진 질문, object tags, image region features)를 concatenate한 하나의 입력 시퀀스를 구성하게 된다. [CLS] 토큰에 대한 Oscar의 출력이 task specific한 선형 분류기에 입력으로 제공되어 적절한 답변을 예측하게 된다. 본 연구는 VQA를 multi-label 분류 문제로 취급하여 cross entropy loss를 최소화 하도록 모델을 미세 조정한다 : 실제 정답 답변과의 연관성을 고려하여 각 답변 후보들에 대해 soft target score를 부여하고, 예측된 score와 soft target scores를 비교하여 cross entropy loss를 계산한다. Inference시에는 예측을 위해 softmax 함수만을 사용한다.

<br/>

## 5. Experimental Results & Analysis

7개의 downstream task에 대해 SOTA 성능을 보인다.

![image](https://user-images.githubusercontent.com/44194558/141678388-c88a585d-313b-47c6-8983-6468341f74d5.png)

![image](https://user-images.githubusercontent.com/44194558/141678410-4f175621-6941-4cf0-ab4e-806c1d065df6.png)

![image](https://user-images.githubusercontent.com/44194558/141678420-61ccc008-25f6-4c32-8233-7038d4878ef1.png)

<br/>

Object tags 유무에 따른 성능 비교

![image](https://user-images.githubusercontent.com/44194558/141678597-5d3d1ccc-d164-48f0-a1d4-6e5aacb177f9.png)

Object tags를 사용하는 경우 동일한 객체에 대한 텍스트, 이미지 features간의 거리가 매우 가깝고 overlap된 (perfectly aligned) 케이스도 있다.

![image](https://user-images.githubusercontent.com/44194558/141678719-6af1fe0b-bec2-4f19-b192-b9db8c20cf33.png)

<br/>

Tags를 사용하지 않는 경우 몇 몇 객체에 대한 텍스트, 이미지 features가 크게 분리되어 있는 것을 확인할 수 있다. 또한 서로 다른 객체에 대한 이미지 features의 거리가 너무 가까운 경우도 존재한다.

![image](https://user-images.githubusercontent.com/44194558/141678737-102c83b3-a50b-445c-aeed-cf56b2b3a8ee.png)


<br/>

## 6. Related Work

<br/>

**Vision-Language Pre-training**

다양한 V+L 문제들을 해결하기 위한 일반적인 모델을 사전 학습 시키는 것에 대한 관심이 증대했다. 기존의 방법들은 BERT-like objectives를 사용하여 visual region features, language token embeddings를 결합한 concatenated sequence로부터 cross-modal representation을 학습하는 것을 목표로했다. 이러한 방법들은 이미지와 텍스트 두 modality 모두에 대해 잘 conexualized된 joint representation을 학습하는데 있어 self-attention 기법에 크게 의존하는 경향이 있다.

본 연구는 기존의 방법들과 달리 두 modality의 요소들을 aligning 시키는데 있어 object tags를 사용한다. 그리고 이러한 기법이 VLP 모델이 cross-modal semantic alignment를 학습하는 것을 개선시킨다.

`Compared to existing VLP methods, the most salient diﬀerence of the proposed Oscar is the use of object tags for aligning elements in two modalities. It alleviates the challenge of VLP models having to ﬁgure out the cross-modal semantic alignment from scratch, and thus improves the learning eﬃciency.`

<br/>

**Object Tags**

본 연구는 사전 학습된 linguistic semantic space에서 object-region features를 align하기 위해 object tags를 사용한다. 본 연구에서 object tags의 구성은 (region features, word embeddings와 함께) linguistic entity embeddings가 사전 학습되었을 때 객체에 대해 보다 완전하고 유익한 representation을 제공할 수 있다.

`Our construction of object tags with their corresponding region features & word embeddings yields more complete and informative representations for objects, particularly when the linguistic entity embeddings are pre-trained, as described next.`

<br/>

**Multimodal Embeddings**

V+L 태스크에서 이미지와 텍스트 간의 shared embedding space는 많은 이점을 갖는다. 

























