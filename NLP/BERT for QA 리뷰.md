# BERT for Question Answering on SQuAD 2.0

<br/>

## Abstract

ELMo, BERT 같은 사전 학습된 문맥 임베딩은 NLP 태스크에서 높은 성능을 보여왔다. 본 연구는 BERT 모델에 추가적인 task specific 레어를 추가하고, 미세 조정을 적용하여 Stanford Quesion Answering Dataset에 대한 성능을 개선하고자 한다. 최적의 성능을 보인 모델은 BERT base uncased 에 highway network를 추가하고, LSTM 인코더와 디코더를 구축한 모델로 77.96의 F1 스코어를 얻었다. 선택된 모델에 앙상블 기법을 적용하여 최종적으로 77.827의 F1 스코어를 획득했다.

<br/>

## 1. Introduction

본 연구는 **Reading Comprehension Question Answering** 태스크에 주목한다. 해당 태스크는 자연어에 대한 포괄적인 이해와 추론을 위한 추가적인 능력을 필요로 하기 때문에 언제나 해결하기 어려운 과제이다. 본 연구는 실제적인 문맥 독해와 가장 유사한 상황을 담고 있는 SQuAD 2.0 데이터셋을 사용하며, 해당 데이터셋을 대상으로 높은 성능을 보이는 모델은 RC, QA 문제를 해결하는 주요한 벤치마크가 될 수 있을 것이다. 본 연구는 **BERT의 linear output layer를 encoder-decoder 아키텍쳐로 변경하여, SQuAD 2.0 문제를 잘 해결할 수 있는 task specific layer를 성공적으로 수행**할 수 있었다. 

<br/>

## 2. Related work

언어 모델의 사전 학습은 많은 NLP 태스크에 있어 효과적이었고, 사전학습된 BERT representation은 추가적인 아키텍쳐를 통해 특정 태스크에 적합한 방식으로 미세조정될 수 있다. 본 연구는 BERT 모델의 linear output layer 위에 추가적인 task specific output layer들을 추가하였다. 본 연구는 SQuAD 2.0 과제에서 **BERT 모델의 성능을 향상시키기 위한 사후 처리로 encoder-decoder 아키텍쳐를 구축**하는 것을 목표로한다. 주요 encoder-decoder 아키텍쳐로 RNN 기반의 양방향 LSTM과 GRU를 채택하였고, CNN 기반의 encoder block역시 시도했다. 

순환 신경망 네트워크에서의 multi layer transition에 있어 적응적으로 representation을 변환시키 위해 highway 네트워크를 사용했다.
`we used highway network [8] to adaptively copy or transform representations.`

또한 최종 output layer에 있어, 기존의 linear layer와 QA layer를 비교했다.

<br/>

## 3. Approach

BERT 논문이 제안하는 default 방식인 추가적인 linear output layer를 추가한 BERT-Base-Uncased 모델부터 시작하였다. 추가적으로 본 연구의 방식인 새로운 task specific output 아키텍쳐를 설계하고 미세조정을 수행하였다. 또한 더 나은 성능을 위해 BERT-Large-Cased 모델과의 앙상블을 적용하였다.

<br/>

## 3.1 Pre-trained BERT Baseline Model

BERT는 트랜스포머의 양방향 encoder representation으로 양방향의 언어 모델링 태스크에 대한 학습을 통해, 사전학습된 BERT representation은 추가적인 하나의 output layer에 대한 미세 조정을 통해 다양한 NLP 태스크에 적용될 수 있다. 학습 과정에서 입력으로 주어진 토큰은 token embedding, segmentation embedding, position embedding의 합인 input embedding으로 변환되고, 양방향의 masked LM, NSP 태스크에 대해 사전학습된다. 본 연구는 input question과 paragraph를 single packed sequence(질문은 A로, 단락은 B로 segmentation embedding됨)로 제공하고, 미세조정을 위한 새로운 2개의 파라미터인 `Start vector S`, `End vector E`를 제안한다. 특정 입력 토큰에 대한 BERT의 최종 hidden vector는 `T_i`로 표기한다. T_i, S의 내적과 단락의 모든 단어에 대한 Softmax 결과를 이용하여, i 번째 단어가 answer의 시작이 될 확률을 계산할 수 있다. 학습 목적 함수는 정확한 start, end position에 대한 로그 가능도로, 질문에 대한 답이 없으면 start, end position을 모두 0으로 예측한다.

![image](https://user-images.githubusercontent.com/44194558/140260958-d8813597-d4a6-4017-b4c1-70e976ae087d.png)

<br/>

## 3.2 Modules on Top of BERT


사전 학습된 BERT-base 모델 위에 output 아키텍쳐를 추가하였다.

<br/>

### 3.2.1 Encoder and Decoder Blocks

<br/>

#### Bi-directional LSTM Layer Encoder/Decoder

LSTM은 RNN의 기울기 소실 문제를 해결하는 아키텍쳐로, 매 타임 스텝 t 마다 hidden state h_t, cell state c_t가 존재한다. cell state가 장기적인 정보를 저장하고, LSTM은 forget gate, input gate, output gate를 사용하여 cell의 정보를 통제한다. BERT 모델 위에 LSTM encoder/decoder를 추가함으로써 토큰화된 output sequence의 시간 단계 간 시간 의존성을 보다 잘 반영할 수 있다. 


<br/>

#### GRU Encoder/Decoder

LSTM에 대한 대안으로 cell state를 사용하는 대신 update gate, reset gate를 사용하여 input, hidden state를 통제한다.

<br/>

#### CNN Encoder

본 연구는 RNN unit 이후 convolution 연산을 수행하는 CNN-based encoder를 시도해 보았다. RNN 모델은 global interaction을, CNN 모델은 local interaction을 포착한다. 첫 번째 계층은 양방향 LSTM이고, 그 이후 CNN 부분에서 임베딩된 sequence에 2D convolution을 적용한다. 임베딩된 sequence 텐서 (batch size, seq_length, hidden_state)를 unsqueeze하여 (batch size, 1, seq_length, hidden_state)로 변환하고, (seq_length, hidden_state)부분에 2D convolution을 적용하여 (batch size, hidden_state, seq_length, 1) 크기의 ouput을 출력한다. 최종적으로 마지막 차원을 squeeze하고 나머지 2개의 차원을 바꿔 입력 차원과 동일하게 만든다. **Convolution은 input tensor의 차원을 변경시키지 않으면서 sequence내의 가까운 word embedding들 사이의 관계를 추출한다.**

<br/>

### 3.2.2 Self-attention Layer

특정 NLP 태스크들에 대해 CNN이나 LSTM 모델에 attention을 적용하는 sentence embedding을 위한 추가적인 정보를 추출하는 방식이 제안되어 왔고, 본 연구도 self-attention layer를 적용하여 각 위치의 output token들이 해당 위치를 포함하여 모든 위치에 대해 처리되도록 하였다. 이러한 방식은 output sequence이 서로 다른 위치들에 대한 추론에 도움을 줄 수 있다.

`This can help for better interpreting the inference between different positions in the output sequence.`

<br/>

### 3.2.3 Highway Network

`Highway network is a novel architecture that enables the optimization of networks with virtually arbitrary depth [8]. By applying a gating mechanism, a neural network can have paths along which information can ﬂow across several layers without attenuation.`

**모델을 깊게 만들면서도 정보의 흐름을 통제하고 학습 가능성을 극대화 시킴**

참고 :  https://towardsdatascience.com/review-highway-networks-gating-function-to-highway-image-classification-5a33833797b5  /  https://lazyer.tistory.com/8  /  https://lyusungwon.github.io/studies/2018/06/05/hn/  / 

![image](https://user-images.githubusercontent.com/44194558/140264195-7d3a39f9-364a-410a-bdb6-54a25ef8bdf4.png)


![image](https://user-images.githubusercontent.com/44194558/140264838-b1b353e2-9bf1-4d16-9701-aadf8f6f4031.png)

![image](https://user-images.githubusercontent.com/44194558/140265114-02c8ef8d-4a83-4141-81b8-425d8f9b0740.png)

<br/>

### 3.2.4 Output Layer

Output layer에 대해 BERT의 linear output layer (original), Bi-directional Attention Flow (BiDAF-Out)를 각각 적용시켜 보았다.

<br/>

#### BERT Output Layer

Output sequence의 차원 (batch_size, seq_len, hidden_state)를 (batch_size, seq_len, 2) 차원으로 변환하고 쪼개어 start logit, end logit을 얻는다. start, end position의 벡터를 이용하여 cross entropy 손실을 계산한다.

<br/>

#### QA output layer from BiDAF-Out

BERT의 linear output layer를 대체한다. BERT tokenized output sequence ![image](https://user-images.githubusercontent.com/44194558/140266098-ffa0c28f-d699-4dee-9ea4-8067a69eaf48.png)에 LSTM을 적용하여 model output sequence ![image](https://user-images.githubusercontent.com/44194558/140266143-b0504526-6bff-4bda-baf3-352799498d5f.png) 로 변환한다. 이후에 양방향 LSTM을 적용하여 m'를 출력한다.

![image](https://user-images.githubusercontent.com/44194558/140266226-4234a33e-73b6-4f90-959e-d177c4c0c9f6.png)

![image](https://user-images.githubusercontent.com/44194558/140266342-2633a00b-e07d-4707-a267-0e8499d02f3b.png)

<br/>

### 3.3 Proposed Models

![image](https://user-images.githubusercontent.com/44194558/140266695-de5ee01c-6ced-4bd0-b01a-fa79e203002f.png)

<br/>


## 5. Analysis


BERT 위에 encoder-decoder 아키텍쳐를 추가하여 BERT ouput representation에 대한 후처리를 진행하는 것이 효과적이다. 

![image](https://user-images.githubusercontent.com/44194558/140267029-3c568ad7-f27a-4da1-91bf-4e332afc1d0c.png)

  * `RNN encoder-decoder architecture can help to integrate temporal dependencies  between  time-steps  of  the  output  tokenized  sequence  better,  thus  reﬁne  the  output sequence for following operations.`
 
  * BERT 위에 추가되는 task specific layer에 self attention을 추가하는 방식은 효과적이지 않을 수 있음. BERT 모델 자체가 이미 수 많은 masked attention layer에 기반하고, 특정 태스크들에 대해 사전 학습 되어 있기 때문.
  
  * CNN encoder는 제한적인 효과를 가진다. CNN encoder의 성능이 RNN encoder보다 성능이 낮아도 시간, 계산 측면에서 항상 효율적이다 (iterative한 특성이 없기 때문에). 
  
  * BiDAF output layer는 BERT output layer보다 항상 성능이 낮다. 
  
  * Highway 네트워크는 multi layer state transition에 있어 언제나 높은 성능을 보인다. (encoder, decoder 사이에 추가하거나, encoder-decoder 네트워크 이후에 추가할 때나 언제나 성능 개선이 있었음)    

BERT 위에 순환 신경망 unit, highway network와 함께 encdoer-decoder network를 추가하는 것이 BERT baseline보다 높은 성능을 보인다. **Encoder-decoder 아키텍쳐가 미세 조정 단계에서 일종의 후 처리 역할**을 하기 때문인 것으로 추정된다. 사전 학습된 BERT base가 보다 일반적이기 때문에, **Encoder-decoder 네트워크는 SQuAD 문제 해결에 있어 추가적인 정제 및 작업**을 수행한다.












