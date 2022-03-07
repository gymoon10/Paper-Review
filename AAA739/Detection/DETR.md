# End-to-End Object Detection with Transformers

<br/>

## Abstract

Object detection을 direct set prediction 문제로 바라보는 방법을 제안. 기존 object detection 방법들에 사용된 NMS, anchor 등의 수작업이 필요한 요소들을 제거함으로써 detection pipeline을 간소화했다.

본 연구가 제안하는 DETR 프레임워크는 transformer encoder-decoder구조, 양자간 매칭 (bipartite matching)을 통해 각 object에 대한 unique prediction을 가능하게 하는 set-based global loss라고 할 수 있다 (Faster R CNN의 경우 한 object에 대한 다양한 예측들이 존재하고 이를 NMS를 통해 제거하므로 not unique prediction).

학습된 object queries의 small set이 주어지면 DETR은 object와 global image context 사이의 관계를 추론하고, 최종적인 예측 set을 반환함. 

 - `Given a ﬁxed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the ﬁnal set of predictions in parallel.`

<br/>

## 1. Introduction

Object detection의 목적은 bbox, label의 set을 예측하는 것이고, 대부분의 detector들이 proposal, anchor, window center 등을 활용한 간접적인 방식으로 set prediction 태스크를 수행한다.

본 연구는 위와 같은 surrogate task 이슈를 피하고, detection pipeline의 간소화를 위한 end-to-end philosophy에 입각한 direct set prediction을 제안한다. 이를 위해 transformer encoder-decoder 구조를 도입했고, transformer의 self-attention은 sequence에 존재하는 모든 elements간의 pairwise interaction을 효과적으로 모델링한다.

<br/>

![image](https://user-images.githubusercontent.com/44194558/156988010-f0cd8c5a-1e2c-4d7d-bc8d-a86c90dcb524.png)

 - `Our DEtection TRansformer (DETR, see Figure 1) predicts all objects at once, and is trained end-to-end with a set loss function which performs bipartite matching between predicted and ground-truth objects.`

 - 기존의 detector들에 적용되어 prior knowledge를 encoding하던 다수의 hand-designed component들을 제거

<br/>

`Compared to most previous work on direct set prediction, the main features of DETR are the conjunction of the bipartite matching loss and transformers with (non-autoregressive) parallel decoding.`

 - 기존 RNN 방식의 autoregressive decoding과는 차이점이 있음.

본 연구가 제안하는 matching loss는 GT object에 unique한 예측을 할당하기 때문에, 예측된 object들 간의 순서(permutation)은 변하지 않으므로 autoregressive decoding task를 제거할 수 있음.

 - `Our matching loss function uniquely assigns a prediction to a ground truth object, and is invariant to a permutation of predicted objects, so we can emit them in parallel.`

DETR은 transformer 구조에 기반한 non-local computation이 가능하기 때문에 큰 객체들을 잘 탐지하는 경향이있다 (multi-scale feature map을 이용하지 않아 작은 객체들에 대해서는 성능이 떨어지는 단점 존재).

<br/>

## 2. Related work

<br/>

### 2.1 Set Prediction

`The basic set prediction task is multilabel classiﬁcation for which the baseline approach, one-vs-rest, does not apply to problems such as detection where there is an underlying structure between elements (i.e., near-identical boxes).`

 - Near duplicates (하나의 GT object에 다수의 예측 bbox 존재) 문제가 direct prediction을 어렵게 만듬
 
 - 이를 해결하기 위해 NMS 방식이 있으나, 본 연구의 direct set prediction은 이러한 후처리 과정이 필요 없음 
   
   - `They need global inference schemes that model interactions between all predicted elements to avoid redundancy.`

손실 함수는 모든 상황에서 예측값의 순서에 강건해야 하는데, RNN의 auto-regressive sequence model 방식은 적합하지 않다. 따라서 본 연구는 각 GT에 고유한 예측값을 할당함으로써 (bipartite matching) permutation invariance를 보장한다.

<br/>

### 2.2 Transformers and Parellel Decoding

Attention 기반 방법의 장점은 global (non-local) computation이 가능하다는 점이다. Transformer는 원래 auto-regressive model에 사용되어 decoding 출력을 하나씩, 차례대로 출력했지만, 이러한 방식은 inference cost 측면에서 비효율적이다 (proportional to output length, and hard to batch). 

이러한 문제를 해결하기 위해 본 연구는 parellel decoding 방식을 제안한다.

`We also combine transformers and parallel decoding for their suitable trade-oﬀ between computational cost and the ability to perform the global computations required for set prediction.`

<br/>

### 2.3 Object detection

`Most modern object detection methods make predictions relative to some initial guesses.`

 - proposals, anchors, grid of object centers

본 연구는 위와 같은 hand-crafted process를 제외시키고, 입력 이미지에 대해 직접 box prediction을 계산하여 (anchor 기반 예측 x) detection process를 간소화한다.

**Set-based loss**

초기 딥러닝 모델에서는 서로 다른 예측들간의 relation은 conv/fully connected layer나 NMS를 통해 모델링되었다. 본 연구는 NMS같은 hand-crafted 요소를 제거하고, 모델에 encoding되는 prior knowledge를 배제하고자 한다.

<br/>

## 3. The DETR model

객체 검출에서 direct set prediction을 가능케 하는 중요한 요소들은

 - `A set prediction loss that forces unique matching between predicted and ground truth End-to-End Object Detection with Transformers boxes.`
 
 - `An architecture that predicts (in a single pass) a set of objects and models their relation.` 

<br/>

### 3.1 Object detection set prediction loss

DETR은 고정된 크기의 N개의 예측값을 출력 (이미지에 있는 객체수 보다 많도록 넉넉하게 설정).

![image](https://user-images.githubusercontent.com/44194558/156994870-d81bb58a-5fa5-4e67-b386-4adf5546a7ff.png)

Matching cost는 class 예측, predicted boxes-GT boxes간의 유사도를 모두 고려함.

![image](https://user-images.githubusercontent.com/44194558/156995154-77c773a1-13ce-4a99-830e-5f454cac01cf.png)

이전 단계에서 매칭한 모든 pair들에 대한 hungarian loss 연산은

![image](https://user-images.githubusercontent.com/44194558/156995371-56950e18-7c62-416d-b504-980eda8b249f.png)

<br/>

**Bounding box loss**

`Unlike many detectors that do box predictions as a ∆ w.r.t. some initial guesses, we make box predictions directly.`

![image](https://user-images.githubusercontent.com/44194558/156995570-c0279c18-4b0b-4784-b096-158169989172.png)

<br/>

### 3.2 DETR architecture

메인 요소는

1. CNN backbone
 
2. encoder-decoder Transformer
 
3. Feed forward network (최종적인 detection 예측 반환)

![image](https://user-images.githubusercontent.com/44194558/156998038-afc7823e-8cce-48ac-9536-faafff34d9b8.png)

<br/>

![image](https://user-images.githubusercontent.com/44194558/157014131-7761e019-e0aa-4231-ace7-626caab5b70c.png)

<br/>

**Backbone**

입력 : (3, H, W) -> 출력 : (C, h, w) / C=2048, h=H/32, w=W/32

**Encoder**

1x1 conv를 통해 차원 축소 -> (d, h, w), d < C

Encoder는 입력으로 sequence를 받기 때문에 spatial dimension을 낮추면 -> (d, hxw)

Permutation invariacne를 위해 fixed positonal encoding을 추가

**Decoder**

`The decoder follows the standard architecture of the transformer, transforming N embeddings (learnt positional encodings, object queries) of size d using multi-headed self-and encoder-decoder attention mechanisms.`

N개의 object들을 decoding layer에서 병렬적으로 (parellel) decoding하는 것이 기존 방식들과의 차이점. 순서에 무관하기 때문에  N개의 입력 임베딩들은 서로 달라야하고, 이 input embeddings는 positional encodings를 학습하는데, 이를 object queries라고 함.

N개의 object query들은 decoder에 의해 output embedding으로 변환되고, FFN을 거치면서 '독립적으로' bbox 좌표, class label로 디코딩됨.

`Using self- and encoder-decoder attention over these embeddings, the model globally reasons about all objects together using pair-wise relations between them, while being able to use the whole image as context.`

<br/>

## 4. Experiments

`The encoder seems to separate instances already, which likely simpliﬁes object extraction and localization for the decoder.`

![image](https://user-images.githubusercontent.com/44194558/157011908-6ca0a1d8-3719-431a-bd93-30e410e4e57f.png)

 - `By using global scene reasoning, the encoder is important for disentangling objects.`
  
<br/>

`A single decoding layer of the transformer is not able to compute any cross-correlations between the output elements, and thus it is prone to making multiple predictions for the same object. `

![image](https://user-images.githubusercontent.com/44194558/157013171-5463c0fd-5bae-4925-ac63-c358b29caf17.png)

 - `We hypothesise that after the encoder has separated instances via global attention, the decoder only needs to attend to the extremities to extract the class and object boundaries.`

<br/>

FFN의 1x1 conv는 차원 축소를 통한 파라미터 감소 효과가 있음.

<br/>


![image](https://user-images.githubusercontent.com/44194558/157013616-4046cddf-6d61-4ee2-9d5a-679206b65ce1.png)

<br/>

`DETR learns diﬀerent specialization for each query slot.`

![image](https://user-images.githubusercontent.com/44194558/157013855-6b147d06-e652-4b61-b32d-1d4cd1a3f337.png)








