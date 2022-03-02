# MoCoV3: An Empirical Study of Training Self-Supervised Vision Transformers

<br/>

## Abstract

CNN 네트워크에 Self-Supervised Learning(SSL)을 적용하는 연구는 많이 이루어졌지만, Vision Transformer에는 어떻게 SSL을 적용해야 하는 지에 대해서는 아직 많은 연구가 이루어지지 않았다. 본 연구는 다양한 실험을 통해 Self-supervised ViT의 효과를 조사함.

실험을 위해 batch size, lr, optimizer등 학습에 필수적인 요소를 조사하며, SSL ViT 학습에 발생하는 instability의 문제를 발견함. 이 문제를 완화하기 위한 simple trick을 제안하며, ViT의 다양한 모델 구조 및 크기를 바탕으로 실험을 진행함.

<br/>

## 1. Introduction

`This work focuses on training Transformers with the leading self-supervised frameworks in vision. This investigation is a straightforward extension given the recent progress on Vision Transformers (ViT) [16].`

위에서 언급했듯이 instability는 매우 심각한 수준의(catastrophic) 문제까지는 아니더라도 SSL ViT학습에 있어 가장 큰 이슈가 된다 (약 1~3% 수준의 정확도 저하). 이러한 문제는 기존의 CNN 네트워크에서는 굉장히 드물게 발생하기 때문에 보다 주목할 필요성이 있다.

본 연구는 경험적인 관점에서(theoretical x) instability를 완화할 수 있는 simple trick을 제안한다.

 - `Based on an empirical observation on gradient changes, we freeze the patch projection layer in ViT, i.e., we use ﬁxed random patch projection. We empirically show that this trick alleviates the instability issue in several sce- narios and consistently increases accuracy.`

본 연구는 contrastive learning framework를 사용함으로써 SSL Transformer의 성능을 향상시킬 수 있음을 보인다 (기존의 masked auto-encoding과 비교했을 때). NLP에서 Transformer를 학습시킬 때는 masked language model을 사용한다는 점을 고려하면, 굉장한 차별점이 있음. 또한 다음 2가지 측면에서 SSL ViT가 기존의 big CNN ResNet네트워크에 비해 이점이 있음을 보인다.

 - `It(SSL ViT) achieves competitive results using relatively fewer inductive biases` 
  
 - `We observe that removing the position embedding in ViT only degrades accuracy by a small margin. This reveals that self-supervised ViT can learn strong representations without the positional inductive bias`

<br/>

## 2. Related Work

컴퓨터 비전 분야에서 contrastive learning은 SSL에서 큰 성공을 거두었다. Constrative learning의 방법론은 `learn representations that attract similar (positive) samples and dispel dif- ferent (negative) samples.`

NLP 분야에서 Transformer는 self-attention연산을 통해 문장 시퀀스의 장기 의존성을 효율적으로 학습할 수 있고, 실제로 놀라운 성과를 보였다. 이러한 Transformer를 컴퓨터 비전 태스크에 도입하려는 시도는 성공적이었다 (ViT). 점점 NLP와 conputer vision간의 architectural gap은 줄어들고 있으며, 본 연구는 컴퓨터 비전 분야에서의 SSL이 반드시 연구되어야만 하는 baseline이라고 간주한다.

컴퓨터 비전 분야에서의 SSL Transformer 학습은 NLP 분야와 유사하게 masked auto-encoding기법을 활용했지만, 본 연구는 contrastive learning의 프레임워크 하에서 Transformer의 학습에 주목한다.

 - `In this work, we focus on training Transformers in the contrastive/Siamese paradigm, in which the loss is not deﬁned for reconstructing the inputs.`

<br/>

## 3. MoCoV3

![image](https://user-images.githubusercontent.com/44194558/156308586-f0ea9f43-6123-4ac2-86a8-94fbe5fef36f.png)

1. 개별 이미지에 대해 서로 다른 random augmentation을 적용하여 2개의 crop 생성
 
2. 생성된 crop들은 query encoder, key encoder를 통과

3. Contrastive loss를 최소화하는 방향으로 학습됨 (`the goal of learning is to retrieve the corresponding key`).


![image](https://user-images.githubusercontent.com/44194558/156309058-b7ab8ec2-2227-4420-8510-bd9cf0f1d251.png)

<br/>

기존의 MoCo와 달리, batch size가 4096으로 충분히 크기 때문에 memory queue를 사용하지 않고 동일한 batch에 존재하는 입력들을 key로 사용함. 손실은 symmetrized loss 형태로 계산됨.

Query encoder는 backbone + proj MLP + pred MLP (extra prediction head)로, key encoder는 backbone + proj MLP로 구성.


![image](https://user-images.githubusercontent.com/44194558/156309420-a1ab93fe-e3b7-447c-8b0a-4c5c96bc3a45.png)

<br/>

## 4. Stabilityy of Self-Supervised ViT Training

`In principle, it is straightforward to replace a ResNet backbone with a ViT backbone in the contrastive/Siamese self-supervised frameworks. But in practice, a main challenge we have met is the instability of training.`

Instability로 인한 성능 저하가 있긴 하나 학습 결과는 'apparently good'이며, 그럴듯한 결과를 보여준다. 관점에 따라 'partially successful' 혹은 'partially failed'로 볼 수 있다. 본 연구는 이러한 문제를 해결하기 위해 simple trick을 제안하며, 다양한 케이스에서 성능 향상이 있었다.

<br/>

### 4.1 Empirical Observations on Basic Factors

### Batch size

![image](https://user-images.githubusercontent.com/44194558/156310263-1b55e219-7246-44aa-8594-dedbbcd70475.png)

batch size가 4096, 6144일 때 불안정한 학습 곡선과 이로 인한 성능 저하가 발생 (단, 학습 자체가 발산하지는 않음).

 - `We hypothesize that the training is partially restarted and jumps out of the current local optimum, then seeks a new trajectory.` 

<br/>

### Learning rate

![image](https://user-images.githubusercontent.com/44194558/156310837-89bd1c37-aa3e-4824-943d-908b1b0eb35e.png)

학습률이 작은 경우 학습이 더 안정적이지만, under-fitting이 발생하고, 학습률이 높을 때는 학습이 불안정해지고 (특히 lr=1.5e-4), 정확도도 낮아짐.

<br/>

### Optimizer

AdamW, LAMB optimizer 2개 실험.

![image](https://user-images.githubusercontent.com/44194558/156311215-cbb12122-2c56-4c50-88d9-0d1fe3da7f20.png)

LAMB이 학습률에 더 민감하기 때문에 본 연구는 AdamW를 활용하여 실험 진행.

<br/>

## 4.2 A Trick for Improving Stability

Instability는 네트워크의 비교적 얕은 layer에서 발생함.

 - `During training, we notice that a sudden change of gradients (a “spike” in Fig. 4) causes a “dip” in the training curve, which is as expected. By comparing all layers’ gradients, we observe that the gradient spikes happen earlier in the ﬁrst layer (patch projection), and are delayed by couples of iterations in the last layers (see Fig. 4).`

![image](https://user-images.githubusercontent.com/44194558/156312511-79ebb4c4-3fd7-4ab8-a0ad-3a7b9d8263a2.png)

<br/>

따라서 본 연구는 학습 도중 patch projection layer의 가중치를 동결하는 방식을 제안한다 (fixed random patch projection, not learned, stop-gradient operation).

![image](https://user-images.githubusercontent.com/44194558/156312666-29fbce9d-faac-4bce-bd76-5510c5b3ba8d.png)

Random patch projection을 사용했을 때 보다 개선된 성능을 보이며, 다른 SSL + ViT 방법에도 효과적임.

### Discussion

실험을 통해 patch projection layer의 학습이 필수적이지 않음을 보이며, 기존 patch의 정보를 보존하는데 있어 random projection으로도 충분함. 해당 layer를 동결하는 것은 architecture를 변경시키지 않으며, 최적화 문제의 solution space를 작게 만드는 효과가 있다.

단, 언제까지나 이 trick은 문제를 '완화'하는 것이지 '해결'한 것은 아니다. 이 trick을 사용해도 lr이 너무 크면 불안정성이 야기될 수 있다. 해당 trick을 사용하는 이유는 projection layer가 backbone에서 유일하게 non-Transformer layer이기에 개별적으로 처리하기 쉽기 때문이다.

<br/>

## 6. Experimental Results

<br/>

### Position embedding

![image](https://user-images.githubusercontent.com/44194558/156313691-767e1270-892a-4dc3-8a85-feaa9b216757.png)

굳이 학습시키지 않고 sin-cos방식을(no positional inductive bias) 활용해도 더 나은 성능을 보인다. 이러한 결과는 2가지 관점에서 해석이 가능하다.

 - 긍정적 : patch 집합들 만을 이용하여 충분히 유용한 (fully permutation-invariant) representation을 얻을 수 있다. 일종의 bag-of-words 모델과 유사함.
 
 - 부정적 : 현재의 모델이 위치 정보를 충분히 활용하지 못하고 있다 (gesture of the object contributes relatively little to the representation).

<br/>

### Class token

![image](https://user-images.githubusercontent.com/44194558/156314285-46dffc44-660f-45d4-ac45-61492ceea624.png)

[CLS]토큰이 반드시 필수적이진 않고, layer 정규화 방식이 성능 차이를 야기할 수 있다.

<br/>

### BatchNorm in MLP heads


![image](https://user-images.githubusercontent.com/44194558/156314561-80cfc006-d90b-4599-9619-cf68bab3fe5d.png)

BN은 contrastive learning에 있어 필수적이지 않지만, 성능 향상에 도움이 됨.

<br/>

### Prediction head


![image](https://user-images.githubusercontent.com/44194558/156314666-728e5522-a627-4c63-8653-fa4abd3ad845.png)

추가적인 MLP head가 성능 향상에 도움을 줌.

<br/>

### Momentum encoder

![image](https://user-images.githubusercontent.com/44194558/156314835-f72ed6ad-59c3-44d0-8741-4f098d712663.png)

m=0 <-> SimCLR, m=0.99 : 본 연구의 default

Momentum encoder를 사용하는 것이 좋음.

<br/>

## 6.4 Transfer Learning

`Our self-supervised ViT has better transfer learning accuracy when the model size increases from ViT-B to ViT-L, yet it gets saturated when increased to ViT-H.`

 - Small data의 경우 큰 규모의 ViT모델을 학습시킬 때 과적합이 발생했음.
 
 - 데이터가 충분치 않으면 ViT를 완전히 처음부터 학습시키는데 있어 (from scratch) inductive bias가 부족하기 때문에 유용한 representation을 학습하는데 어려움이 발생함. 

 - Self-supervised pre-training이 해결책이 될 수 있을 것