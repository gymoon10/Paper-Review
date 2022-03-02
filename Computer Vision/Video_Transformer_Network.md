# Video Transformer Network

<br/>

## Abstract

3D conv에 의존하는 기존의 video action recognition에서 벗어나, 전체 입력 비디오 sequence의 정보를 반영하는 VTN을 제안한다. VTN은 그 어떤 2D spatial network위에 쌓는 것이 가능한 generic network이다.

 - `Introduce a method that classiﬁes actions by attending to the entire video sequence information.`

<br/>

## 1. Introduction

Video recognition 태스크에서 temporal dimension을 다루는 전형적인 방식은 3d conv였고, 입력 단계에서 temporal dimension을 따로 더해주어야만 했다. 이와 달리, 본 연구는 2D CNN 네트워크로부터 spatial feature representation을 학습하고, 그 이후에 attention 메커니즘을 활용하여 temporal information을 더해주는 방식을 사용하며 optical flow, multi-view inference 등의 추가적인 정보 없이 오직 RGB 비디오 프레임만을 입력으로 받는다.

 - `In contrast to other studies that add the temporal dimension straight from the input clip level, we aim to move apart from 3D net- works. We use state-of-the-art 2D architectures to learn the spatial feature representations and add the temporal information later in the data ﬂow by using attention mechanisms on top of the resulting features.`

컴퓨터 비전 분야의 Transformer에서도 매우 긴 길이를 가진 입력 시퀀스를 처리하는 것은 어려운 일이다. 특히 단순히 몇 초간의 clip을 이용하는 clip-based inference 방식은 수 분 짜리 영상의 정보조차 제대로 처리하기 힘들다는 것을 직관적으로 알 수 있다. 기존 연구들은 전체 비디오의 장기 의존성을 완벽히 처리하지 못한다는 한계가 존재한다.

VTN의 temporal processing은 Longformer 방식의 Transformer에 기반한다. 해당 방식은 수 천개의 토큰들로 구성된 긴 시퀀스를 효율적으로 처리할 수 있다.

 - `The attention mechanism proposed by the Longformer makes it feasible to go beyond short clip processing and maintain global attention, which attends to all tokens in the input sequence.`

![image](https://user-images.githubusercontent.com/44194558/156333198-5b73f7e0-e3a0-4104-843d-63604b6ef033.png)

<br/>

## 2. Related Work

### Applying Transformers on long sequences

NLP 분야에서 BERT와 RoBERTa는 다양한 태스크에서 SOTA 성능을 보인 뛰어난 모델들이지만, 긴 시퀀스의 처리에 있어 한계를 보인다. 그리고 이는 self-attention 연산이 n^2의 계산 복잡도를 가진다는 사실에 기인한다.

Longformer는 n의 계산 복잡도를 가진 attention 연산을 제안함으로써 이러한 문제를 해결한다. Longformer의 attention은 다음과 같이 구성된다.

 - Local context self-attention : sliding window에 의해 수행됨
 
 - Task specific global attention : [CLS]토큰이 전체 시퀀스의 정보를 반영할 수 있도록

다수의 windowed attention layer를 쌓음으로써 Longformer는 전체 시퀀스에 대한 정보를 통합할 수 있다.

<br/>

## Longformer

Transformer의 self attention은 연산이 입력 시퀀스 길이의 제곱에 비례하여 메모리와 계산량이 늘어난다는 단점이 있고, 이는 다양한 NLP 태스크를 수행하는 데 있어 큰 문제가 됨. BERT가 입력 시퀀스의 길이를 512로 제한한다는 사실을 고려하면, 길이가 긴 텍스트를 처리하는 데는 여전히 한계가 있다.

Longformer는 입력 시퀀스의 길이에 '선형적으로' 비례하는 계산량을 가진 다음과 같이 3 종류의 attention 연산을 (not-all pairwise) 제안함.

![image](https://user-images.githubusercontent.com/44194558/156335020-fe012720-f7e4-469c-8803-01cbbb6eaeae.png)

**Sliding window** 

 토큰 주변의 정해진 크기의 window 만큼의 토큰에만 attention 적용 (CNN과 유사). 이러한 layer를 다수 쌓으면 CNN과 유사하게 receptive field가 커지고, 포괄적인 context 정보를 가지는 representation을 생성할 수 있다.

**Dilated sliding window**

d개씩 토큰을 건너띄며 window를 듬성듬성하게(sparse) 확장하여 구성하는 방법으로, 위의 sliding window 보다 훨씬 넓은 영역에 대한 representation을 생성할 수 있음. MHA에서 각각의 head에 다른 dilation 수치를 적용하는 것이 성능 향상에 도움이 되었음 (어떤 head는 보다 지역적인 정보를, 다른 head는 보다 긴 문맥 정보를 처리).

**Global Attention**

[CLS], [SEP] 등의 special token의 경우 전체적인 문맥에 대해 attend하는 것이 유리하기 때문에, 미리 지정해둔 special token에 대해서는 global attention을 적용하여 모든 토큰들을 참고할 수 있게 한다. Downstream 태스크를 수행하는 데 있어 유리하게 작용됨.

<br/>

Longformer는 Dilated sliding window 방식을 기반으로 Autoregressive Language Modeling 셋팅에서 모델을 개발, 평가했음.

 - 낮은 layer에서는 지역적인 정보를 모델링하고, 계산의 효율성을 위해 작은 윈도우를 사용 & 즉각적으로 지역 정보를 활용하기 위해 dilate X
 
 - 높은 layer에서는 입력에 대한 high level representation을 얻을 수 있도록 큰 윈도우 사용 & 지역적인 정보를 희생하지 않으면서도 먼 거리에 있는 토큰에도 attend할 수 있도록 2개의 head에 대해 dilate 적용 

<br/>

이상적으로는 최대한 큰 윈도우 크기를 활용해 GPU에 로드할 수 있는 최대 길이의 시퀀스에 대해 훈련하는 것이 좋으나, 학습의 효율성을 위해 학습 단계에 따라 attention 윈도우 크기와 입력 시퀀스 길이를 점진적으로 증가시키는 전략을 사용.

 - 첫 번째 단계에서는 짧은 시퀀스에 대해 작은 윈도우로 학습을 시작
  
 - 이후 훈련 단계에 따라 윈도우 크기, 시퀀스 길이를 2배씩 늘리며 lr는 반감
 
 - 2048개의 토큰에서 시작하여 최종적으로는 23040개의 토큰까지 학습

<br/>

## 3. Video Transformer Network

`VTN is scalable in terms of video length during inference, and enables the processing of very long sequences.`

![image](https://user-images.githubusercontent.com/44194558/156347628-e1f01340-865f-49e8-98be-34bd359e2649.png)

 - Spatial backbone : 2D model 사용

 - Temporal attention-based encoder : 시퀀스 데이터의 global dependency를 처리하는 것이 목적.

![image](https://user-images.githubusercontent.com/44194558/156349854-14172f23-14eb-4db0-94aa-244a51771cc8.png)

<br/>


Inference 방식으로 3가지 방법을 제안함

1. 전체 비디오 입력은 end-to-end 방식으로 처리

2. 비디오 프레임들을 chunk단위로 처리 (feature들을 먼저 추출한 후, temporal attention encoder에 입력으로 제공) 

3. 미리 모든 프레임에 대한 feature를 추출한 후, temporal encoder에 입력으로 제공 

<br/>

Inference에 있어 전체 비디오를 한 번만 처리 (process the entire video at once), temporal dimension을 입력 단계에서 추가로 받지 않기 때문에 메모리와 속도 측면에서 효율적.

![image](https://user-images.githubusercontent.com/44194558/156348693-2b9c2140-78db-44de-a9ef-8ea782319584.png)

 - ` A more intuitive way is to “look” at the entire video context before deciding on the action, rather than viewing only small portions of it.`
 
 - `Allows full video processing during test time, making it more suitable for dealing with long videos.` 
