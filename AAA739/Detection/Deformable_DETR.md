# Deformable DETR: Deformable Transformers for End-to-End Object Detection

<br/>

참고 : https://www.youtube.com/watch?v=YuFqxh3XvY0


<br/>

## Abstract

DETR은 NMS, anchor같은 hand-crafted component들을 제거하여 detection process를 간소화 하였지만, 느린 수렴과 feature map의 이용가능한 spatial resolution이 제한적 (FPN처럼 high resolution feature map의 장점을 활용하지 못함)이라는 단점이 있다.

`To mitigate these issues, we proposed Deformable DETR, whose attention modules only attend to a small set of key sampling points around a reference.`

![image](https://user-images.githubusercontent.com/44194558/157381199-86ac7ec9-749d-4fbc-a882-d7af40e0e538.png)


<br/>

## 1. Introduction

DETR은 detection pipeline을 간소화하고, 최초의 'fully' end-to-end object detector를 구축했다는 의의가 있지만, 다음과 같은 이슈들이 존재한다.

 1. 학습에 필요한 epoch가 굉장히 많음
    
    - 학습 초기 단계에 attention weight는 모든 픽셀에 대해 uniform 분포를 따름 (학습을 통해 갱신되어야 함)

    - `Long training epoches is necessary for the attention weights to be learned to focus on sparse meaningful locations.`  

 2. 작은 object 검출 능력이 떨어짐

    - 대다수의 detector들은 multi-scale feature map을 이용하고, high-resolution feature map에서 작은 객체들을 검출하지만 DETR은 high-resolution(low level) feature를 이용하면 큰 성능 저하가 발생한다.
    
    - Attention 연산은 pixel 개수에 대해 n^2의 계산 복잡도를 가지기 때문에 high-resolution feature map을 이용하는 것은 계산과 메모리 측면에서 굉장히 비효율적 

<br/>

**Deformable Conv**
    
`In the image domain, deformable convolution (Dai et al., 2017) is of a powerful and efﬁcient mechanism to attend to sparse spatial locations. While it lacks the element relation modeling mechanism, which is the key for the success of DETR.`

기존의 CNN 모델들은 고정된 구조만을 사용하는 단점이 있음 (ex. 3x3 filter는 3x3 사각형 안에서만 특징을 추출함). 이러한 translation invariance는 dense prediction인 segmentation이나 detection 성능에 부정적인 영향을 끼칠 수 있음.

DCN은 고정된 수용 영역안에서만 특징을 추출하는 것이 아닌, 좀더 flexible한 영역에서 특징을 추출하는 방법을 제안함. 규격화된 grid내의 값을 샘플링하는 것이 아닌, 보다 광범위한 범위의 grid cell의 값들을 샘플링함.

![image](https://user-images.githubusercontent.com/44194558/157219795-57680da6-0007-495d-8bfb-d622b07d464a.png)



<br/>

Conv filter의 kernel 위치를 학습시켜 보다 넓은 receptive field를 갖게 함. 기준이 되는 픽셀에 대해 offset을 FC layer로 학습시켜 해당 offset에 대한 conv 연산을 수행. 기준 픽셀과 관련이 있는 offset 픽셀을 학습하여, 큰 객체의 경우 큰 offset을, 작은 객체의 경우 작은 offset을 학습함.

Deformable DETR은 encoder내 attention 연산의 입력이 되는 key를 offset point로 사용 (default로 4개 사용). 


![image](https://user-images.githubusercontent.com/44194558/157219839-510a8b73-e50c-46e0-b0b0-4b1585a9377d.png)

- kernel 부분의 feature (초록색 사각형)에 conv 연산을 적용하여, feature에 포함되는 pixel들에 대한 offset을 계산

- 계산된 offset을 바탕으로 pixel들을 샘플링 (파란색 사각형들)



<br/>

`In this paper, we propose Deformable DETR, which mitigates the slow convergence and high complexity issues of DETR. It combines the best of the sparse spatial sampling of deformable convolution, and the relation modeling capability of Transformers. `

Transformer의 주요 문제는 가능한 모든 spatial locations (all pixels)를 고려한다는 점이기 때문에, Deformable attention은 reference point (query 좌표) 주변의 key sampling point들만 고려한다.
 
 - `We propose the deformable attention module, which attends to a small set of sampling locations as a pre-ﬁlter for prominent key elements out of all the feature map pixels.`
 
 - 학습 가능한 query feature에 linear proj를 수행한 후, 학습 가능한 파라미터로 offset point를 정의. 학습된 offset이 attention 연산의 key, value가 됨.

<br/>

## 2. Related Work

### Efficient Attention

`One of the most well-known concern of Transformers is the high time and memory complexity at vast key element numbers, which hinders model scalability in many cases.`

 - 단순히 attention 연산의 범위를 local neighborhood로 제안하는 방식은 자칫 global information의 상실로 이어질 수 있음
 
적어도 image recognition 단계에서는 deformable conv가 Transformer self-attention보다 효과적이고 효율적임.

`Our proposed deformable attention module is inspired by deformable convolution, and belongs to the data-dependent sparse attention category. It only focuses on a small ﬁxed set of sampling points predicted from the feature of query elements.`

<br/>

### Multi-scale Feature Representation for Object Detection

다양한 크기의 객체를 detection하기 위해서는 multi-scale feature를 이용하는 것이 중요함. FPN이 대표적인 예시.

`Our proposed multi-scale deformable attention module can naturally aggregate multi-scale feature maps via attention mechanism, without the help of these feature pyramid networks.`

<br/>

## 3. Revisiting Transformers and DETR

### MHA in Transformer

`Long training schedules are required so that the attention weights can focus on speciﬁc keys. In the image domain, where the key elements are usually of image pixels, # of keys can be very large and the convergence is tedious.`

 - Attention 파라미터 초기화의 경우 uniform dist를 따르는데, Key의 개수가 많으면 input feature에 대해 유의미한 gradient를 전달하기 어려워 epoch가 많이 필요

`Multi-head attention module suffers from a quadratic complexity growth with the feature map size.`

<br/>

### DETR

`DETR (Carion et al., 2020) is built upon the Transformer encoder-decoder architecture, combined with a set-based Hungarian loss that forces unique predictions for each ground-truth bounding box via bipartite matching.`

DETR의 단점은

 1. Image feature map을 attention 연산의 key로 사용하는 것의 부적절함 (high resolution feature를 사용하지 못해 작은 객체 탐지가 어려움)
 
 2. 초기 단계에 attention 가중치는 전체 feature map에 대해 uniform 분포를 따르고, 학습이 끝난 attention map은 굉장히 sparse하다 (object에 집중하기 때문에).

<br/>

## 4. Method

### Deformable Attention Module


Attention 연산을 적용할 때 모든 픽셀이 key가 되는 것이 아니고, 특정 layer를 통해 예측된 sampling point들에 대해서만 attention 연산을 수행 (=Deformable Conv)

 - `Deformable attention module only attends to a small set of key sampling points around a reference point, regardless of the spatial size of the feature maps.`

![image](https://user-images.githubusercontent.com/44194558/157224851-d9d1b60e-69df-469b-9362-b8bc5bbde3ca.png)

 - x : 입력 feature map
 - q, z_q : query element / query element의 content feature
 - p_q : query element의 reference point
 - m : attention head idx
 - k : sampling point idx (deformable)
 - delta_p_mqk : reference point에 더해줄 offset (offset을 더해준 게 sampling point)


<br/>

![image](https://user-images.githubusercontent.com/44194558/157225242-91f2b4cd-e411-45be-82aa-7139a72948aa.png)

 - Reference point : 기준점 (query가 곧 reference point가 됨). 이 기준점을 중심으로 offset이 계산됨

 - query feature z_q에 linear proj를 적용 (D -> 2MK). M개의 head에 대해 k개의 offset (delta x, delta y)을 나타낼 수 있도록 - (M, K, 2) M개의 각 head에 대해 K개의 point 샘플링.
 
 - Input feature map x에 대해 linear proj를 적용시켜 M개의 head를 만듬 (Attention 연산의 value) 
 
     - 위의 K개의 sampling point (sampling offsets) 고려
 
 - query feature에 linear proj를 적용하여 (D -> MK), q와 k사이의 attention score를 계산
     
     - 기존 attention과 다르게 K개만 고려하기 때문에 score가 크고 분명하게 나오는 장점  

* 특이하게 내적이 아닌 linear layer로 attention weight를 구하지만 성능상의 큰 차이가 없었다고 함

<br/>

### Multi-scale Deformable Attention Module

여러 stage의 feature map 고려. 여러 layer를 고려한다는 점 외에 위의 수식과 큰 차이 x

![image](https://user-images.githubusercontent.com/44194558/157228234-8e5263b9-f882-44a6-8403-87d5292f814d.png)

![image](https://user-images.githubusercontent.com/44194558/157227620-6fd72b0b-adcf-4a32-a860-33a7f72db001.png)

 - Attention 연산을 통해 multi-scale feature map들을 참고하되, 샘플링된 포인트들에 대해서만 연산 수행

`Note that the top-down structure in FPN (Lin et al., 2017a) is not used, because our proposed multi-scale deformable attention in itself can exchange information among multi-scale feature maps.`

Multi-scale에서도 query, key는 feature map들의 픽셀들이다. 개별 query pixel마다 고유의 referencfe point가 존재함. 어떤 feature map에 어떤 query pixel이 있는 지 식별하기 위해 학습 가능한 scale-level embedding이 수행됨.

<br/>

### Encoder 

Encoder는 backbone에서 여러 stage의 feature map들을 입력으로 받음. 각 feature map들은 1x1 conv를 통해 동일한 dim으로 변환됨

각 feature map의 픽셀들이 곧 reference point가 되며, 독립적인 linear layer에 의해 계산된 sampling offset, attention weights를 이용하여 multi-scale deformable attention 수행

![image](https://user-images.githubusercontent.com/44194558/157382404-48778060-fa73-4fdb-8eed-4f14ea5988be.png)

 - 균일하게 모든 지점이 reference point가 됨
 
 - linear layer를 통해 sampling point 추출


### Decoder

기존 DETR과 동일하게 object query들을 self attention layer에 통과시켜 서로 간의 상호 작용을 학습 (DETR과 동일한 구조)

Decoder는 self-attention, cross attention으로 나누어져 있음.

 - `In the cross- attention modules, object queries extract features from the feature maps, where the key elements are of the output feature maps from the encoder.`
 
![image](https://user-images.githubusercontent.com/44194558/157382236-0915d8a7-508e-4815-88b5-048f267be61c.png)

 - `In the self-attention modules, object queries interact with each other, where the key elements are of the object queries.`

Decoder는 object query embedding에 대해 linear proj, sigmoid를 통해 reference point를 예측하게 됨 (initial guess of box center). Detection head가 relative offset계산.

 - `Multi-scale deformable attention module extracts image features around the reference point, we let the detection head predict the bounding box as relative offsets w.r.t. the reference point to further reduce the optimization difﬁculty.`


