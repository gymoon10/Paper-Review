# Pixel-Adaptive Convolutional Neural Networks

<br/>

## Abstract

CNN에서 가중치들이 입력 이미지, feature map의 모든 위치에서 spatially share된다는 점은 범용적으로 활용가능하다는 장점이 있지만, 그렇기 때문에 contetn-agnostic하다는 한계가 있다.

본 연구는 CNN에 약간의 변형을 가해 filter의 가중치들을 공간적으로 상이한(<->spatially shared) kernel과 곱 연산을 수행하는 PAC를 제안한다. 해당 kernel은 학습 가능한 local pixel feature를 바탕으로 학습된다. 

`We propose a pixel-adaptive convolution (PAC) operation, a simple yet effective modiﬁcation of standard convolutions, in which the ﬁlter weights are multiplied with a spatially varying kernel that depends on learnable, lo- cal pixel features.`

PAC는 기존의 여러 filtering teqchnique들의 보다 일반화된 버전이기 때문에 범용적으로 활용이 가능하며, 사전 학습된 네트워크에서도 기존의 conv layer를 대체할 수 있다.

<br/>

## 1. Introduction

전형적인 CNN은 전체 입력에 대해 필터의 모든 가중치를 공유한다. 그렇기 때문에 spatially(translation) invariance 특성을 가지고, 파라미터의 수를 획기적으로 줄일 수 있다는 장점이 존재한다. 하지만 이러한 spatially invariant convolution은 단점이 있는데, 특히 semantic segmentation같은 dense pixel prediction 태스크에서 특히 그러하다.

### Spatial Sharing

Dense pixel prediction task에서 pixel grid마다 서로 다른 객체와 요소들이 존재하기 때문에 loss 역시 공간마다 다른(spatially varying) 특성이 있다. 따라서 각 픽셀마다 파라미터에 대한 최적의 gradient 방향 역시 크게 달라질 수 밖에 없다. 하지만 기존의 CNN의 spatial sharing 특성에 의해 모든 이미지 위치에 대한 gradient loss는 filter를 학습시키기 위해 globally pool되게 된다. 이로 인해 CNN은 모든 픽셀 위치에서 에러를 한 번씩 최소화하는 filter를 학습하게 되고, 이는 특정 위치에서는 최적이 아닐 수 있다. 

### Content Agnostic

`Ideally, we would like CNN ﬁlters to be adaptive to the type of image content, which is not the case with standard CNNs.`

CNN이 학습되고 나면, 동일한 conv filter가 이미지의 contents와 상관없이 모든 픽셀에 동일하게 적용되게 된다. 이미지 content는 동일한 이미지 내에서도 픽셀마다 크게 다르기 때문에, 기존의 CNN은 모든 타입의 이미지와 서로 다른 픽셀에는 최적이 아닐 수 있다.

이러한 문제는 많은 수의 가중치 filter를 도입하여 이미지와 pixel variationd을 보다 잘 파악함으로써 어느 정도 해결할 수 있으나, 파라미터의 수가 증가하기 때문에 이상적인 방식이 아니다. 

이에 대한 대안으로는 네트워크에 content-adaptive filter를 도입하는 방식이 있다. 기존의 bilateral filter, guided image filter등은 CNN의 출력을 강화하는 측면이 있으나 표준 CNN을 완전히 대체할 수는 없다. 다른 방식으로는 독립적인 별개의 sub-network를 사용하여 매 pixel마다 conv filter의 가중치를 예측함으로써, 최종적으로 position-specific kernel을 학습하는 Dynamic Filter Networks 방식이 있다.

이 DFN 방식은 기존의 CNN을 대체할 수 있을 정도의 훌륭한 아이디어이지만, 그러한 kernel prediction 자체가 매우 복잡하고 전체 네트워크에 대한 확장성이 부족하다는 단점이 있다. 따라서 본 연구는 기존의 content adaptive layer의 한계를 극복하면서도, 전통적인 CNN의 spatailly invariant 특성을 최대한 활용할 수 있는 새로운 방식 PAC를 제안한다.

![image](https://user-images.githubusercontent.com/44194558/148719313-ffb4ab87-f50c-4e5c-ba3f-d21d4eacce0f.png)

 - 기존의 spatially invariant conv filter W의 각 픽셀에 spatially varing filter K (adapting kernel)를 곱한다
 - K는 사전 정의된 형식을 가지며 (ex. Gaussian ), pixel feature에 의존한다
 - Pixel feature f (adapting feature)는 pixel의 위치나 colr feature 등을 활용하여 사전 정의될 수도 있고, cNN을 통해 학습될 수도 있다.

`As a result of its simplicity and being a generalization of several widely used ﬁltering techniques, PAC can be useful in a wide range of computer vision problems.`

<br/>

## 2. Related Work

### Image-adaptive filtering

`A common line of research is to make these ﬁlters differentiable and use them as content-adaptive CNN layers.`   

본 연구와 유사한 기법은 기존의 2d CNN을 고차원의 conv로 일반화하여 content adaptive하게 만드는 네트워크인데, 개념상으로는 고차원으로의 mapping 덕분에 PAC보단 일반화 가능성이 높다고 할 수 있지만 실제로는 고차원의 네트워크는 adapting feature를 학습하기 어렵고, 계산 비용 역시 많이 소요된다.

<br/>

### Dynamic fiilter networks

Filter 가중치를 독립적인 network branch를 사용하여 직접적으로 예측하여, 입력 데이터에 specific한 custom filter를 제공. 하지만 이미지의 모든 위치에 대해 position-specific한 filter 가중치를 예측하는 것은 많은 수의 파라미터를 필요로하기 때문에, DFN은 정교한 아키텍쳐가 요구되며 확장성이 떨어진다.

이와 달리 PAC는 기존의 CNN처럼 spatial filter를 재사용하며, filter만 position-specific한 방식으로 수정한다.

<br/>

### Self-attention mechanism

PAC는 개별 픽셀 위치에서 filter에 대한 반응을 (receptive) 계산하면서, global context를 반영하는 self-attention과도 유사한 측면이 있다. Attention 없이 전체 이미지를 처리하는 것은 계산 비용이 크기 때문에 저차원의 feature map을 사용할 수 밖에 없다.

PAC layer는 local context에 보다 민감한 반응을 계산하기 때문에 효율적이다.

<br/>

## 3. Pixel Adaptive Convolution

다음은 기존의 standard spatial convolution

![image](https://user-images.githubusercontent.com/44194558/148730801-8be39c8e-e2e4-4d59-811f-e6db2cf6dc47.png)

 - Filter 가중치 W가 픽셀의 위치에만 의존하기 때문에 content agnostic
 - W를 content adaptive하게 만드는 것이 본 연구의 목적

<br/>

![image](https://user-images.githubusercontent.com/44194558/148731087-a7dbcd5a-6598-4e59-9424-890c0a6d4b25.png)

 - W는 고차원의 filter operation (d차원의 feature space에서)
 - `freely paremerize and learn W in high-dimensional space`

따라서 입력 signal(이미지) v를 d 차원의 공간인 pixel features f로 mapping하고, W를 통해 d 차원의 conv 연산을 수행하는 방식을 생각해볼 수 있으나. 고차원으로의 mapping은 sparse하고, 기존 CNN의 spatial sharing의 장점이 사라지기 때문에 바람직하지 않다.

<br/>

위와 같은 이유로 본 연구에서는 고차원의 conv연산을 수행하기 보다, Eq.1의 spatially invariant convolution을 직접 수정하는 방식을 택한다.

![image](https://user-images.githubusercontent.com/44194558/148731770-ccc6ac50-d6f4-46d8-87d2-84a2f4ef610a.png)

 - K : spatially varying kernel, adapting kernel (pre-defined되어 있기 때문에 non-parametric)
 - K를 pixel feature에 직접 적용함으로써 고차원으로의 mapping을 방지할 수 있음
 - f : pixel features (adapting features)
    - hand crafted feature, end-to-end로 학습되는 deep feature (learned pixel embeddings) 모두 가능 

`We can perform this ﬁltering on the 2D grid itself without moving onto higher dimensions. We call the above ﬁltering operation (Eq.3) as “Pixel-Adaptive Convolution” (PAC) because the standard spatial convolution W is adapted at each pixel using pixel features f via kernel K.` 

<br/>

PAC는 spatial conv, bilateral filter, pooling등 다양한 filter 연산의 일반화된 버전. Eq.2의 고차원 mapping이나 DFN이 PAC보다 우수한 일반화 능력을 보일 수는 있으나, PAC는 spatially invariant filter를 재사용한다는 측면에서 보다 효율적이다 (Good trade off btw standard conv and DFNs).

`In DFNs, ﬁlters are solely generated by an auxiliary network and different auxiliary networks or layers are required to predict kernels for different dynamic-ﬁlter layers. PAC, on the other hand, uses learned pixel embeddings f as adapting features, which can be reused across several different PAC layers in a net- work.`

또한 Eq.2와 비교했을 때 PAC는 고차원의 filter를 spatial filter W와 adapting kernel f의 product로 분해(factorize)하는 방식으로 볼 수 있다.

Learneable deep features f를 adapting feature로 재사용 함으로써 f에 대한 역전파 가능,

<br/>

## 4. Deep Joint Up-sampling Networks

`Joint upsampling is the task of upsampling a low-resolution signal with the help of a corresponding high-resolution guidance image.` (ex 저해상도의 depth map을 이에 대응하는 고해상도 RGB 이미지를 guide로 사용하여 고해상도로 upsampling)

PAC는 adapting featue f를 이용하여 filter operation을 guide하는 것으로 이해할 수 있고, deep joint upsampling에 유용하게 사용됨.

![image](https://user-images.githubusercontent.com/44194558/148734232-b8956ac0-00e2-42f9-84b0-74d999e5a0d9.png)

 - Encoder : 저해상도의 신호에 직접적으로 conv 적용
 - Guidance : Guidance 이미지에 conv를 적용하여 adapting feature f를 생성. 생성된 feature들은 이후 모든 PAC layer에서 재사용됨.
 - Decoder : PAC layer들의 sequence로 시작하여 pixel adaptive conv 적용. 이후 최종 output 출력.

![image](https://user-images.githubusercontent.com/44194558/148734797-091b66fc-704d-407f-830e-562ac39a44ae.png)



