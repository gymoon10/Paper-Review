# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

<br/>

참고

https://herbwood.tistory.com/10

https://herbwood.tistory.com/11

https://yeomko.tistory.com/17

<br/>

## Abstract

Fully conv layer로 구성된 네트워크인 Region Proposal Network를 제안하고, 이를 기존의 detection 모델인 Fast R-CNN과 결합시켜 ent to end로 학습이 가능한 single(unified) network를 구성. RPN은 일종의 attention 메커니즘으로 detection network에 '어디에 집중할 지 (where to look)'에 대한 정보를 제공한다. 

 - `In this work, we introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals`

기존의 Fast R-CNN의 구조를 계승하면서도 selective search를 RPN을 통해 ROI를 계산. 이를 통해 GPU를 통한 연산이 가능해짐.

<br/>

## 1. Introduction

`In this paper, we show that an algorithmic change - computing proposals with a deep convolutional neural network.`

 - Fast R-CNN 같은 region based detector가 활용하는 feature map들을 활용하여 region proposal을 생성할 수 있다는 아이디어에 착안

 - RPN은 conv layer들을 detection network와 공유
 
 - Conv 연산의 feature map위에 다수의 conv layer로 구성된 RPN을 쌓음으로써 (입력을 일정한 간격으로 나눈) 각 grid에 대한 bbox와 objectness를 계산할 수 있음
 
 - RPN은 다양한 크기와 비율을 가진 region proposal들을 예측함 


<br/>

## 2. Faster R-CNN

`The ﬁrst module is a deep fully convolutional network that proposes regions, and the second module is the Fast R-CNN detector [2] that uses the proposed regions.`

`Single, uniﬁed network for object detection` 

![image](https://user-images.githubusercontent.com/44194558/156708417-13cb56d5-ab09-4595-8e5b-5f65cd789ec6.png)


<br/>

![image](https://user-images.githubusercontent.com/44194558/156705506-1fbe72d8-b2ab-4242-8bd2-ef08e66c0099.png)

 - Pyramid of regression references

<br/>

### RPN

`The RPN module tells the Fast R-CNN module where to look.`

`To generate region proposals, we slide a small network over the convolutional feature map output by the last shared convolutional layer.`

N x N feature map에 sliding window 방식을 사용하여, n x n 크기의 sptial window를 입력으로 활용. 각 sliding window는 inter mediate layer를 거쳐 256차원의 저차원으로 mapping. 

Mapping된 feature가 독립적인 네트워크를 거쳐 bbox와 objectness를 예측하게 됨.

![image](https://user-images.githubusercontent.com/44194558/156708053-e9145596-b726-4d1a-a91b-1ea03d472d70.png)


원본 이미지에서 region proposals를 추출. 원본 이미지에 anchor box를 생성하면 많은 수의 region proposals가 만들어지고, RPN은 conv 연산을 통해 각 region proposals에 대해 bbox와 objectness를 계산.

`Note that because the mini-network operates in a sliding-window fashion, the fully-connected layers are shared across all spatial locations.`

<br/>

![image](https://user-images.githubusercontent.com/44194558/156706393-34193487-9318-469e-a057-6388cc3b9ce1.png)

1. 원본 이미지를 pre-trained VGG에 입력하여 feature map 계산 (8x8x512)

2. feature map에 3x3 conv 연산 적용 (크기가 유지될 수 있도록 padding 적용, 8x8x512)  

3. 1x1 conv 연산을 통해 각 anchor box에 객체가 포함되어 있는 지 여부만을 판단 (channel = anchor box 9개 x object 여부 2 = 18, 8x8x2x9)  

4. 1x1 conv 연산을 통해 bbox regression 결과 계산 (channel = bbox 정보 4개 x 9, 8x8x4x9)

<br/>

![image](https://user-images.githubusercontent.com/44194558/156707121-2829471e-a090-4086-8479-1568ba0bb6df.png)

 - 원본 이미지는 800x800, sampling ratio=0.01에 의해 총 8x8개의 grid cell 생성 (각 cell은 100x100 만큼의 영역에 대한 정보 보유)
 
 - 총 64개의 grid cell마다 9개의 anchor box 존재 (total 576개의 region proposals)
 
 - RPN의 conv 연산을 통해 객체 포함 여부, bbox regression 결과 획득
 
 - class score에 따라 특정 N개 만을 추출, NMS를 통해 최적의 region proposal들 만을 Fast-R CNN에 전달   

<br/>

### Anchor box

`We introduce novel “anchor ” boxes that serve as references at multiple scales and aspect ratios.`

`At each sliding-window location, we simultaneously predict multiple region proposals, where the number of maximum possible proposals for each location is denoted as k.`

<br/>

![image](https://user-images.githubusercontent.com/44194558/156705355-64bee31d-f481-4cdb-a600-0b9c03ffec5b.png)

![image](https://user-images.githubusercontent.com/44194558/156705881-a0f9df9f-34ba-4c24-8470-2e88ad024af1.png)

 - 서로 다른 크기, 종횡비를 가지는 bounding box인 anchor box 9개 생성 (다양한 크기의 객체를 포착할 수 있도록)
 
 - 원본 이미지의 각 grid cell의 중심을 (anchor box를 생성하는 기준점, anchor) 기준으로 생성됨 

<br/>

### Translation-Invariant Anchors

`An important property of our approach is that it is translation invariant, both in terms of the anchors and the functions that compute proposals relative to the anchors.`

Transloation variant한 multi box 방식은 이미지나 객체에 변형이 가해지면, region proposal도 달라지게 되므로 이를 방지하기 위해 800개의 anchor box를 필요로 함 (dim=(4+1)x800). 이에 비해 RPN은 9개의 anchor box를 필요로 하기 때문에 model size를 대폭 감소시킬 수 있음(dim=(4+1)x9).

파라미터 수를 대폭 줄일 수 있기 때문에 과대 적합에 보다 강건하다.

<br/>

### Multi-Scale Anchors as Regression References

`Our design of anchors presents a novel scheme for addressing multiple scales (and aspect ratios).`

위의 Fig.1에서 보듯 image pyramid, filter pyramid를 사용하지 않고 pyramid of anchors 사용.

 - 다양한 크기와 종횡비를 가진 anchor box에 대해 bbox regression, classification 수행
 
 - Single scale의 이미지, feature map, filter(sliding window용)을 필요로 함 

<br/>

### Loss Function

Multi-task loss 사용. RPN은 객체의 존재 여부 만을 분류하고, Fast R-CNN은 배경을 포함한 class를 분류.

![image](https://user-images.githubusercontent.com/44194558/156710672-241aed63-bf0d-43e1-9565-297b4835b02f.png)

<br/>

## Training

### Alternate Training

RPN, Fast R-CNN을 번갈아가며 학습시키는 방식.

![image](https://user-images.githubusercontent.com/44194558/156711434-4903d54f-4165-408a-9782-31616613a529.png)

1. Anchor generation layer에서 원본 이미지에 anchor box 생성, GT box를 사용하여 RPN 학습 (pre-trained VGG 16도 학습됨). 

2. Anchor generation layer에서 생성된 anchor box, 학습된 RPN에 feature map(VGG 16 출력)을 입력으로 제공하여 region proposals 추출. 이를 활용하여 Fast R-CNN 학습 (VGG 16도 학습됨)

3. RPN에 해당하는 부분만 학습 (fine-tuning). RPN, Fast R-CNN이 공유하는 VGG 16의 conv layer는 동결   

4. 3의 과정에서 학습된 RPN으로 추출한 region proposals를 활용하여 Fast R-CNN을 학습 (fine-tuning). RPN, VGG 16은 동결.

RPN, Fast R-CNN을 번갈아 학습시키면서, 공유된 conv layer를 사용. 실제 학습 절차가 복잡하기 때문에 이후 두 네트워크를 병합하여 학습시키는 approximate joint training 방식으로 대체됨.
