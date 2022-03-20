# Mask R-CNN

<br/>

## Abstract

Mask R-CNN은 이미지에 존재하는 객체들을 검출하는 동시에 각 객체 instance들에 대한 segmentation mask를 생성한다.

 - Object Detection + Semantic Segmentation

Faster R-CNN은 RPN을 통해 획득한 RoI에 대해 객체의 class를 예측하는 classification branch, bbox regression branch가 존재하는데, Mask R-CNN은 이러한 branch들과 병렬적으로 segmentation mask를 예측하는 mask branch를 추가한 구조이다.

Mask branch는 개별 RoI를 작은 크기의 FCN에 통과시켜 예측 mask를 획득하며, 객체의 spatial location 정보를 보존하기 위한 RoI Align layer가 추가되었다.

<br/>

## 1. Introduction

Instance segmentation은 이미지에 존재하는 모든 객체들에 대한 정확한 검출과 개별 객체 instance에 대한 정확한 segmentatino을 필요로 하기 때문에 어려운 태스크이다.

 - classify + bbox localize + segmentation


Mask R-CNN은 기본적으로 Faster-R-CNN에 기반하고 있으나, Faster R-CNN의 RoI Pooling은 pixel-to-pixel alignment에는 적합하지 않기 때문에 이러한 misalignment를 해결하고자 정확한 spatail location 정보를 보존하는 RoI Align layer가 추가되었다.

 - class, bbox prediction과 달리, mask prediction은 객체에 대한 finer spatial 정보를 필요로 함.

또한 RoI Align을 통해 class prediction, mask prediction을 각각 독립적으로 수행할 수 있게 되었다.

<br/>

## 2. Main Idea - RoI Align

<br/>

### 2.1 Get RoI from the feature map

<br/>

![image](https://user-images.githubusercontent.com/44194558/159150119-d81b704d-97da-4296-95ef-50aaeb9e821c.png)

 - 입력 이미지 크기 : 512 x 512
 - 입력 이미지에서 회색 고양이의 (오른쪽 중간) GT bbox 크기 : 145 x 200

sclae factor 32로 나누면 feature map은 16 x 16 grid, 회색 고양이의 bbox는 4.53 x 6.25.

 - 실수 입력값을 정수와 같은 이산 수치로 강제, 제한하는 **Quantization** 필요

![image](https://user-images.githubusercontent.com/44194558/159150356-ba5cde9f-d94e-4c70-ac67-8eed49eaf2b1.png)

 - 4.53 x 6.25 -> 4 x 6

<br/>

Quantization 과정에서 아래와 같은 정보 손실 (파란색) 발생

![image](https://user-images.githubusercontent.com/44194558/159150401-4dab53c9-cfdc-41dc-be25-d99badeb68ff.png)

<br/>

### 2.2 RoI Pooling

<br/>

![image](https://user-images.githubusercontent.com/44194558/159150439-a2c41e74-e13c-4b34-bfca-829d44b1d359.png)

 - Feature map에 mapping된 RoI에 pooling 적용
 
 - RoI Pooling layer 이후에 고정된 크기의 fully connected layer가 존재
 
    - RoI는 서로 다른 크기를 가지고 있기 때문에 고정된 크기로 pooling하는 과정이 필요
    
<br/>

Feature map에 mapping된 RoI의 크기를 4x6x512 (**quantized**), fully connected layer가 3x3x512를 입력으로 받는다고 가정함 (4는 3으로 나눠지지 않기 때문에 **quantization**이 필요하게 됨). 아래와 같이 Pooling을 적용하면

![image](https://user-images.githubusercontent.com/44194558/159150579-763fb49e-b9b6-410a-a531-ebdfa6a53f54.png)

 - 4x6 RoI에서 색칠되지 않은 부분의 data loss가 발생하는 것을 확인 (misalignment)

위와 같은 정보 손실이 dense prediction 태스크에서 정확한 pixel mask를 예측하는데 문제가 됨. 고정된 크기의 feature map을 얻는 대신 quantization으로 인한 feature와 RoI사이의 어긋남이 발생함.

<br/>

### 2.3 RoI Align

<br/>

![image](https://user-images.githubusercontent.com/44194558/159151468-fa28120d-3321-4510-8ac0-83f270e5f20c.png)

Data pooling에 있어 quantization을 사용하지 않는 대신, 원본 RoI (4.53 x 6.25)를 9개의 동일한 크기를 가진 box로 분할하고, 개별 box마다 bilinear interpolation을 적용함.

![image](https://user-images.githubusercontent.com/44194558/159150934-307496a7-bcf5-44c6-b0ce-090d61e94854.png)

  - Align 연산 proces : https://miro.medium.com/max/1400/0*7WFmQBxoOCPu2BDJ.gif

<br/>

![image](https://user-images.githubusercontent.com/44194558/159151203-2bf0351d-c9af-4a47-ad68-b9a24bfe39e4.png)

 - 좌 (RoI Align) / 우 (RoI Pooling)
 
 - 초록색 : 연산에 사용되는 추가적인 data / 짙은 파랑 : quantization에서 손실되는 data / 옅은 파랑 : RoI Pooling 연산에서 손실되는 data 

RoI Align 방법을 통해 feature와 RoI 사이의 misalignment를 해결하고, 이를 통해 RoI의 정확한 spatial location을 보존하는 것이 가능해짐.

<br/>

### 2.4 Mask Representation

Mask는 객체의 spatial layout 정보를 인코딩함. 

 - `Unlike class labels or box offsets that are inevitably collapsed into short output vectors by fully-connected (fc) layers, extracting the spatial structure of masks can be addressed naturally by the pixel-to-pixel correspondence provided by convolutions.`

 - `We predict an m × m mask from each RoI using an FCN [30]. This allows each layer in the mask branch to maintain the explicit m × m object spatial layout without collapsing it into a vector representation that lacks spatial dimensions.`