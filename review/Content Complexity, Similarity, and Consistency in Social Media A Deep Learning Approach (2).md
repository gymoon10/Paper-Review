# Abstract

본 연구는 소셜 미디어 컨텐츠가 likes, reblogs같은 customer engagement에 미치는 영향을 조사하기 위해 다음과 같은 요소에 주목 : visual and textual
딥러닝의 SOTA기법을 활용하여 이미지와 텍스트 source로부터 그것들의 의미를 효과적으로 담고 있는 data-driven feature를 학습하는것이 목적.
=> **semantic representation 학습**

Semantic representation의 학습을 위해 소셜 미디어 컨텐츠에 대한 지표로 novel complexity, similarity, consistency를 도입.

본 연구는 적절한 시각적 자극(예쁜 이미지, 유명인, 성적 컨텐츠등), 이에 상응하는(complementary) 텍스트 컨텐츠, 일관적인 주제는 customer engagement에 긍정적인 영향을 끼친다는 것을 확인함. 추가적으로 이미지와 텍스트의 이질적인(heterogeneous) 효과에 대해서도 주목하였음.

본 연구는 딥러닝을 활용한 멀티미디어 컨텐츠들에 대한 의미론적 분석(semantic analysis)을 통해 효과적인 마케팅과 소셜 미디어 전략 수립에 대한 개선된 분석 방법과 모델을 제시.

<br/>

# 1. Introduction

소셜 미디어 컨텐츠들은 크게 이미지와 텍스트 두 타입의 정보들로 구성되어 있고, 본 연구는 머신러닝 알고리즘을 활용하여 기업 포스트들의 이미지, 텍스트 컨텐츠를 분석하고 그것들이 customer engagement에 끼치는 영향을 조사.

본 연구는 기존 연구들과 달리 그동안 무시되어 왔던 시각적 컨텐츠에 대한 분석을 실시하고, 어떤 종류의 이미지가 customer engagement에 긍정적인 영향을 끼치는 지 조사하고자 함.

시각적 컨텐츠에 대한 기존의 접근법들은 너무 공학적이어서 실제 소셜 미디어의 시각적 컨텐츠들에 대한 **의미론적(semantic)** 분석에 있어 한계가 있었음. 특히 이미지가 담고 있는 실제 의미와는 거리가 있는 픽셀값으로부터 도출되는 색, 밝기, 질감 등의 **feature complexity**에 주목하였음. 또한 이러한 feature complexity는 이미지의 사이즈, 위치, 구도에 따라 달라질 수 있다는 단점이 존재함. 

따라서 본 연구는 이미지에 대한 **semantic complexity**를 **feature complexity**와 구별하고, 사람이 이미지를 인식하고 수용하는데 있어 semantic complexity가 중요하다는 가정하에서 진행됨.

> 예시
> 
> 한 명의 패션 모델이 워킹하는 이미지 : 거리, 앵글, 채도 측면에서 feature complexity는 높을 수 있어도, 이미지에 한 명의 사람만 존재하기 때문에 semantic complexity는 낮음

전통적인 이미지 처리 기법은 도메인 지식과 특성 공학의 작업을 필요로 했지만, 딥러닝은 자동으로 데이터에 내재되어 있는 구조와 특징을 학습하고, 이미지 컨텐츠의 내용과 의미(semantic)를 파악할 수 있게 한다. - robust and data driven

### Semantic complexity

이러한 딥러닝 기법 덕분에 본 연구는 주어진 이미지에 등장하는 객체들을 인식, 예측하여 **semantic complexity**를 계산. 사전 학습된 모델을 통한 전이학습을 이용하여 aesthetic score를 계산하고, 성인 컨텐츠나 유명인을 포함하는 지 판별함.

### Textual & order complexity

또한 본 연구는 텍스트 컨텐츠의 complexity역시 고려함. **Textual complexity**를 global, local 측면에서 분석.
- global : 텍스트의 전체적인 토픽들 -> LDA를 통해 주어진 포스트의 텍스트의 latent 토픽들을 탐색하고, 개별 포스트의 토픽 distribution을 조사함. 
- local : 개별 단어 토큰들의 sequence -> word2vec을 활용하여 단어에 대한 벡터 표현을 학습,
(by maximizing the predicted probability of words co-occurring within a small window of consecutive words )
    - 단어들의 순서를 기반으로 주어진 포스트에서 개별 문장이 등장할 확률을 계산 (**order complexity**)
 
    - 확률이 낮은 문장은 독자가 예상하기 힘든 문장 

### novel similarity

텍스트와 이미지를 분리해서 독립적으로 분석했지만, 그 둘의 **상호 연관성** 역시 중요한 고려 대상임. 이에 대한 지표로 **novel similarity** 제안.
서로 다른 유형의 컨텐츠를 비교하기 위해 이미지를 딥러닝 모델을 통해 semantic content로 전환하고, 토픽 모델링을 통해 텍스트로부터 토픽과 토픽 확률 분포를 학습. => 서로 다른 유형의 컨텐츠로부터 도출된 토픽들의 유사도 계산 **content similarity**

### content consistency

개별 포스트의 이미지와 텍스트에 대한 semantic representation을 바탕으로 회사 블로그의 전체 포스트에 대한 average content를 구성. => 개별 포스트가 이 average contetent와 얼마나 유사한지 계산

본 연구는 메인 변수인 content complexity, similarity, consistency를 비롯한 다양한 관련 변수들이 사용자 engagement에 끼치는 영향을 분석. 
=> 선형 모델을 사용 (for time invariant fixed effects, control company level)

본 연구는 머신러닝 알고리즘을 활용하여 새로운 컨셉의 feature인 content complexity, similarity, consistency를 제안함. 컴퓨터 비전 태스크의 이미지 분류와 인식을 넘어서 보다 유용하고 추상적인 특징을 파악하는 semantic analysis를 제안. 이미지와 텍스트의 상호작용을 고려하여 보다 포괄적인 분석을 진행.



<br/>

# 2. Hypothesis Development


## Attention Attraction

**가설 1**
- 이미지와 움짤의 수가 customer engagement에 긍정적인 영향을 끼침

**가설 2**
- 이미지의 높은 feature complexity가 customer engagement에 긍정적인 영향을 끼침

feature complexity가 높음 <-> 이미지 픽셀값들의 variation이 큼 (보다 복잡하고 다양한 색, 채도 등)

## Comprehension

사람의 객체 인식은 이미지의 기본적인 특성(feature complexity)와는 무관하기 때문에 semantic complexity를 고려하고 반영해야 함.

**가설 3**

이미지의 명확한(salient)객체들의 수가 적으면 낮은 semantic complexity
- semantic complexity와 명확한 객체들의 수는 customer engagement에 부정적인 영향을 끼침
  
**가설 4**

- 이미지의 aesthetic score와 유명인의 포함 여부는 customer engagement에 긍정적인 영향을 끼침

**가설 5**

적은 수의 토픽에 주목하는 짧은 문장들로 구성된 텍스트가 선호되는 경향. (order complexity가 낮은 문장들이 선호됨)
- 문장, 토픽의 수와 order complexity는 customer engagement에 부정적인 영향을 끼침

**가설 6**

포스트의 태그들은 텍스트와 이미지의 semantic content와 부합해야 함. (topic complexity, semantic complexity가 낮을수록 좋음)
- 태그들의 topic complexity는 customer engagement에 부정적인 영향을, 태그의 수는 긍정적인 영향을 끼침

**가설 7**

- 이미지와 텍스트의 semantic similarity가 customer engagement에 긍정적인 영향을 끼침


## Preference

**가설 8**
- 이미지와 텍스트의 semantic consistency가 customer engagement에 긍정적인 영향을 끼침


# 3. Tumblr Data and Post Content Characteristics

![image](https://user-images.githubusercontent.com/44194558/135707305-64abfa29-752e-4d2e-9d5c-c13471a367e6.png)

## 3.1 Visaul Features

**Feature Complexity**

색과 밝기가 다양할 수록 높은 feature complexity - 이미지의 용량, 압축률로 판단

**Semantic Complexity**

이미지에 등장하는 객체들을 파악 - 컴퓨터 비전의 이미지 분류 태스크
CNN 알고리즘 활용 => It **automatically** discovers **robust representations** needed for accurate classiﬁcation–that is, the layers are **not designed by humans**, but are learned from the data

Each convolutional layer convolves the output of its previous layer with **a set of learned kernels (ﬁlters)** that extracts **distinct motifs** from the input. In between, a pooling layer performs a **sub-sampling** of the convoluted image to **reduce noise and suppress irrelevant variations for robustness**.

![image](https://user-images.githubusercontent.com/44194558/135707996-6b66e6c6-ac2b-4470-8ee9-93f9f695c5ac.png)


![image](https://user-images.githubusercontent.com/44194558/135708062-73cea61e-57c5-4941-b1fd-e93c38aba252.png)

![image](https://user-images.githubusercontent.com/44194558/135708077-056ac5ea-ed17-465a-9da5-8ad8b1532039.png)

 - d : 이미지에 등장하는 object의 개수
 - 개별 p_i를 모두 더하면 1
 - 이미지에 등장하는 객체 수가 적을 수록 complexity가 낮아짐

**Other Relevant Image Features**

사전 학습된 모델을 활용한 전이학습, 미세조정을 통해 유명인, 성인 컨텐츠를 포함하는지, 이미지에 몇 개의 salient object가 등장하는 지 예측.

![image](https://user-images.githubusercontent.com/44194558/135708226-7f5ed1b8-80b5-429b-b698-befc630176b7.png)

- 첫 번째 이미지는 복잡한 픽셀값 구조(색, 질감 등)를 가짐, 다양한 유형의 물체들이 존재 -> feature, semantic complexity 높음
  
- 네 번째 이미지는 첫 번째 이미지보다 다양한 유형의 물체들이 나타나기 때문에 보다 높은 semantic complexity
  
- 두 번째 이미지는 그레이 스케일이기 때문에 feature complexity낮음


## 3.2 Textual Features

텍스트의 개별 단어들은 micro level에서, 텍스트의 전체적인 의미는 macro level에서 분석됨. => global, local level에서 textual complexity를 계산

**Topic Complexity via Topic Modeling**

LDA는 주어진 텍스트는 몇 개의 잠재 토픽들로 구성되어 있고, 실제로 등장하는 단어들은 내재된 토픽들의 실현이라고 가정함.

LDA는 개별 토픽들의 키워드 집합과, 개별 텍스트에 대한 토픽 분포를 출력. => 토픽의 키워드들을 통해 토픽들의 의미를 알 수 있음

토픽 분포를 통해 complexity를 계산 (토픽이 다양할 수록 complexity 높음)

Perplexity 기준 활용 - 토픽의 적절한 개수를 탐색

Perplexity quantiﬁes how well the word counts of the held-out test documents are represented by the word distributions of the learned topics


![image](https://user-images.githubusercontent.com/44194558/135708486-34340fd1-11dd-49b0-80cd-68ae24d6df69.png)

  - 텍스트에서 20개, 태그에서 40개의 토픽을 선정
  - 해당 토픽들의 키워드 집합들이 coherent topic을 구성한다는 것을 확인


## 3.2.2 Order Complexity via Word2vec

**word2vec** has been proposed that **embeds words in a latent factor space** in a manner that captures a large number of precise **syntactic and semantic word relationships.**

word2vec 알고리즘은 네거티브 샘플링을 활용한 skip-gram모델을 활용 => 비슷한 문맥에서 사용되는 단어들이 벡터 공간에서 가깝게 위치할 수 있도록 개별 단어를 d 차원의 벡터로 표현

해당 벡터 표현은 연속적인 단어들의 window size내에서 동시 출현하는 단어들의 예측 확률을 최대화 하는 방식으로 학습됨. 

![image](https://user-images.githubusercontent.com/44194558/135708817-48ebdd17-3cc8-46f1-8feb-78441442fd43.png)
 
  - T : 문장 s에 존재하는 단어들의 수, b : window size
  - s_i : 문장 s의 i 번째 단어에 대한 벡터 표현

word2vec은 LDA와 달리 지역적인 문맥 정보를 표현함. word2vec이 특정 단어가 주어졌을 때 근처의 단어를 예측한다면 (focal words -> nearby words), LDA는 단어들을 문서 레벨에서 예측함 (documents -> topics -> words)

word2vec은 단어의 순서가 중요하지만 LDA는 그렇지 않음 (bag of words)

![image](https://user-images.githubusercontent.com/44194558/135708916-962fe0ef-50b9-44aa-9558-5b7db3f2bea3.png)

  - 주어진 포스트에서 특정 문장 s에 대한 확률을 계산
  - 특정 단어 i의 앞, 뒤 b개의 단어들에 대해 단어 j의 확률을 계산
  - 위의 과정을 총 T번 반복하여 문장에 대한 확률 계산
  - 해당 확률이 높을 수록 이웃 단어들을 고려했을 때 등장할 가능도가 높은 문장

![image](https://user-images.githubusercontent.com/44194558/135709304-459eb778-f965-42c7-ac77-adf65613bdc4.png)

  - The proposed order complexity can be considered **a measure of readability** considering the **likelihood of the sentence**.


![image](https://user-images.githubusercontent.com/44194558/135709353-4c1e2ea4-b298-48b8-af85-a362bc8ff0ad.png)

  - 첫 번째 문장은 하나의 토픽에 집중 => topic complexity 낮음
  - 나머지 문장둘은 여러개의 토픽들을 다루기 때문에 상대적으로 topic complexity 높음
  - 두 번째 문장은 세 번째 문장에 비해 확률(가능도)이 높은 문장들로 구성되었기 때문에 order complexity가 낮음


## 3.3 Content Consistency

개별 포스트가 회사 블로그의 일반적인 semantic content와 유사한지 평가

![image](https://user-images.githubusercontent.com/44194558/135709682-cf781d9c-580f-4704-b9d7-61b5c96e41e8.png)
  
   - omega : 특정 포스트 i를 제외한 나머지 모든 포스트들의 집합
   - c_j : LDA를 통해 계산된 상응하는 토픽 분포 or 이미지에서 예측된 라벨

![image](https://user-images.githubusercontent.com/44194558/135709772-03795e9b-3650-449d-9692-b5e915776ae1.png)

   - 컨텐츠 source인 텍스트, 태그, 이미지에 대한 consistency measure

![image](https://user-images.githubusercontent.com/44194558/135709795-4590a709-ed34-4283-9082-39734a223a42.png)


## 3.4 Visual and Textual Content Similarity

일관된 포스트는 서로 관련성이 있는 이미지와 텍스트로 구성됨. 픽셀 베이스의 이미지와 문자 베이스의 텍스트의 관계를 수치화하는 것은 단순한 태스크가 아님

이미지와 텍스트 두 컨텐츠에 대한 유사도를 계산하기 위해 공통된 표현으로 변환할 필요가 있음. => 이미지를 CNN 모델에 의해 예측된 라벨들의 집합으로 표현 (image corpus)

Image corpus는 일반 텍스트와 달리 단어(예측된 라벨들)의 순서가 중요하지 않음

텍스트 코퍼스와 이미지 코퍼스에 LDA를 적용시켜 이미지와 텍스트에 대한 토픽 분포를 계산. => P_image, P_text의 코사인 유사도를 계산

![image](https://user-images.githubusercontent.com/44194558/135709948-7ed4f125-5ea9-40f7-aacf-ac4f452094c6.png)


## 3.5 Variable Construction

reblog, like의 분포가 편향되어 있어 로그 변환 수행

동영상이나 GIF를 포함하는 지, 유명인이 포함되어 있는 지 등에 대한 다양한 이진 변수

![image](https://user-images.githubusercontent.com/44194558/135710084-fb3492ac-4a26-4e82-9b29-ee5837f675b8.png)


# 4. Model and Empirical Results

Linear model with company and time fixed effects

![image](https://user-images.githubusercontent.com/44194558/135710115-6b0e0c14-0cf7-4cdb-b98f-68de348228a4.png)




