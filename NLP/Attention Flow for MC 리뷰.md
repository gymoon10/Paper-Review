# Bi-Directional Attention Flow for Machine Comprehension

<br/>

## Abstract

MC (Machine Comprehension)는 주어진 문맥에서 쿼리에 대한 옳바른 답을 출력하는 것을 의미하고, 이를 위해서는 문맥과 쿼리의 interaction을 잘 모델링하는 것이 중요하다. 

최근에 attention 매커니즘이 MC 영역까지 확장되고 있다. 본 연구가 제안하는 BiDAF는 **문맥을 다양한 레벨의 granularity에서 표현하고, 압축/요약 없이 query aware context representation을 얻는 multi stage hierarchical 프로세스**이다. 

`A hierarchical multi-stage architecture for modeling the representations of the context paragraph at different levels of granularity` 

<br/>

## 1. Introduction

모델이 context paragrph의 특정 타깃 부분에 집중할 수 있도록 하는 attention 메커니즘 덕분에 자연어, 컴퓨터 비전 영역에서 MC, QA 문제를 해결하는 데 큰 발전이 있었다. 

`Neural attention mechanism, which enables the system to focus on a targeted area within a context paragraph (for MC) or within an image (for Visual QA), that is most relevant to answer the question`

기존 Attention 메커니즘의 특징은 다음과 같다.

1. 가장 연관성 있는 정보를 추출하기 위해 문맥 (context or context document)을 고정된 길이의 벡터로 압축/요약한 후 attention을 계산
   
   `The computed attention weights are often used to extract the most relevant information from the context for answering the question by sum- marizing the context into a ﬁxed-size vector`

 2. 텍스트 도메인에 있어 가끔씩 dynamic한 측면이 있다. 즉, 현재의 attention은 이전 time step에서의 attention vector에 기반하여 만들어진 함수.
 
 3. Query를 보고 주목해야 할 context의 부분을 찾는다. (Query-to-Context, uni-directional)

본 연구가 제안하는 BiDAF는 문자 레벨, 단어 레벨, 문맥 임베딩을 포함하고 양방향의 attention을 활용하여 query aware context representaion을 얻는다. 본 연구에서 사용하는 attention 기법은 기존의 attention 파라다임과 비교했을 때 다음과 같은 이점들이 있다.

1. Attention 레이어에서 더 이상 **문맥을 고정된 길이의 벡터로 요약하지 않는다**.
    
   Attention은 매 time step마다 계산되고, 매 시점마다 계산된 attended 벡터는 이전 계층의 representation과 함께 다음의 계층으로 전달된다.

   -> Early summarization으로 인한 정보의 손실을 방지

2. **Memory less** attention mechanism 사용. 
   
   개별 time step마다 계산되는 attention은 오직 현재 시점의 query와 context paragraph의 함수이고, 이전 시점들의 attention 레이어들을 사용하지 않는다. 
   
   이러한 단순화는 modeling layer와 attention layer의 분업을 유도하여, attention layer는 query와 context 간의 attention을, modeling layer는 query aware context representation을 학습하도록 한다. 

   -> 이전 시점들의 부정확한 attention의 영향을 받지 않는다.


3. **양방향** attention 매커니즘 사용. 
   
   상호 보완 가능한 query-to-context, context-to-query 정보를 모두 활용.


<br/>

## 2. Model

![image](https://user-images.githubusercontent.com/44194558/140277073-a98dbf03-cf9c-42a8-80f7-7d3ec23ca5fb.png)

본 연구가 제안하는 모델은 Hierarchical multi stage process이고 6개의 계층으로 구성된다.

1. Character Embedding Layer : 문자 레벨 CNN을 통해 개별 단어를 벡터 공간으로 mapping

2. Word Embedding Layer : 사전 학습된 word embedding 모델을 사용하여 단어를 벡터 공간으로 mapping

3. Contextual Embedding Layer : 특정 단어의 주변 단어들을 통해 해당 단어의 embedding을 정제
 
   **위 3개의 레이어는 query, context 모두 적용됨**

4. Attention Flow Layer : Query, context를 쌍으로 묶어 context의 각 단어들에 대해 query aware feature vector를 생성

5. Modeling Layer : RNN을 사용하여 context를 탐색

6. Output Layer : Query에 대한 답을 생성

<br/>

**Character Embedding Layer**

각 단어를 고차원의 벡터 공간으로 mapping 한다. CNN을 사용하여 문자 단위의 embedding 벡터 표현을 학습한다. 각 단어에 1D 벡터를 할당하고, 이를 입력으로 받는 형식이다.  CNN의 출력은 각 단어에 대한 고정된 길이의 벡터를 얻기 위해 max pooling 된다. 

![image](https://user-images.githubusercontent.com/44194558/140279382-7fff6a1f-3451-4971-8ffd-309a10124896.png)

<br/>

**Word Embedding Layer**

단어를 고차원의 벡터 공간으로 mapping한다. 각 단어에 대한 고정된 길이의 word embedding 벡터를 얻고자 사전 학습된 단어 벡터 GloVe를 이용한다. 문자, 단어 임베딩 벡터는 columnwise하게 concatenate되어 (m번째 단어는 m번째 열을 가짐) Highway 네트워크에 전달된다. 

![image](https://user-images.githubusercontent.com/44194558/140279810-74996880-719a-4498-b24a-47fbc5f2b43d.png)

<br/>

**Contextual Embedding Layer**

Concatenate된 문자, 단어 임베딩을 입력으로 받아 LSTM을 적용시켜 단어 간의 문맥적 상호작용을 학습한다. **양방향 LSTM을 사용하여 주변 문맥을 파악**하는 과정이다. 양방향이기 때문에 출력은 2차원이고, 코드에서는 계산량 감소를 위해 GRU를 사용한다.


![image](https://user-images.githubusercontent.com/44194558/140280874-fc5822a4-11d2-4cc8-b938-a272ebfd1d92.png)

**위 3 계층은 CNN 신경망의 다단계 특징 계산과 유사하게 서로 다른 granularity 수준에서 query와 context에 대한 특징을 계산한다**

<br/>

**Attention Flow Layer**

**Query, context의 단어들의 정보를 연결**시키는 작업이다. 기존 방식과 다르게 query, context를 고정된 길이로 요약하고, single feature vector로 변환하는 작업을 하지 않는다. 매 시점의 attention 벡터는 이전 계층들의 embeddings와 함께 이후의 모델 계층으로 전달된다. 이러한 방식은 early summarization으로 인한 정보의 손실을 방지한다.

해당 계층의 입력으로 context H, query U에 대한 contextual vector representation이 주어진다 (LSTM에 의해). 해당 계층의 출력은 context의 단어들에 대한 **query aware vector representation**이다. 

Attention을 양방향으로 계산한다 : context to query & query to context. Query2Context는 context의 어느 정보가 query와 관련이 있는 가를 학습하고, Context2Query는 Query의 어느 정보가 Context와 관련이 있는 지를 학습한다. **Query, Context에 대한 attention이 양방향으로 이루어지기 때문에** BiDAF라고 불린다.

또한, query, context의 유사도를 파악하기 위한 유사도 행렬이 사용되고, 이 **유사도 행렬로부터 양방향의 attention을 도출**하게 된다. Context는 전체 document가 입력으로 주어지는 형식이며, t번째 context 단어와 j 번째 query 단어의 유사도를 계산한다. a는 훈련 가능한 scalar function으로 두 입력 벡터들 간의 유사도를 encoding한다. 계산된 S를 활용하여 양방향의 attention, attended vector를 얻는다.


![image](https://user-images.githubusercontent.com/44194558/140282563-d2c74a4f-f2f7-4c2e-9aeb-1383841da671.png)


![image](https://user-images.githubusercontent.com/44194558/140281449-6bd7e7fa-0178-4252-95f4-f3c433160402.png)

<br/>

* **Context-to-query Attention**

  Attention weight : ![image](https://user-images.githubusercontent.com/44194558/140283251-8fdabc57-c1cf-4902-a365-a857c04302b8.png)

    - query 단어 중 어떤 단어가 각각의 context 단어와 연관이 있는 지 확인
    - 계산된 S_tj에 대해 유사성이 큰 weight만 값이 커짐



  Attended query vector : ![image](https://user-images.githubusercontent.com/44194558/140283444-e3a6d58e-5575-419e-b0e4-d664f55502b5.png)

    - 유사성이 큰 query 단어들이 반영됨

  <br/>

* **Query-to-context Attention**
 
  `signiﬁes  which  context  words have  the  closest  similarity  to  one  of  the  query  words  and  are  hence  critical  for  answering  the  query.`

  Attention weight : ![image](https://user-images.githubusercontent.com/44194558/140283829-8d1fda3c-1f49-4df8-9acd-98bdbfea084b.png)

    - max_col : 각 row에서 가장 큰 값을 추출
    - 각 context 단어마다 query 단어들 중 유사도가 가장 큰 항만 추출
    - softmax를 통해 상대적으로 더 query와 연관성이 있는 context 단어들만 남김


  Attended context vector : ![image](https://user-images.githubusercontent.com/44194558/140283938-19da9ad1-3da1-4ab3-bbaf-df3f2301901b.png)

    - query에 대해 가장 중요한 context 단어들의 가중합
    - query 입장에서 중요한 단어들만 살아남음


최종적으로 contextual embedding, attention vectors가 결합된다 : ![image](https://user-images.githubusercontent.com/44194558/140284319-1e73e0e4-6b0f-4c77-a0d7-ca01bb88791f.png)

<br/>

**Modeling Layer**

입력으로 context 단어들에 대한 query aware representation을 인코딩한 G를 입력으로 받아 Query, context간의 interaction을 학습하기 위해 양방향 LSTM을 사용. 

이전의 contextual embedding은 query를 고려하지 않고 context 단어들간의 interaction을 고려했다는 점에서 차이가 있다.

최종 출력인 M의 각 열은 전체 context, query의 단어들에 대한 문맥 정보를 담고 있음.

![image](https://user-images.githubusercontent.com/44194558/140285725-d1d847ee-cdf6-40db-9770-7010162ab138.png)

<br/>

**Output Layer**

Application specific. QA 태스크는 query에 대한 답변을 하기 위해 주어진 단락에서 sub phrase를 찾고자 한다. Phrase는 단락에서 phrase의 start, end index를 예측함으로써 얻어짐. 

Start index에 대한 확률 분포 : 

![image](https://user-images.githubusercontent.com/44194558/140287383-8a9aacdf-f655-4ca5-974b-2093f01c1d9a.png)

End index에 대한 확률 분포 : 

![image](https://user-images.githubusercontent.com/44194558/140287412-4284ea48-5661-4921-9d7b-1af04626d618.png)


![image](https://user-images.githubusercontent.com/44194558/140286700-d1577080-3ea6-4740-8e98-5aaa70d576d7.png)

<br/>

**Training**

`The  sum  of  the  negative  log  probabilities of  the true  start  and end indices by the predicted  distributions, averaged  over  all  examples`

![image](https://user-images.githubusercontent.com/44194558/140287733-d6499349-3aa1-4be3-8492-cab0605117f8.png)

<br/>

## 3. Result & Conclusion

![image](https://user-images.githubusercontent.com/44194558/140296726-aaa2dceb-ca05-4523-826d-70eb938b26e3.png)

단일 모델로는 SOTA라고 하기 어렵지만, 앙상블 모델로서는 SOTA 성능을 보이고 있다. C2Q, word embedding이 성능에 많은 영향을 끼치고 있음.

![image](https://user-images.githubusercontent.com/44194558/140297130-6ab7a63e-c77f-4910-a615-2d36721d1ec1.png)

Word보다 Context가 query에 대한 정보를 잘 파악하고 있다.

![image](https://user-images.githubusercontent.com/44194558/140297205-6cabb9bd-ff09-4163-9220-27ab4709e1f7.png)

Context embedding layer가 동음이의어에 대해 semantic한 의미를 잘 clustering하고 있다.

날짜로서의 may, ~할 것 같아로서의 may가 잘 구분되고 있다.

![image](https://user-images.githubusercontent.com/44194558/140297578-fc537954-5db3-46a3-9079-1004689011ca.png)

Query에 대해 context의 어떤 부분에 좀 더 주목해야 하는지, 가중치를 시각화

![image](https://user-images.githubusercontent.com/44194558/140297742-f944372b-c8b7-43b7-8532-27a736b6fb56.png)

CNN/DailyMail 데이터에 대해서도 SOTA 성능을 보인다

<br/>

## Conclusion

Context, query에 대한 양방향 attention을 통해 context나 query에 대한 요약 없이 query aware context representation을 찾아내는 모델 구조를 제시했다.

Attention을 어떻게 사용하는가에 대한 유용한 지침을 제안했다.




