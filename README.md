# Purpose
근거리 기기만을 사용한 실험을 원거리로 확장시키자. (가능하면 real-time)   

(pupil 실험)을 수행하기 위해서는 eye image가 필요함.  
따라서 cropped된 eye image를 output으로 할 수 있는 detection 모델을 만들자.  

먼저 RGB image로 detection model을 만들고, 후에 IR image로 traninig 시켜 적합한 model 만들자.  

# Dataset
**사용할 모델(detection / landmark)가 정해져야 명확해질 거 같긴 함**     

### Detection
(pupil 실험)의 input image size = 224x224.  

원거리가 내가 생각하는 엄청 먼 거리까지를 의미하는 거 같지는 않고, 상반신 정도에서 detect를 수행하려는 거 같음.  

(pupil 실험)에서 사용 중인 dataset
> CASIA?

(조사)
> EIMDSD: 승인 받아야 함.

### Landmark    
<span>
  <img src="https://github.com/kimhbbb/eye_cropping/blob/main/assets/11.jpg" style="width:400px; margin-right:10px;">
  <img src="https://github.com/kimhbbb/eye_cropping/blob/main/assets/22.jpg" style="width:400px;">
</span>

위 이미지와 같이 landmark를 이용했을 때(mediapipe face mesh), detection model보다 측면에서의 탐지 성능이 좋을 거 같음.(확실X)  

(% 고민사항)  
- real-time을 고려했을 때, 가벼운 모델 사용해야 함.
- 측면에 대해서 detection과 landmark 성능을 비교해본 건 아님.
- 위 사항들이 걸림돌이 되지 않는다면 landmark 사용하는 게 좋을 거 같음.(내 생각)

# Papers


# Implementation





<details>
  <summary>오 이게 토글</summary>

  숨겨진 내용입니다.  
  여러 줄도 가능하고, 마크다운 문법도 함께 쓸 수 있어요!

  - 리스트도 되고
  - **굵은 글씨**, _기울임_ 등도 다 됨!
</details>
