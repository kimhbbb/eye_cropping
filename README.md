# Purpose
근거리 기기만을 사용한 실험을 원거리로 확장시키자. (가능하면 real-time)   

(pupil 실험)을 수행하기 위해서는 eye image가 필요함.  
따라서 cropped된 eye image를 output으로 할 수 있는 detection 모델을 만들자.  

먼저 RGB image로 detection model을 만들고, 후에 IR image로 traninig 시켜 적합한 model 만들자.  

# Dataset
### Detection
(pupil 실험)의 input image size = 224x224.  

원거리가 내가 생각하는 엄청 먼 거리까지를 의미하는 거 같지는 않고, 상반신 정도에서 detect를 수행하려는 거 같음.  

(pupil 실험)에서 사용 중인 dataset
> CASIA?

(조사)
> EIMDSD: 승인 받아야 함.

### Landmark    
얼굴 측면에서의 eye detect를 생각했을 때, landmark 이용하는 것도 좋을 거 같음.  
![Logo](https://github.com/사용자명/저장소명/raw/브랜치명/경로/파일명.png)
![Logo](https://github.com/사용자명/저장소명/raw/브랜치명/경로/파일명.png)





<details>
  <summary>오 이게 토글</summary>

  숨겨진 내용입니다.  
  여러 줄도 가능하고, 마크다운 문법도 함께 쓸 수 있어요!

  - 리스트도 되고
  - **굵은 글씨**, _기울임_ 등도 다 됨!
</details>
