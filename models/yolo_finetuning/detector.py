import torch
import os
import yaml
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class ProgressiveUnfreezeCallback():
    def __init__(self, freeze_layers, unfreeze_epoch):
        self.freeze_layers = freeze_layers
        self.unfreeze_epoch = unfreeze_epoch

    def __call__(self, trainer):
        if hasattr(self, 'on_train_epoch_end'):
            self.on_train_epoch_end(trainer)
    
    def on_train_epoch_end(self, trainer):
        epoch = trainer.epoch

        if epoch == 0:
            for i in self.freeze_layers:
                try:
                    for param in trainer.model.model[i].parameters():
                        param.requires_grad = False
                    print(f"[에폭 {epoch}] 레이어 {i} 동결 완료")
                except Exception as e:
                    print(f"[에폭 {epoch}] 레이어 {i} 동결 실패: {e}")
        elif epoch == self.unfreeze_epoch:
            for param in trainer.model.parameters():
                param.require_grad = True
            print(f"[에폭 {epoch}] 모든 레이어 해동 완료")

class NIRDetector():
    def __init__(self, config_path = None):
        self.cfg = {
            'model_type': 'yolov8s.pt',
            'data_yaml': '/data/data.yaml',
            'epochs': 50,
            'batch_size': 16,
            'image_size': 640,
            'freeze_layer': [0,1,2,3],
            'device' : 'cuda:0' if torch.cuda.is_available() else 'cpu',
            'lr': 0.01,
            'patience': 15,
            'output_dir': 'nir_eye_detection',
            'progressive_unfreeze': True,
            'unfreeze_epoch': 10
        }

        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_cfg = yaml.safe_load(f)
                self.cfg.update(user_cfg)
        
        os.makedirs(self.cfg['output_dir'], exist_ok=True)

        self.model = None

    # def create_data_yaml(self, train_path, val_path, class_names=['left_eye', 'right_eye']):
    #     data_dict = {
    #         'path': os.path.dirname(os.path.abspath(train_path)),
    #         'train': train_path,
    #         'val': val_path,
    #         'nc': len(class_names),
    #         'names': class_names
    #     }

    #     yaml_path = os.path.join(self.cfg['output_dir'], 'dataset.yaml')
    #     with open(yaml_path, 'w') as f:
    #         yaml.dump(data_dict, f)
        
    #     self.cfg['data_yaml'] = yaml_path
    #     print(f"데이터 YAML 파일 생성 완료: {yaml_path}")

    #     return yaml_path

    def prepare_model(self):
        self.model = YOLO(self.cfg['model_type'])

        # if self.cfg['freeze_layer'] and not self.cfg['progressive_unfreeze']:
        #     print(f"freeze 시작: 레이어 인덱스 {self.cfg['freeze_layer']}")
        #     for i in self.cfg['freeze_layer']:
        #         try:
        #             for param in self.model.model.model[i].parameters():
        #                 param.requires_grad = False
        #             print(f"레이어 {i} 동결 완료: {type(self.model.model.model[i])}")
        #         except Exception as e:
        #             print(f"레이어 {i} 동결 실패: {e}")
        
        return self.model
    
    def train(self):
        if self.model is None:
            self.prepare_model()

        # 점진적 해동 콜백 정의
        # callbacks = []
        # if self.cfg['progressive_unfreeze']:
        #     callbacks.append(self.progressive_unfreeze_callback)

        progreesive_callback = ProgressiveUnfreezeCallback(freeze_layers=self.cfg['freeze_layers'], 
                                                   unfreeze_epoch=self.cfg['unfreeze_epoch'])
        
        self.model.add_callback('on_train_epoch_end', progreesive_callback)

        results = self.model.train(
            data=self.cfg['data_yaml'],
            epochs=self.cfg['epochs'],
            imgsz=self.cfg['img_size'],
            batch=self.cfg['batch_size'],
            device=self.cfg['device'],
            lr0=self.cfg['lr'],
            patience=self.cfg['patience'],
            project=self.cfg['output_dir'],
            name='train',
            freeze=self.cfg['freeze_layers'] if not self.cfg['progressive_unfreeze'] else None, # 이거 써야하나?
            # callbacks=[progreesive_callback] # 콜백 함수 커스터마이징
        )
        
        print(f"학습 완료. 결과 저장 경로: {self.model.trainer.save_dir}")
        return results
    
    # def progressive_unfreeze_callback(self, trainer):
    #     """
    #     점진적 레이어 해동을 위한 콜백 함수
    #     """
    #     if not hasattr(trainer, 'epoch') or trainer.epoch < self.cfg['unfreeze_epoch']:
    #         # 초기 에폭: 레이어 동결
    #         if trainer.epoch == 0:
    #             for i in self.cfg['freeze_layers']:
    #                 for param in trainer.model.model[i].parameters():
    #                     param.requires_grad = False
    #             print(f"레이어 {self.cfg['freeze_layers']} 동결 완료")
    #     else:
    #         # 일정 에폭 이후: 모든 레이어 해동
    #         if trainer.epoch == self.cfg['unfreeze_epoch']:
    #             for param in trainer.model.parameters():
    #                 param.requires_grad = True
    #             print(f"에폭 {trainer.epoch}: 모든 레이어 해동 완료")

    def validate(self, weights=None):
        """
        학습된 모델 검증
        
        Args:
            weights (str): 검증할 가중치 파일 경로 (None이면 학습된 최종 모델 사용)
        """
        if weights:
            model = YOLO(weights)
        elif self.model:
            model = self.model
        else:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        # 검증 실행
        results = model.val(data=self.cfg['data_yaml'])
        print(f"검증 결과: mAP@0.5 = {results.box.map50:.4f}, mAP@0.5:0.95 = {results.box.map:.4f}")
        return results
    
    def test_on_image(self, image_path, weights=None, conf=0.25):
        """
        단일 이미지에서 눈 영역 감지 테스트
        
        Args:
            image_path (str): 테스트할 이미지 경로
            weights (str): 사용할 가중치 파일 경로 (None이면 학습된 최종 모델 사용)
            conf (float): 감지 신뢰도 임계값
        """
        if weights:
            model = YOLO(weights)
        elif self.model:
            model = self.model
        else:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        # 이미지에서 예측 수행
        results = model.predict(image_path, conf=conf)
        
        # 결과 시각화
        for r in results:
            im_array = r.plot()
            plt.figure(figsize=(12, 8))
            plt.imshow(im_array[..., ::-1])  # BGR to RGB
            plt.axis('off')
            plt.tight_layout()
            
            # 결과 저장
            save_path = os.path.join(self.cfg['output_dir'], f"result_{Path(image_path).stem}.jpg")
            plt.savefig(save_path)
            print(f"감지 결과 저장: {save_path}")
            
            # 감지된 눈 영역 좌표 출력
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            
            class_names = model.names
            for i, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
                print(f"감지 {i+1}: {class_names[int(cls)]}, 신뢰도: {conf:.4f}, 좌표: {box}")
        
        return results
    
    def crop_eyes(self, image_or_dir, output_dir=None, weights=None, conf=0.25):
        """
        이미지 또는 디렉토리에서 감지된 눈 영역을 크롭하여 저장
        
        Args:
            image_or_dir (str): 이미지 파일 경로 또는 디렉토리 경로
            output_dir (str): 크롭된 이미지 저장 디렉토리 (None이면 기본값 사용)
            weights (str): 사용할 가중치 파일 경로
            conf (float): 감지 신뢰도 임계값
        """
        if weights:
            model = YOLO(weights)
        elif self.model:
            model = self.model
        else:
            raise ValueError("모델이 로드되지 않았습니다.")
        
        # 출력 디렉토리 생성
        if output_dir is None:
            output_dir = os.path.join(self.cfg['output_dir'], 'cropped_eyes')
        os.makedirs(output_dir, exist_ok=True)
        
        # 입력이 디렉토리인 경우 모든 이미지 처리
        if os.path.isdir(image_or_dir):
            image_files = [os.path.join(image_or_dir, f) for f in os.listdir(image_or_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
        else:
            image_files = [image_or_dir]
        
        print(f"총 {len(image_files)}개 이미지에서 눈 영역 크롭 시작...")
        
        # 각 이미지 처리
        for img_path in tqdm(image_files):
            # 이미지 로드
            img = plt.imread(img_path)
            if img.dtype == np.float32:  # matplotlib이 0-1 범위로 정규화하는 경우
                img = (img * 255).astype(np.uint8)
            
            # 눈 감지
            results = model.predict(img_path, conf=conf)
            
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                
                # 각 감지된 눈 영역 크롭
                for i, (box, cls) in enumerate(zip(boxes, classes)):
                    x1, y1, x2, y2 = map(int, box)
                    class_name = model.names[int(cls)]
                    
                    # 이미지 크롭
                    cropped_eye = img[y1:y2, x1:x2]
                    
                    # 크롭된 이미지 저장
                    img_name = Path(img_path).stem
                    save_path = os.path.join(output_dir, f"{img_name}_{class_name}_{i}.jpg")
                    plt.imsave(save_path, cropped_eye)
        
        print(f"눈 영역 크롭 완료. 결과 저장 경로: {output_dir}")

    def convert_annotations(source_dir, target_dir, format_type='yolo'):
        """
        다양한 형식의 어노테이션을 YOLO 형식으로 변환 (별도 구현 필요)
        
        Args:
            source_dir (str): 원본 어노테이션 디렉토리
            target_dir (str): 변환된 어노테이션 저장 디렉토리
            format_type (str): 원본 어노테이션 형식 ('coco', 'voc', 등)
        """
        # 형식에 따라 변환 로직 구현
        # (이 함수는 예시로, 실제 구현은 형식에 따라 달라질 수 있음)
        pass