import torch
from ultralytics import YOLO
import argparse
from detector import NIRDetector

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NIR 이미지에서 눈 영역 감지')
    parser.add_argument('--config', type=str, default=None, help='설정 파일 경로')
    parser.add_argument('--mode', type=str, default='train', choices=['train','val','test','crop'],
                          help='실행 모드')
    parser.add_argument('--weights', type=str, default=None, help='학습된 가중치 파일 경로 (test용)')
    parser.add_argument('--input', type=str, default=None, help='이미지 또는 디렉토리 경로')
    parser.add_argument('--output', type=str, default=None, help='출력 저장 디렉토리 경로')
    parser.add_argument('--train_path', type=str, default=None, help='train 데이터 경로')
    parser.add_argument('--val_path', type=str, default=None, help='val 데이터 경로')

    args = parser.parse_args()

    detector = NIRDetector(args.config)
    
    if args.mode == 'train':
        detector.prepare_model()
        detector.train()
    
    elif args.mode == 'val':
        detector.validate(weights = args.weights)

    elif args.mode == 'test':
        # 단일 이미지 테스트
        if args.input:
            print(args.weights)
            detector.test_on_image(args.input, weights=args.weights)
        else:
            print("테스트할 이미지 경로를 지정해주세요.")