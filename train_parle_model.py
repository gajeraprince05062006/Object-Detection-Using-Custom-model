import sys
import os
import argparse

# Add yolov5 to the Python path
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))

from yolov5 import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='custom_dataset/train/data.yaml')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--weights', type=str, default='yolov5s.pt')
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='parle_g_detector')
    parser.add_argument('--device', type=str, default='cpu')  # use '0' if using GPU

    opt = parser.parse_args()
    opt_dict = vars(opt)

    print("✅ Starting Training with these settings:")
    for k, v in opt_dict.items():
        print(f"{k}: {v}")

    train.run(**opt_dict)
    print("\n✅ Training complete. Check runs/train/parle_g_detector/weights/best.pt")

if __name__ == '__main__':
    main()
