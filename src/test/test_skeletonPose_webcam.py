from src.humanpose.skeletonPose import HumanPose
import argparse
import cv2

def main(args):
    humanPose_estimator = HumanPose(args.model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return 1

    success, img = cap.read()

    while success:
        try:
            poses, img  = humanPose_estimator.getSkeleton(img)
        except Exception as e:
            print(e)
            success, img = cap.read()

            continue

        cv2.imshow("test", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        success, img = cap.read()


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument(
        '--model_path',
        default="./humanpose/checkpoint/checkpoint_iter_370000.pth",
        type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)