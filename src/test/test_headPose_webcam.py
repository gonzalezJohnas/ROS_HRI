from src.headpose.headPose import HeadPose
import argparse
import cv2

def main(args):
    headPose_estimator = HeadPose(args.model_path, args.device_id)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return 1

    success, img = cap.read()

    while success:
        try:
            faces, img_test = headPose_estimator.getPose(img)
        except Exception as e:
            print(e)
            success, img = cap.read()

            continue

        cv2.imshow("test", img_test)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        success, img = cap.read()


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument(
        '--model_path',
        default="./headpose/checkpoint",
        type=str)

    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)