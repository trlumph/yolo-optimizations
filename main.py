from typing import List, Type, Union
from detectors import BaseDetector, PTDetector, ONNXDetector, TFLiteDetector, RknnDetector, DetectionResult
import cv2
import numpy as np
from argparse import ArgumentParser
from yolo_model import YOLOModelInfo
import time


class YOLOv8Inference:
    def __init__(self, detector: BaseDetector, model_info: YOLOModelInfo, host_ip: str, host_port: int):
        self.detector = detector
        self.model_info = model_info
        self.host_ip = host_ip
        self.host_port = host_port

    def draw_bboxes(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        for det in detections:
            if det.class_id >= len(self.model_info.classes):
                print(f"Skipping invalid class ID {det.class_id}")
                continue
            color = self.model_info.color_palette[det.class_id]
            cv2.rectangle(frame, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), color, 2)
            label = f"{self.model_info.classes[det.class_id]}: {det.confidence:.2f}"
            cv2.putText(frame, label, (int(det.x1), int(det.y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

    def stream_video(self, video_source: Union[int, str]):
        cap = cv2.VideoCapture(video_source)
        output_writer = cv2.VideoWriter(
            f"appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
            f"rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={self.host_ip} port={self.host_port}",
            cv2.CAP_GSTREAMER, 0, 30,
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            detections = self.detector.infer(frame)
            inference_time = time.time() - start_time
            print(f"Inference time: {inference_time:.2f} seconds")
            frame = self.draw_bboxes(frame, detections)

            output_writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        output_writer.release()

def main():
    parser = ArgumentParser(
        prog='YOLOv8 Inference',
        description='Perform inference on a video stream using YOLOv8 model')
    parser.add_argument('--engine', type=str, choices=['pt', 'tflite', 'onnx', 'rknn'], required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--input-video-path', type=str, required=True)
    parser.add_argument('--host-ip', type=str, required=True)
    parser.add_argument('--host-port', type=int, required=True)

    args = parser.parse_args()

    detector_classes = {
        'pt': PTDetector,
        'onnx': ONNXDetector,
        'tflite': TFLiteDetector,
        'rknn': RknnDetector
    }

    model_info = YOLOModelInfo()

    detector_class = detector_classes[args.engine]
    detector = detector_class(args.model_path, model_info)

    inference = YOLOv8Inference(detector, model_info, args.host_ip, args.host_port)
    inference.stream_video(args.input_video_path)


if __name__ == "__main__":
    main()
