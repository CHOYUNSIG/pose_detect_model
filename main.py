from process.CameraProcess import CameraProcess
from process.MediapipeProcess import MediapipeProcess
from process.YoloProcess import YoloProcess
from PipedProcess import Pipeline


if __name__ == "__main__":
    camera_process = CameraProcess(camera_id=0)
    yolo_process = YoloProcess(model_path="yolo11n.pt")
    mediapipe_process = MediapipeProcess(
        model_params={
            'min_detection_confidence': 0.6,
            'min_tracking_confidence': 0.6,
        }
    )

    pipeline = Pipeline(camera_process, yolo_process, mediapipe_process)
    pipeline.start()