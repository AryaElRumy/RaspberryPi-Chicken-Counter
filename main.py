import cv2
from src.camera import Camera
from src.image_processing import ImageProcessor
from src.mqtt_handler import MQTTHandler
from src.yolo_detector import YOLODetector
from src.data_handler import DataHandler

def main():
    camera = Camera()
    image_processor = ImageProcessor()
    mqtt_handler = MQTTHandler()
    yolo_detector = YOLODetector()
    data_handler = DataHandler()

    camera.initialize()
    mqtt_handler.connect()

    frame_width, frame_height = camera.get_dimensions()
    center, radius = image_processor.get_roi_params(frame_width, frame_height)
    mask = image_processor.create_circular_mask((frame_height, frame_width), center, radius)

    while True:
        frame = camera.get_frame()
        display_frame = image_processor.draw_roi(frame, center, radius)
        display_frame = cv2.resize(display_frame, (frame_width//2, frame_height//2))
        cv2.imshow("Chicken Detection", display_frame)

        if mqtt_handler.trigger_processing:
            roi = image_processor.process_frame(frame, mask)
            results = yolo_detector.detect(roi)
            count = yolo_detector.count_chickens(results)

            if count > 0:
                result_frame = image_processor.draw_results(display_frame, count, results)
                cv2.imshow("Chicken Detection", result_frame)
                image_path = data_handler.save_frame(result_frame, count, mqtt_handler.current_weight)
                mqtt_handler.publish_data(mqtt_handler.current_weight, count, image_path)
            else:
                print("No chickens detected. Skipping data saving and publishing.")

            mqtt_handler.reset_trigger()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    mqtt_handler.disconnect()

if __name__ == "__main__":
    main()