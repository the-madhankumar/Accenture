from ultralytics import YOLO
import shutil
import os

path_list = [
    "./Classification/Model/best.pt",
    "./Detection/Model/best.pt"
]

Output_path_list = [
    "./Classification/Model/best.engine",
    "./Detection/Model/best.engine"
]

for i in range(len(path_list)):
    model = YOLO(path_list[i])

    exported_model = model.export(format="engine", imgsz=640, batch=1, device="0")

    output_directory = Output_path_list[i]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    shutil.move(exported_model[0], output_directory)

    print(f"Model exported to {output_directory}")
