import csv
import os
import logging
import random
from imageai.Detection.Custom import CustomObjectDetection
import pandas as pd
import numpy as np

__author__ = 'ING_DS_TECH'
__version__ = "201909"

FORMAT = '%(asctime)-15s %(levelname)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)

input_dir = "dataset/test"
answers_file = "test.csv"

labels = ['standard', 'task2_class', 'tech_cond', 'Bathroom', 'Bathroom cabinet', 'Bathroom sink', 'Bathtub', 'Bed',
          'Bed frame', 'Bed sheet', 'Bedroom', 'Cabinetry', 'Ceiling', 'Chair', 'Chandelier', 'Chest of drawers',
          'Coffee table', 'Couch', 'Countertop', 'Cupboard', 'Curtain', 'Dining room', 'Door', 'Drawer',
          'Facade', 'Fireplace', 'Floor', 'Furniture', 'Grass', 'Hardwood', 'House', 'Kitchen',
          'Kitchen & dining room table', 'Kitchen stove', 'Living room', 'Mattress', 'Nightstand',
          'Plumbing fixture', 'Property', 'Real estate', 'Refrigerator', 'Roof', 'Room', 'Rural area',
          'Shower', 'Sink', 'Sky', 'Table', 'Tablecloth', 'Tap', 'Tile', 'Toilet', 'Tree', 'Urban area',
          'Wall', 'Window']

labels_task_2 = ["Bathroom", "Bedroom", "Dining room", "Kitchen", "Living room", "Property", "Real estate",
                 "Rural area", "Urban area"]

labels_task3_1 = [1, 2, 3, 4]
labels_task3_2 = [1, 2, 3, 4]


def task_1():
    logger.debug("Performing task 1.")

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("detection_model-ex-003--loss-0012.363.h5")
    detector.setJsonPath("detection_config.json")
    detector.loadModel()

    file_names = list()
    for file in os.listdir(input_dir):
        if file[0] == "." or ".db" in file:
            continue
        file_names.append(file)
    df = pd.DataFrame(np.zeros((len(file_names), len(labels))), columns=labels, dtype=int, index=file_names)

    for file in os.listdir(input_dir):
        if file[0] == "." or ".db" in file:
            continue
        detections = detector.detectObjectsFromImage(input_image=input_dir + "/" + file,
                                                     output_image_path=(input_dir + "_done/" + file),
                                                     minimum_percentage_probability=90,
                                                     extract_detected_objects=False)
        detection_list = list()
        for detection in detections:
            same = False
            detection_list.append(detection)
            for i in range(len(detection_list)):
                if same:
                    break
                if detection_list[i]["box_points"] == detection["box_points"] and detection["percentage_probability"] > detection_list[i]["percentage_probability"]:
                    df.at[file, detection_list[i]["name"]] = 0
                    detection_list[i] = detection
                    df.at[file, detection_list[i]["name"]] = 0
                    same = True
            if not same:
                detection_list.append(detection)
                df.at[file, detection["name"]] = 1
    df.to_csv("test.csv", index_label="filename")
    logger.debug("Done with Task 1.")
    return df


def task_2(df):
    logger.debug("Performing task 2.")
    df["task2_class"] = "house"

    file_names = list()
    for row in os.listdir(input_dir):
        if ".jpg" not in row:
            continue
        file_names.append(row)

    pd.set_option("display.max_rows", 30)
    pd.set_option("display.max_columns", 60)
    groups = {
        "Bathroom": [0, 1, 1, 1, -1, -1, -1, 0, 1, 1, 1, 0, -1, -1, 0, 1, -1, 1, 0, 1, 0, -1, -1, 1, 1, -1, 1, -1, 0, -1, -1, # kitchen stove last
                     -1, -1, 1, 1, 0, 0, -1, -1, 1, 0, 1, 1, -1, 0, 0, 1, 1, 1, -1, 0, 1, 1],
        "Bedroom": [0, -1, -1, -1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, -1, 0, 1, 1, -1, 0, -1, 0, -1, -1,
                    -1, 1, 1, -1, 0, 0, -1, 0, 0, 0, -1, -1, -1, 1, 1, -1, -1, -1, -1, 0, 1, 1],
        "Dining room": [0, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 0, 1, -1, 1, 1, 1, 0, 1, 0, -1, -1, 1, 1, -1, 1, -1, 0, -1, -1,
                        -1, -1, 0, -1, 0, 0, 0, -1, 1, 0, -1, 0, -1, 1, 1, 0, 0, -1, -1, 0, 1, 1],
        "Kitchen": [0, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 0, 0, -1, 1, 1, 1, 0, 1, 0, -1, 0, 1, 1, -1, 1, -1, 0, 1, 1,
                    -1, -1, 0, 0, 0, 0, 1, 0, 1, 0, -1, 0, -1, 1, 1, 0, 1, -1, -1, 0, 1, 1],
        "Living room": [0, -1, -1, -1, -1, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, -1, 1, 1, 1, -1, 1, -1, 0, -1, -1,
                        1, 1, 1, -1, 0, 0, -1, -1, 1, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0, 1, 1],
        "Property": [0, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, -1, -1,
                     -1, -1, -1, -1, 0, 0, -1, 1, -1, 0, -1, -1, 1, -1, -1, -1, -1, -1, 1, 0, 1, 1],
        "Real estate": [0, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, 1, -1, 0, -1, 1, -1, 1, 0, -1, -1,
                        -1, -1, -1, -1, 0, 0, -1, 1, -1, 0, -1, -1, 1, -1, -1, -1, -1, -1, 1, 0, 1, 1],
        "Rural area": [0, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, 1, -1, 0, -1, 1, 0, 1, 0, -1, -1,
                       -1, -1, -1, -1, 0, 0, -1, 1, -1, 0, -1, -1, 1, -1, -1, -1, -1, -1, 1, 0, 1, 1],
        "Urban area": [0, -1, -1, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 1, -1, 1, -1, 0, -1, -1, 0, 0, 0, -1, -1,
                       -1, -1, -1, -1, 0, 0, -1, 1, -1, 0, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0, 1, 1],
    }

    for key in groups.keys():
        for row in file_names:
            i = 0
            score = 0
            for col in labels[3:]:
                score = score + (df.at[row, col] * groups[key][i])
                i += 1
            df.at[row, key] = score

    for row in file_names:
        if ".jpg" not in row:
            continue
        df.at[row, "standard"] = random.randint(3, 4)
        df.at[row, "tech_cond"] = random.randint(3, 4)
        maxi = df.at[row, "Bathroom"]
        max_label = "Bathroom"
        for col in df[labels_task_2].keys():
            if df.at[row, col] > maxi:
                maxi = df.at[row, col]
                max_label = col
        if maxi <= 0:
            max_label = random.choice(["house", "bathroom", "bedroom", "dinning_room", "kitchen", "living_room"])
        for col in df[labels_task_2].keys():
            if col != max_label:
                df.at[row, col] = 0
            else:
                df.at[row, col] = 1
                if col not in df[["Property", "Real estate", "Rural area", "Urban area"]]:
                    if col in df[["Bathroom", "Bedroom", "Dining room", "Kitchen", "Living room"]]:
                        df.at[row, "Room"] = 1
                    df.at[row, "House"] = 0
                    df.at[row, "task2_class"] = col.replace(" ", "_").lower()
                    if(df.at[row, "task2_class"]) == "dining_room":
                        df.at[row, "task2_class"] = "dinning_room"
                else:
                    df.at[row, "task2_class"] = "house"
        for col in df[labels_task_2].keys():
            if df.at[row, col] < 0:
                df.at[row, col] = 0

    df.to_csv("test.csv", index_label="filename")
    logger.debug("Done with Task 2.")

    return df


def main():
    logger.debug("Sample answers file generator")
    df = task_1()
    task_2(df)


if __name__ == "__main__":
    main()
