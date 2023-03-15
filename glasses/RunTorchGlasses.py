"""
This module loads a pytorch model, reads video input frame by frame (OpenCV),
transforms the images into a pytorch tensor and uses the model to label if
someone wears glasses or not.

This code was inspired by
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

from __future__ import print_function
from __future__ import division
import torch
import torchvision
from PIL import Image
import pickle
import cv2
from CommonTorchGlasses import get_evaluation_transform, get_classes_file_path, get_model_path
from CommonGlasses import CvFaceCapture

if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    # detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Computation device: %s" % (device))

    # load classes (idx, strings)
    with open(get_classes_file_path(), "rb") as fp:
        classes = pickle.load(fp)

    # load model
    print("Load model: %s" % (get_model_path()))
    model_ft = torch.load(get_model_path())
    model_ft.eval()
    model_ft = model_ft.to(device)

    # load torch image transformation
    t = get_evaluation_transform(224)

    # create the frame face capture
    cv_cap = cv2.VideoCapture(0)
    cap = CvFaceCapture(cv_cap)

    while (True):
        frame, faces = cap.read()
        for i in range(0, len(faces)):
            [x, y, w, h] = faces[i]
            face = frame[y:y + h, x:x + w]

            # transform opencv image into torch tensor
            pil_image = Image.fromarray(face)
            image_t = t(pil_image).unsqueeze_(0)
            image_t = image_t.to(device)

            prediction = model_ft(image_t)
            class_idx = (torch.max(prediction, 1)[1]).data.cpu().numpy()[0]
            class_str = classes[class_idx][1]
            print(class_str, prediction)

            pen_color = (0, 0, 255) if class_idx == 0 else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), pen_color, 4)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'Class: ' + class_str, (40, 40 + 50 * i), font, 1, pen_color, 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        keyCode = cv2.waitKey(1)
        if keyCode == ord('q'):
            break

    cv_cap.release()
    cv2.destroyAllWindows()
