import tkinter as tk
from tkinter import ttk
import cv2 as cv
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import argparse
from tkinter import messagebox
from tkinter import Tk, Frame, Checkbutton
from tkinter import BooleanVar, BOTH
from tkinter import filedialog



# creating the main window
class Main(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.create_main_window()

    global video_flag
    global image_flag
    video_flag = False
    image_flag = False
    global webcam_flag
    webcam_flag = False

    def create_main_window(self):
        self.pic = tk.PhotoImage(file="pic.png")
        upload_image = tk.Button(command=self.open_image, image=self.pic)
        upload_image.place(x=0, y=0)
        caption_image = tk.Label(text='Загрузить фото', font="times 15")
        caption_image.place(x=0, y=145)
        self.pic_web_cam = tk.PhotoImage(file="Webcam.png")
        turn_on_web_cam = tk.Button(command=self.activate_webcam, image=self.pic_web_cam)
        turn_on_web_cam.place(x=190, y=0)
        caption_web_cam = tk.Label(text="Включить веб камеру", font="times 15", justify=tk.LEFT)
        caption_web_cam.place(x=190, y=145)
        self.pic_video = tk.PhotoImage(file="op.png")
        upload_video = tk.Button(command=self.open_video, image=self.pic_video)
        upload_video.place(x=395, y=0)
        caption_video = tk.Label(text="Загрузить видео", font="times 15", justify=tk.LEFT)
        caption_video.place(x=395, y=145)
        self.pic_neural_network = tk.PhotoImage(file="neural.png")
        apply_network = tk.Button(command=self.child_window, image=self.pic_neural_network)
        apply_network.place(x=615, y=0)
        caption_network_opencv = tk.Label(text='Нейронные сети/OpenCV', font="times 15", justify=tk.LEFT)
        caption_network_opencv.place(x=580, y=145)

    def activate_webcam(self):
        global webcam_flag
        webcam_flag = True

    def child_window(self):
        Child()

    def open_video(self):
        ftype = [('Видео(*.mp4)', '*.mp4')]
        dlg = filedialog.Open(filetypes=ftype)
        global fil_video
        fil_video = dlg.show()
        global video_flag
        video_flag = True

    def open_image(self):
        ftypes = [('PNG Image', '*.png'), ('JPG Image', '*.jpg')]
        dlg = filedialog.Open(filetypes=ftypes)
        global fil_image
        fil_image = dlg.show()
        global image_flag
        image_flag = True


# creating the child window
class Child(tk.Toplevel):
    def __init__(self):
        super().__init__(root)
        self.ie = None
        self.parser = None
        self.args = self.create_child_window()
        self.person_vehicle_bike_detection_model_online = False
        self.face_detection_model_online = False
        self.age_gender_recognition_model_online = False
        self.emotions_recognition_model_online = False
        self.landmarks_regression_model_online = False
        self.super_resolution_model_online = False
        self.person_reidentification_model_online = False
        self.var = None
        self.v = None

    def create_child_window(self):
        self.title('Нейронные сети/OpenCV')
        self.geometry('1320x900+0+0')

        text_ = tk.Label(self, text='OpenCV:', font="times 13")
        text_.place(x=880, y=80)
        button_begin = ttk.Button(self, text='Начать', command=self.opencv_run)
        button_begin.place(x=1070, y=130)

        text_1 = tk.Label(self, text='Нейронные сети:', font="times 15")
        text_1.place(x=930, y=200)

        # РЕШИТЬ, НУЖНЕН ЛИ checkbtton вообще
        self.v = BooleanVar()
        self.var = BooleanVar()
        ch = Checkbutton(self, text="Распознать возраст/пол", font="times 13", variable=lambda: self.var.set(True))
        ch.place(x=880, y=230)
        button_on = tk.Button(self, text='Включить', command=self.age_gender_begin)
        button_on.place(x=900, y=260)
        button_off = tk.Button(self, text='Выключить', command=lambda: self.v.set(False))
        button_off.place(x=980, y=260)
        che = Checkbutton(self, text="Распознать эмоцию(ии) человека/людей", font="times 13",
                          variable=lambda: self.var.set(True))
        che.place(x=880, y=300)
        button_begin_1 = tk.Button(self, text='Включить', command=self.emotion)
        button_begin_1.place(x=900, y=330)
        button_off = tk.Button(self, text='Выключить', command=lambda: self.v.set(False))
        button_off.place(x=980, y=330)
        self.inf_icon = tk.PhotoImage(file="i.png")
        inf_button = tk.Button(self, image=self.inf_icon, command=self.information_window)
        inf_button.place(x=1265, y=0)

        cheс = Checkbutton(self, text="Увеличить без потери качества", font="times 13",
                           variable=lambda: self.var.set(True))
        cheс.place(x=880, y=370)
        butto = tk.Button(self, text='Включить', command=self.superres)
        butto.place(x=900, y=400)
        butt_2 = tk.Button(self, text='Выключить', command=lambda: self.v.set(False))
        butt_2.place(x=980, y=400)
        self.combobox = ttk.Combobox(self, values=[u"Change of size", u"Flip", u"Gaussian blur",
                                                   u"Black and white filter"])
        self.combobox.current(0)
        self.combobox.place(x=1000, y=80)

        cheсk = Checkbutton(self, text="Распознование лиц/а", font="times 13", variable=lambda: self.var.set(True))
        cheсk.place(x=880, y=435)
        butto_1 = tk.Button(self, text='Включить', command=self.face_detect)
        butto_1.place(x=900, y=470)
        butt_2 = tk.Button(self, text='Выключить', command=lambda: self.v.set(False))
        butt_2.place(x=980, y=470)
        cheсkb = Checkbutton(self, text="5 точек лица", font="times 13", variable=lambda: self.var.set(True))
        cheсkb.place(x=880, y=500)
        butto_4 = tk.Button(self, text='Включить', command=self.landmarks)
        butto_4.place(x=900, y=530)
        butt_5 = tk.Button(self, text='Выключить', command=lambda: self.v.set(False))
        butt_5.place(x=980, y=530)
        cheсkb_1 = Checkbutton(self, text="Повторное распознование лиц", font="times 13",
                               variable=lambda: self.var.set(True))
        cheсkb_1.place(x=880, y=560)
        butto_6 = tk.Button(self, text='Включить', command=self.reidentification)
        butto_6.place(x=900, y=590)
        cheсkb_2 = Checkbutton(self, text="Обнаружить людей/транспортные средства/велосипеды ", font="times 13",
                               variable=lambda: self.var.set(True))
        cheсkb_2.place(x=880, y=620)
        butto_7 = tk.Button(self, text='Включить', command=self.person_vehicle)
        butto_7.place(x=900, y=650)
        self.inf_models_icon = tk.PhotoImage(file="inf.png")
        butt_6 = tk.Button(self, image=self.inf_models_icon, command=self.models)
        butt_6.place(x=900, y=700)
        btn_cancel = ttk.Button(self, text='Закрыть', command=self.destroy)
        btn_cancel.place(x=895, y=750)

        self.grab_set()
        self.focus_set()

        # IECore
        self.ie = IECore()

        # Load the models
        self.parser = argparse.ArgumentParser(description='Run models with OpenVINO')
        self.parser.add_argument('-mFD', dest='face_detection_model', default='face-detection-adas-0001',
                                 help='Path to the model_1')
        self.parser.add_argument('-mLR', dest='landmarks_regression_model', default='landmarks-regression-retail-0009',
                                 help='Path to the model_2')
        self.parser.add_argument('-mAGR', dest='age_gender_recognition_model',
                                 default='age-gender-recognition-retail-0013',
                                 help='Path to the model_3')
        self.parser.add_argument('-mER', dest='emotions_recognition_model', default='emotions-recognition-retail-0003',
                                 help='Path to the model_4')
        self.parser.add_argument('-mSR', dest='super_resolution_model', default='single-image-super-resolution-1033',
                                 help='Path to the model_5')
        self.parser.add_argument('-mPVB', dest='person_vehicle_bike_detection_model',
                                 default='person-vehicle-bike-detection-crossroad-1016', help='Path to the model_6')

        self.parser.add_argument('-mPR', dest='person_reidentification_model',
                                 default='person-reidentification-retail-0107',
                                 help='Path to the model_7')

        return self.parser.parse_args()

    def change_size(self, frame, sc_p):
        scale_percent = sc_p
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        # уменьшаем изображение до подготовленных размеров
        resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
        return resized

    def loc(self, fr):
        cv.imwrite('C:\picture\size.png', fr)
        self.newsize = tk.PhotoImage(file='C:\picture\size.png')
        imshow_ = tk.Label(self, image=self.newsize)
        imshow_.place(x=0, y=0)

    def location(self, frame):
        cv.imwrite('C:\picture\change.png', frame)
        self.new_size = tk.PhotoImage(file='C:\picture\change.png')
        look = tk.Label(self, image=self.new_size)
        h = frame.shape[0] + 50
        look.place(x=0, y=h)

    def check_size_image(self, fr, pr):
        n_s = self.change_size(fr, pr)
        self.loc(n_s)
        return n_s
    # Load&Setup face_detection_model

    def face_detection_model_init(self):
        net_FD = self.ie.read_network(self.args.face_detection_model + '.xml', self.args.face_detection_model + '.bin')
        exec_net_FD = self.ie.load_network(net_FD, 'CPU')
        return net_FD, exec_net_FD

        # Load&Setup landmarks_regression_model

    def landmarks_regression_model_init(self):
        net_LR = self.ie.read_network(self.args.landmarks_regression_model + '.xml',
                                      self.args.landmarks_regression_model + '.bin')
        exec_net_LR = self.ie.load_network(net_LR, 'CPU')
        return net_LR, exec_net_LR

        # Load&Setup age_gender_recognition_model

    def age_gender_recognition_model_init(self):
        net_AGR = self.ie.read_network(self.args.age_gender_recognition_model + '.xml',
                                       self.args.age_gender_recognition_model + '.bin')
        exec_net_AGR = self.ie.load_network(net_AGR, 'CPU')
        return net_AGR, exec_net_AGR

        # Load&Setup emotions_recognition_model

    def emotions_recognition_model_init(self):
        net_ER = self.ie.read_network(self.args.emotions_recognition_model + '.xml',
                                      self.args.emotions_recognition_model + '.bin')
        exec_net_ER = self.ie.load_network(net_ER, 'CPU')
        return net_ER, exec_net_ER

        # Load&Setup super_resolution_model

    def super_resolution_model_init(self):
        net_SR = self.ie.read_network(self.args.super_resolution_model + '.xml',
                                      self.args.super_resolution_model + '.bin')
        return net_SR

        # Load&Setup person_vehicle_bike_detection_model

    def person_vehicle_bike_detection_model_init(self):
        net_PVB = self.ie.read_network(self.args.person_vehicle_bike_detection_model + '.xml',
                                       self.args.person_vehicle_bike_detection_model + '.bin')
        exec_net_PVB = self.ie.load_network(net_PVB, 'CPU')
        return net_PVB, exec_net_PVB

    def person_reidentification_model_init(self):
        net_PR = self.ie.read_network(self.args.person_reidentification_model + '.xml',
                                      self.args.person_reidentification_model + '.bin')
        exec_net_PR = self.ie.load_network(net_PR, 'CPU')
        return net_PR, exec_net_PR
        # person_vehicle_bike_detection_model

    def person_reidentification(self, frame, net_PR, exec_net_PR, net_PVB, exec_net_PVB):
        img = frame

        input_name = next(iter(net_PVB.input_info))
        net_1_width, net_1_height = 512, 512

        rei_input_name = next(iter(net_PR.input_info))
        rei_widht, rei_height = 128, 256

        img_widht, img_height = img.shape[0], img.shape[1]

        base = []
        acc = 0.7
        rei_acc = 200
        rei_conf = 190
        size_base = 30
        n = 0
        resized = cv.resize(img, (net_1_width, net_1_height), interpolation=cv.INTER_CUBIC)
        inp = resized.transpose(2, 0, 1)
        outs = exec_net_PVB.infer({input_name: inp})
        out = next(iter(outs.values()))
        for outt in out[0][0]:
            if outt[1] == 2.0:
                if outt[2] >= acc:
                    x_min = int(outt[3] * img_height)
                    y_min = int(outt[4] * img_widht)
                    x_max = int(outt[5] * img_height)
                    y_max = int(outt[6] * img_widht)
                    cv.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0, 0, 255))
                    reindefication = img[y_min:y_max, x_min:x_max]
                    rei_resized = cv.resize(reindefication, (rei_widht, rei_height), interpolation=cv.INTER_CUBIC)
                    rei_inp = rei_resized.transpose(2, 0, 1)
                    rei_outs = exec_net_PR.infer({rei_input_name: rei_inp})
                    rei_out = next(iter(rei_outs.values()))
                    ind = -1
                    k = 0
                    min_inc = -1
                    if n > 0:
                        for person in base:
                            i_min_inc = -1
                            for rei in person:
                                inc = 0
                                j = 0
                                while j < 256:
                                    inc += abs(rei[j] - rei_out[0][j])
                                    j += 1
                                if inc < i_min_inc or i_min_inc == -1:
                                    i_min_inc = inc
                            if (i_min_inc < min_inc or ind == -1) and i_min_inc < rei_acc:
                                min_inc = i_min_inc
                                ind = k
                            k += 1
                    if n == 0:
                        base.append(rei_out)
                        cv.putText(img, str(n), (x_min, y_max), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 3)
                        n += 1
                    else:
                        if ind == -1:
                            base.append(rei_out)
                            cv.putText(img, str(n), (x_min, y_max), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 3)
                            # пишем индекс = n
                            n += 1
                        else:
                            cv.putText(img, str(ind), (x_min, y_max), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 3)
                            if len(rei_out[0]) < size_base:
                                if min_inc > rei_conf:
                                    base[ind].append(rei_out[0])
                else:
                    break

        return img

    # предназначена для обнаружения людей/транспортных средств/велосипедов
    def person_vehicle_bike_detection(self, frame, net_PVB, exec_net_PVB):

        # Prepare input
        dim = (512, 512)
        resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
        inp = resized.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)

        # Getting data from the network
        input_name = next(iter(net_PVB.input_info))
        outputs = exec_net_PVB.infer({input_name: inp})

        # Processing data
        outs = next(iter(outputs.values()))
        outs = outs[0][0]
        j = 0
        ROI_person = frame
        for out in outs:
            coords = []
            if out[2] == 0.0:
                break
            if out[2] > 0.6:
                x_min = out[3]
                y_min = out[4]
                x_max = out[5]
                y_max = out[6]
                coords.append([x_min, y_min, x_max, y_max])
                coord = coords[0]
                h = frame.shape[0]
                w = frame.shape[1]
                coord = coord * np.array([w, h, w, h])
                coord = coord.astype(np.int32)
                ROI_person = frame[coord[1]:coord[3], coord[0]:coord[2]]
                if out[1] == 2.0:
                    # Drawing a rectangle
                    cv.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), color=(0, 0, 255))
                if out[1] == 1.0:
                    # Drawing a rectangle
                    cv.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), color=(0, 255, 0))
                if out[1] == 0.0:
                    # Drawing a rectangle
                    cv.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), color=(255, 0, 0))

        return frame, ROI_person

        # face_detection_model


    def face_detection(self, frame, net_FD, exec_net_FD, net_AGR, exec_net_AGR, net_ER, exec_net_ER,net_LR, exec_net_LR):

        # Prepare input
        dim = (672, 384)
        resized_img = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
        inp = resized_img.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)

        # Getting data from the network
        input_name = next(iter(net_FD.input_info))
        outputs = exec_net_FD.infer({input_name: inp})

        # Processing data
        outs = next(iter(outputs.values()))
        outs = outs[0][0]
        j = 0
        ROI_face = frame
        for out in outs:
            coords = []
            conf = out[2]
            if conf == 0.0:
                break
            if conf > 0.8:
                x_min = out[3]
                y_min = out[4]
                x_max = out[5]
                y_max = out[6]
                coords.append([x_min, y_min, x_max, y_max])
                coord = coords[0]
                h = frame.shape[0]
                w = frame.shape[1]
                coord = coord * np.array([w, h, w, h])
                coord = coord.astype(np.int32)
                ROI_face = frame[coord[1]:coord[3], coord[0]:coord[2]]

                # Drawing a rectangle
                cv.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), color=(0, 255, 0))

                broken_ROI_face = False

                try:
                    resize = cv.resize(ROI_face, (64, 64))
                except cv.error as err:
                    broken_ROI_face = True
                    print('Invalid frame!')

                # Models that require face_detection_model
                emotion_str = ""
                gender_str = ""

                if broken_ROI_face == False:

                    if self.age_gender_recognition_model_online == True:
                        age, gender_str = self.age_gender_recognition(frame, ROI_face, net_AGR, exec_net_AGR)

                    if self.emotions_recognition_model_online == True:
                        emotion_str = self.emotions_recognition(frame, ROI_face, net_ER, exec_net_ER)

                    if self.landmarks_regression_model_online == True:
                        frame = self.landmarks_regression(frame, ROI_face, net_LR, exec_net_LR)

                    # Put text on the frame
                    if self.emotions_recognition_model_online == True:
                        j = j + 1
                        text_1 = "ID[" + str(j) + "] Emotion: " + emotion_str


                    elif self.age_gender_recognition_model_online == True:
                        text_2 = "Age: " + str(age) + " Gender: " + gender_str

                    if ((coord[1] + coord[3]) / 2) > ((frame.shape[0]) / 2):
                        # cv.rectangle(frame,(coord[0],coord[1]-52),(coord[0]+l,coord[1]),(0,255,0),0)
                        font = cv.FONT_HERSHEY_SIMPLEX
                        if self.emotions_recognition_model_online == True:
                            cv.putText(frame, text_1, (coord[0], coord[1] - 32), font, 0.7, (255, 255, 0), 2,
                                       cv.LINE_AA)
                        elif self.age_gender_recognition_model_online == True:
                            cv.putText(frame, text_2, (coord[0], coord[1] - 6), font, 0.7, (255, 255, 0), 2, cv.LINE_AA)
                    else:
                        # cv.rectangle(frame,(coord[0],coord[3]+52),(coord[0]+l,coord[3]),(0,255,0),0)
                        font = cv.FONT_HERSHEY_SIMPLEX
                        if self.emotions_recognition_model_online == True:
                            cv.putText(frame, text_1, (coord[0], coord[3] + 46), font, 0.7, (255, 255, 0), 2,
                                       cv.LINE_AA)
                        elif self.age_gender_recognition_model_online == True:
                            cv.putText(frame, text_2, (coord[0], coord[3] + 16), font, 0.7, (255, 255, 0), 2,
                                       cv.LINE_AA)

        return frame, ROI_face

        # landmarks_regression_model

    def landmarks_regression(self, frame, ROI_face, net_LR, exec_net_LR):

        # Getting the image size
        ROI_face_height = ROI_face.shape[0]
        ROI_face_width = ROI_face.shape[1]
        difWH = ROI_face_width / ROI_face_height
        difHW = ROI_face_height / ROI_face_width

        # Prepare input
        dim = (48, 48)
        resized = cv.resize(ROI_face, dim, interpolation=cv.INTER_AREA)
        inp = resized.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)

        # Getting data from the network
        input_name = next(iter(net_LR.input_info))
        outs = exec_net_LR.infer({input_name: inp})

        # Processing data
        out = next(iter(outs.values()))

        # Eye1
        x = int(out[0][0][0][0] * ROI_face.shape[0] * difWH)
        y = int(out[0][1][0][0] * ROI_face.shape[1] * difHW)

        cv.circle(ROI_face, (x, y), 5, (0, 0, 255), -1)

        # Eye2
        x1 = int(out[0][2][0][0] * ROI_face.shape[0] * difWH)
        y1 = int(out[0][3][0][0] * ROI_face.shape[1] * difHW)

        cv.circle(ROI_face, (x1, y1), 5, (0, 0, 255), -1)

        # Other points
        x2 = int(out[0][4][0][0] * ROI_face.shape[0] * difWH)
        y2 = int(out[0][5][0][0] * ROI_face.shape[1] * difHW)

        cv.circle(ROI_face, (x2, y2), 5, (0, 0, 255), -1)

        x3 = int(out[0][6][0][0] * ROI_face.shape[0] * difWH)
        y3 = int(out[0][7][0][0] * ROI_face.shape[1] * difHW)

        cv.circle(ROI_face, (x3, y3), 5, (0, 0, 255), -1)

        x4 = int(out[0][8][0][0] * ROI_face.shape[0] * difWH)
        y4 = int(out[0][9][0][0] * ROI_face.shape[1] * difHW)

        cv.circle(ROI_face, (x4, y4), 5, (0, 0, 255), -1)

        return frame

        # age_gender_recognition_model

    def age_gender_recognition(self, frame, ROI_face, net_AGR, exec_net_AGR):

        # Prepare input
        dim = (62, 62)
        resized = cv.resize(ROI_face, dim, interpolation=cv.INTER_AREA)
        inp = resized.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)

        # Getting data from the network
        input_name = next(iter(net_AGR.input_info))
        outs = exec_net_AGR.infer({input_name: inp})

        # Processing data
        out = next(iter(outs.values()))

        age = outs['age_conv3'][0][0][0][0] * 100
        age = round(age)
        gender = np.argmax(outs['prob'])

        if gender == 0:
            gender_str = 'female'
        else:
            gender_str = 'male'

        return age, gender_str

        # emotions_recognition_model

    def emotions_recognition(self, frame, ROI_face, net_ER, exec_net_ER):

        # Prepare input
        dim = (64, 64)
        resized = cv.resize(ROI_face, dim, interpolation=cv.INTER_AREA)
        inp = resized.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)

        # Getting data from the network
        input_name = next(iter(net_ER.input_info))
        outs = exec_net_ER.infer({input_name: inp})

        # Processing data
        out = next(iter(outs.values()))
        i = 0
        imax = i
        maxs = out[0][i][0][0]
        for i in [0, 1, 2, 3]:
            i = i + 1
            if maxs < out[0][i][0][0]:
                maxs = out[0][i][0][0]
                imax = i

        if imax == 0:
            emotion_str = 'neutral'
        if imax == 1:
            emotion_str = 'happy'
        if imax == 2:
            emotion_str = 'sad'
        if imax == 3:
            emotion_str = 'surprise'
        if imax == 4:
            emotion_str = 'anger'

        return emotion_str

        # super_resolution_model

    def super_resolution(self, frame, net_SR):

        img = frame

        inp_h, inp_w = img.shape[0], img.shape[1]
        out_h, out_w = inp_h * 3, inp_w * 3  # Do not change! This is how model works

        # Workaround for reshaping bug
        c1 = net_SR.layers['79/Cast_11815_const']
        c1.blobs['custom'][4] = inp_h
        c1.blobs['custom'][5] = inp_w

        c2 = net_SR.layers['86/Cast_11811_const']
        c2.blobs['custom'][2] = out_h
        c2.blobs['custom'][3] = out_w

        # Reshape network to specific size
        net_SR.reshape({'0': [1, 3, inp_h, inp_w], '1': [1, 3, out_h, out_w]})

        # Load network to device
        exec_net_SR = self.ie.load_network(net_SR, 'CPU')

        # Prepare input
        inp = img.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)
        inp = inp.reshape(1, 3, inp_h, inp_w)
        inp = inp.astype(np.float32)

        # Prepare second input - bicubic resize of first input
        resized_img = cv.resize(img, (out_w, out_h), interpolation=cv.INTER_CUBIC)
        resized = resized_img.transpose(2, 0, 1)
        resized = resized.reshape(1, 3, out_h, out_w)
        resized = resized.astype(np.float32)

        # Getting data from the network
        outs = exec_net_SR.infer({'0': inp, '1': resized})

        # Processing data
        out = next(iter(outs.values()))
        out = out.reshape(3, out_h, out_w).transpose(1, 2, 0)
        out = np.clip(out * 255, 0, 255)
        out = np.ascontiguousarray(out).astype(np.uint8)

        return img, resized_img, out

        self.title('Нейронные сети/OpenCV')
        self.geometry('1300x900+0+0')

        text_ = tk.Label(self, text='OpenCV:', font="times 13")
        text_.place(x=880, y=80)
        btn_ok = ttk.Button(self, text='Начать', command=self.opencv_begin)
        btn_ok.place(x=1070, y=130)
        text_1 = tk.Label(self, text='Нейронные сети:', font="times 15")
        text_1.place(x=930, y=200)
        self.v = BooleanVar()
        self.var = BooleanVar()
        ch = tk.Label(self, text="Распознать возраст/пол", font="times 13")
        ch.place(x=880, y=230)
        butt_ = tk.Button(self, text='Включить', command=self.age_gender_begin)
        butt_.place(x=900, y=260)
        butt_1 = tk.Button(self, text='Выключить', command=lambda: self.v.set(False))
        butt_1.place(x=980, y=260)
        che = Checkbutton(self, text="Распознать эмоцию(ии) человека/людей", font="times 13", variable=self.var)
        che.place(x=880, y=300)
        But = tk.Button(self, text='Включить', command=self.emotion)
        But.place(x=900, y=330)
        butt_2 = tk.Button(self, text='Выключить', command=lambda: self.v.set(False))
        butt_2.place(x=980, y=330)
        self.inf_icon = tk.PhotoImage(file="i.png")
        b = tk.Button(self, image=self.inf_icon, command=self.information_window)
        b.place(x=1245, y=0)
        cheс = Checkbutton(self, text="Увеличить без потери качества", font="times 13", variable=self.var)
        cheс.place(x=880, y=370)
        butto = tk.Button(self, text='Включить', command=self.superres)
        butto.place(x=900, y=400)
        butt_2 = tk.Button(self, text='Выключить', command=lambda: self.v.set(False))
        butt_2.place(x=980, y=400)
        self.combobox = ttk.Combobox(self, values=[u"Change of size", u"Flip", u"Gaussian blur",
                                                   u"Black and white filter"])
        self.combobox.current(0)
        self.combobox.place(x=1000, y=80)
        cheсk = Checkbutton(self, text="Распознование лиц/а", variable=self.var)
        cheсk.place(x=880, y=435)
        butto_ = tk.Button(self, text='Включить', command=self.face_detect)
        butto_.place(x=900, y=470)
        butt_2 = tk.Button(self, text='Выключить', command=lambda: self.v.set(False))
        butt_2.place(x=980, y=470)
        cheсkb = Checkbutton(self, text="5 точек лица", font="times 13", variable=self.var)
        cheсkb.place(x=880, y=500)
        Butto_ = tk.Button(self, text='Включить')
        Butto_.place(x=900, y=530)
        Butt_2 = tk.Button(self, text='Выключить', command=lambda: self.v.set(False))
        Butt_2.place(x=980, y=530)
        self.ima = tk.PhotoImage(file="inf.png")
        bu = tk.Button(self, image=self.ima, command=self.inf_models)
        bu.place(x=880, y=600)
        tex_3 = tk.Label(self, text='OpCV:', font="times 13")
        tex_3.place(x=0, y=0)
        btn_cancel = ttk.Button(self, text='Закрыть', command=self.destroy)
        btn_cancel.place(x=895, y=700)
        self.grab_set()
        self.focus_set()

    def face_detect(self):
        net_FD, exec_net_FD = self.face_detection_model_init()
        net_AGR, exec_net_AGR = self.age_gender_recognition_model_init()
        net_ER, exec_net_ER = self.emotions_recognition_model_init()
        net_LR, exec_net_LR = self.landmarks_regression_model_init()

        self.face_detection_model_online = True

        if video_flag == True:
            cap = cv.VideoCapture(fil_video)
            ret, frame = cap.read()
        elif webcam_flag == True:
            cap = cv.VideoCapture(0)
        elif image_flag == True:
            picture = cv.imread(fil_image)
            width = picture.shape[1]
            if 600 <= width and width < 1000:
                n_size = self.check_size_image(picture, 30)
                picture = n_size
            elif width > 1000:
                n_size = self.check_size_image(picture, 10)
                picture = n_size
            elif width < 600:
                self.img = tk.PhotoImage(file=fil_image)
                look = tk.Label(self, image=self.img)
                look.place(x=0, y=0)
        if image_flag == True:
            if (self.face_detection_model_online == True) or (self.age_gender_recognition_model_online == True) or (
                    self.emotions_recognition_model_online == True) or (self.landmarks_regression_model_online == True):
                frame, ROI_face = self.face_detection(picture, net_FD, exec_net_FD,net_AGR, exec_net_AGR,net_ER, exec_net_ER,net_LR, exec_net_LR)
                cv.imwrite('C:\picture\change.png', frame)
                self.new_size = tk.PhotoImage(file='C:\picture\change.png')
                look = tk.Label(self, image=self.new_size)
                h = frame.shape[0] + 20
                look.place(x=0, y=h)
        elif video_flag == True or webcam_flag == True:
            while (cap.isOpened()):

                # Reading the frame
                ret, frame = cap.read()
                if cv.waitKey(1) & 0xFF == ord('q') or ret == False:
                    break
                # Face_detection, age_gender, emotions, landmarks models
                if (self.face_detection_model_online == True) or (self.age_gender_recognition_model_online == True) or (
                        self.emotions_recognition_model_online == True) or (
                        self.landmarks_regression_model_online == True):
                    frame, ROI_face = self.face_detection(frame, net_FD, exec_net_FD, net_AGR, exec_net_AGR, net_ER, exec_net_ER, net_LR, exec_net_LR)
                cv.imshow('Frame', frame)
            cap.release()
            cv.destroyAllWindows()
        else:
            messagebox.showinfo("ОШИБКА", "Загрузите фото/видео или включите веб камеру!!!")
    def emotion(self):
        net_FD, exec_net_FD = self.face_detection_model_init()
        net_AGR, exec_net_AGR = self.age_gender_recognition_model_init()
        net_ER, exec_net_ER = self.emotions_recognition_model_init()
        net_LR, exec_net_LR = self.landmarks_regression_model_init()

        self.face_detection_model_online = True
        self.emotions_recognition_model_online = True

        if video_flag == True:
            cap = cv.VideoCapture(fil_video)
            ret, frame = cap.read()
        elif webcam_flag == True:
            cap = cv.VideoCapture(0)
        elif image_flag == True:
            picture = cv.imread(fil_image)
            width = picture.shape[1]
            if 600 <= width and width < 1000:
                n_size = self.check_size_image(picture, 30)
                picture = n_size
            elif width > 1000:
                n_size = self.check_size_image(picture, 10)
                picture = n_size
            elif width < 600:
                self.img = tk.PhotoImage(file=fil_image)
                look = tk.Label(self, image=self.img)
                look.place(x=0, y=0)

        if image_flag == True:
            if (self.face_detection_model_online == True) or (self.age_gender_recognition_model_online == True) or (
                    self.emotions_recognition_model_online == True) or (self.landmarks_regression_model_online == True):
                frame, ROI_face = self.face_detection(picture, net_FD, exec_net_FD, net_AGR, exec_net_AGR, net_ER,
                                                      exec_net_ER, net_LR, exec_net_LR)
                cv.imwrite('C:\picture\change.png', frame)
                self.new_size = tk.PhotoImage(file='C:\picture\change.png')
                look = tk.Label(self, image=self.new_size)
                h = frame.shape[0] + 20
                look.place(x=0, y=h)
        elif video_flag == True or webcam_flag == True:
            while (cap.isOpened()):
                # Reading the frame
                ret, frame = cap.read()
                if cv.waitKey(1) & 0xFF == ord('q') or ret == False:
                    break

                # Face_detection, age_gender, emotions, landmarks models
                if (self.face_detection_model_online == True) or (self.age_gender_recognition_model_online == True) or (
                        self.emotions_recognition_model_online == True) or (
                        self.landmarks_regression_model_online == True):
                    frame, ROI_face = self.face_detection(frame, net_FD, exec_net_FD, net_AGR, exec_net_AGR, net_ER,
                                                          exec_net_ER, net_LR, exec_net_LR)
                cv.imshow('Frame', frame)
            cap.release()
            cv.destroyAllWindows()
        else:
            messagebox.showinfo("ОШИБКА", "Загрузите фото/видео или включите веб камеру!!!")
    def superres(self):
        net_SR = self.super_resolution_model_init()
        self.super_resolution_model_online = True
        if video_flag == True:
            cap = cv.VideoCapture(fil_video)
            ret, frame = cap.read()
        elif webcam_flag == True:
            cap = cv.VideoCapture(0)
        elif image_flag == True:
            picture = cv.imread(fil_image)
        if self.super_resolution_model_online == True:
            if image_flag == True:
                img, resized_img, out = self.super_resolution(picture, net_SR)
                cv.imshow('Source image', img)
                cv.imshow('Bicubic interpolation', resized_img)
                cv.imshow('Super resolution', out)
                cv.waitKey(0)
            elif video_flag == True or webcam_flag == True:
                while (cap.isOpened()):
                    # Reading the frame
                    ret, frame = cap.read()
                    if cv.waitKey(1) & 0xFF == ord('q') or ret == False:
                        break
                    img, resized_img, out = self.super_resolution(frame, net_SR)
                    cv.imshow('Source image', img)
                    cv.imshow('Bicubic interpolation', resized_img)
                    cv.imshow('Super resolution', out)
                cap.release()
                cv.destroyAllWindows()
            else:
                messagebox.showinfo("ОШИБКА", "Загрузите фото/видео или включите веб камеру!!!")
    def reidentification(self):
        net_PR, exec_net_PR = self.person_reidentification_model_init()
        net_PVB, exec_net_PVB = self.person_vehicle_bike_detection_model_init()
        self.person_vehicle_bike_detection_model_online = True
        self.person_reidentification_model_online = True

        if video_flag == True:
            cap = cv.VideoCapture(fil_video)
            ret, frame = cap.read()
        elif webcam_flag == True:
            cap = cv.VideoCapture(0)
        elif image_flag == True:
            picture = cv.imread(fil_image)
            width = picture.shape[1]
            if 600 <= width and width < 1000:
                n_size = self.check_size_image(picture, 30)
                picture = n_size
            elif width > 1000:
                n_size = self.check_size_image(picture, 10)
                picture = n_size
            elif width < 600:
                self.img = tk.PhotoImage(file=fil_image)
                look = tk.Label(self, image=self.img)
                look.place(x=0, y=0)

        if self.person_vehicle_bike_detection_model_online == True and self.person_reidentification_model_online == True:
            if image_flag == True:
                img = self.person_reidentification(picture, net_PR, exec_net_PR, net_PVB, exec_net_PVB)
                cv.imwrite('C:\picture\change.png', img)
                self.new_size = tk.PhotoImage(file='C:\picture\change.png')
                look = tk.Label(self, image=self.new_size)
                h = img.shape[0] + 20
                look.place(x=0, y=h)
            elif video_flag == True or webcam_flag == True:
                while (cap.isOpened()):
                    # Reading the frame
                    ret, frame = cap.read()
                    if cv.waitKey(1) & 0xFF == ord('q') or ret == False:
                        break
                    res = self.person_reidentification(frame, net_PR, exec_net_PR, net_PVB, exec_net_PVB)
                    cv.imshow('frame', res)
                cap.release()
                cv.destroyAllWindows()
            else:
                messagebox.showinfo("ОШИБКА", "Загрузите фото/видео или включите веб камеру!!!")
    def landmarks(self):
        net_FD, exec_net_FD = self.face_detection_model_init()
        net_AGR, exec_net_AGR = self.age_gender_recognition_model_init()
        net_ER, exec_net_ER = self.emotions_recognition_model_init()
        net_LR, exec_net_LR = self.landmarks_regression_model_init()

        self.face_detection_model_online = True
        self.landmarks_regression_model_online = True

        if video_flag == True:
            cap = cv.VideoCapture(fil_video)
            ret, frame = cap.read()
        elif webcam_flag == True:
            cap = cv.VideoCapture(0)
        elif image_flag == True:
            picture = cv.imread(fil_image)
            width = picture.shape[1]
            if 600 <= width and width < 1000:
                n_size = self.check_size_image(picture, 30)
                picture = n_size
            elif width > 1000:
                n_size = self.check_size_image(picture, 10)
                picture = n_size
            elif width < 600:
                self.img = tk.PhotoImage(file=fil_image)
                look = tk.Label(self, image=self.img)
                look.place(x=0, y=0)

        if image_flag == True:
            if (self.face_detection_model_online == True) or (self.age_gender_recognition_model_online == True) or (
                    self.emotions_recognition_model_online == True) or (self.landmarks_regression_model_online == True):
                frame, ROI_face = self.face_detection(picture, net_FD, exec_net_FD, net_AGR, exec_net_AGR, net_ER,
                                                      exec_net_ER, net_LR, exec_net_LR)
                cv.imwrite('C:\picture\change.png', frame)
                self.new_size = tk.PhotoImage(file='C:\picture\change.png')
                look = tk.Label(self, image=self.new_size)
                h = frame.shape[0] + 20
                look.place(x=0, y=h)
        elif video_flag == True or webcam_flag == True:
            while (cap.isOpened()):
                # Reading the frame
                ret, frame = cap.read()
                if cv.waitKey(1) & 0xFF == ord('q') or ret == False:
                    break

                # Face_detection, age_gender, emotions, landmarks models
                if (self.face_detection_model_online == True) or (self.age_gender_recognition_model_online == True) or (
                        self.emotions_recognition_model_online == True) or (
                        self.landmarks_regression_model_online == True):
                    frame, ROI_face = self.face_detection(frame, net_FD, exec_net_FD, net_AGR, exec_net_AGR, net_ER,
                                                          exec_net_ER, net_LR, exec_net_LR)
                cv.imshow('Frame', frame)
            cap.release()
            cv.destroyAllWindows()
        else:
            messagebox.showinfo("ОШИБКА", "Загрузите фото/видео или включите веб камеру!!!")

    def person_vehicle(self):
        net_PVB, exec_net_PVB = self.person_vehicle_bike_detection_model_init()
        self.person_vehicle_bike_detection_model_online = True
        if video_flag == True:
            cap = cv.VideoCapture(fil_video)
            ret, frame = cap.read()
        elif webcam_flag == True:
            cap = cv.VideoCapture(0)
        elif image_flag == True:
            picture = cv.imread(fil_image)
            width = picture.shape[1]
            if 600 <= width and width < 1000:
                n_size = self.check_size_image(picture, 30)
                picture = n_size
            elif width > 1000:
                n_size = self.check_size_image(picture, 10)
                picture = n_size
            elif width < 600:
                self.img = tk.PhotoImage(file=fil_image)
                look = tk.Label(self, image=self.img)
                look.place(x=0, y=0)
        if self.person_vehicle_bike_detection_model_online == True:
            if image_flag == True:
                frame, ROI_person = self.person_vehicle_bike_detection(picture, net_PVB, exec_net_PVB)
                cv.imwrite('C:\picture\change.png', frame)
                self.new_size = tk.PhotoImage(file='C:\picture\change.png')
                look = tk.Label(self, image=self.new_size)
                h = frame.shape[0] + 20
                look.place(x=0, y=h)
            elif video_flag == True or webcam_flag == True:
                while (cap.isOpened()):
                    # Reading the frame
                    ret, frame = cap.read()
                    if cv.waitKey(1) & 0xFF == ord('q') or ret == False:
                        break
                    frame, ROI_person = self.person_vehicle_bike_detection(frame, net_PVB, exec_net_PVB)
                    cv.imshow('Frame', frame)
                cap.release()
                cv.destroyAllWindows()
            else:
                messagebox.showinfo("ОШИБКА", "Загрузите фото/видео или включите веб камеру!!!")
    def age_gender_begin(self):
        net_FD, exec_net_FD = self.face_detection_model_init()
        net_AGR, exec_net_AGR = self.age_gender_recognition_model_init()
        net_ER, exec_net_ER = self.emotions_recognition_model_init()
        net_LR, exec_net_LR = self.landmarks_regression_model_init()

        self.age_gender_recognition_model_online = True
        self.face_detection_model_online = True

        if video_flag == True:
            cap = cv.VideoCapture(fil_video)
            ret, frame = cap.read()
        elif webcam_flag == True:
            cap = cv.VideoCapture(0)
        elif image_flag == True:
            picture = cv.imread(fil_image)
            width = picture.shape[1]
            if 600 <= width and width < 1000:
                n_size = self.check_size_image(picture, 30)
                picture = n_size
            elif width > 1000:
                n_size = self.check_size_image(picture, 10)
                picture = n_size
            elif width < 600:
                self.img = tk.PhotoImage(file=fil_image)
                look = tk.Label(self, image=self.img)
                look.place(x=0, y=0)
        if image_flag == True:
            if (self.face_detection_model_online == True) or (self.age_gender_recognition_model_online == True) or (
                    self.emotions_recognition_model_online == True) or (self.landmarks_regression_model_online == True):
                frame, ROI_face = self.face_detection(picture, net_FD, exec_net_FD, net_AGR, exec_net_AGR, net_ER,
                                                      exec_net_ER, net_LR, exec_net_LR)
                cv.imwrite('C:\picture\change.png', frame)
                self.new_size = tk.PhotoImage(file='C:\picture\change.png')
                look = tk.Label(self, image=self.new_size)
                h = frame.shape[0] + 20
                look.place(x=0, y=h)
        elif video_flag == True or webcam_flag == True:
            while (cap.isOpened()):

                # Reading the frame
                ret, frame = cap.read()
                if cv.waitKey(1) & 0xFF == ord('q') or ret == False:
                    break
                # Face_detection, age_gender, emotions, landmarks models
                if (self.face_detection_model_online == True) or (self.age_gender_recognition_model_online == True) or (
                        self.emotions_recognition_model_online == True) or (
                        self.landmarks_regression_model_online == True):
                    frame, ROI_face = self.face_detection(frame, net_FD, exec_net_FD, net_AGR, exec_net_AGR, net_ER,
                                                          exec_net_ER, net_LR, exec_net_LR)
                cv.imshow('Frame', frame)
            cap.release()
            cv.destroyAllWindows()
        else:
            messagebox.showinfo("ОШИБКА", "Загрузите фото/видео или включите веб камеру!!!")

    def information_window(self):
        Child_2()

    def models(self):
        Child_1()

    def opencv_run(self):

        res = self.combobox.get()
        if video_flag == True:
            cap = cv.VideoCapture(fil_video)
            ret, frame = cap.read()
        if image_flag == True:
            im = cv.imread(fil_image)
            width = im.shape[1]
            if 600<=width and width < 1000:
                n_size = self.check_size_image(im, 30)
                im = n_size
            elif width > 1000:
                n_size = self.check_size_image(im, 10)
                im = n_size
            elif width<600:
                self.img = tk.PhotoImage(file=fil_image)
                look = tk.Label(self, image=self.img)
                look.place(x=0, y=0)

        if webcam_flag == True:
            cap = cv.VideoCapture(0)
        # сделать уменьшение размера, если изображение большое
        if res == "Change of size":
            if image_flag == True:
                new_size = self.change_size(im, 20)
                cv.imwrite('C:\picture\change.png', new_size)
                self.new_image = tk.PhotoImage(file='C:\picture\change.png')
                look = tk.Label(self, image=self.new_image)
                w = im.shape[1] + 100
                look.place(x=w, y=0)
            elif video_flag == True or webcam_flag == True:
                while (cap.isOpened()):
                    ret, img = cap.read()
                    if cv.waitKey(1) & 0xFF == ord('q') or ret == False:
                        break
                    n_s = self.change_size(img, 20)
                    cv.imshow('Source image', img)
                    cv.imshow('new size', n_s)
                cap.release()
                cv.destroyAllWindows()
            else:
                messagebox.showinfo("ОШИБКА", "Загрузите фото/видео или включите веб камеру!!!")
        elif res == "Flip":
            if image_flag == True:
                flip_img_y = cv.flip(im, 1)
                flip_img_x = cv.flip(im, 0)
                flip_img_xy = cv.flip(im, -1)
                cv.imwrite('C:\picture\_flip_y.png', flip_img_y)
                self.flip_y = tk.PhotoImage(file='C:\picture\_flip_y.png')
                flip_look_y = tk.Label(self, image=self.flip_y)
                w_y = im.shape[1] + 50
                flip_look_y.place(x=w_y, y=0)

                cv.imwrite('C:\picture\_flip_x.png', flip_img_x)
                self.flip_x = tk.PhotoImage(file='C:\picture\_flip_x.png')
                flip_look_x = tk.Label(self, image=self.flip_x)
                w_x = im.shape[0] + 50
                flip_look_x.place(x=0, y=w_x)

                cv.imwrite('C:\picture\_flip_xy.png', flip_img_xy)
                self.flip_xy = tk.PhotoImage(file='C:\picture\_flip_xy.png')
                flip_look_xy = tk.Label(self, image=self.flip_xy)
                w_xy = im.shape[1] + 50
                h_xy = im.shape[0] + 50
                flip_look_xy.place(x=w_xy, y=h_xy)

            elif video_flag == True or webcam_flag == True:
                while (
                cap.isOpened()):  # функция изопенд будет возварщать каждый раз тру,пока не дойдет до конца файла/выход из цикла будет,когда функция вернет фолс,либо по брейку
                    ret, img = cap.read()  # функция рид возвращает лбо тру,либо фолс/это значение запишем в переменнуб ret, а текущий кадр запишем в перменную фрейм
                    if cv.waitKey(1) & 0xFF == ord('q') or ret == False:  # функция рид возвращает лбо тру,либо фолс/это значение запишем в переменнуб ret, а текущий кадр запишем в перменную фрейм
                        break
                    flip_img_y = cv.flip(img, 1)
                    flip_img_x = cv.flip(img, 0)
                    flip_img_xy = cv.flip(img, -1)
                    cv.imshow('Source image', img)
                    cv.imshow('flip around the y-axis', flip_img_y)
                    cv.imshow('flip around the x-axis', flip_img_x)
                    cv.imshow('flip around the xy-axis', flip_img_xy)
                cap.release()
                cv.destroyAllWindows()
            else:
                messagebox.showinfo("ОШИБКА", "Загрузите фото/видео или включите веб камеру!!!")
        elif res == "Gaussian blur":
            if image_flag == True:
                gauss_blur = cv.GaussianBlur(im, (9, 9), 10)
                self.location(gauss_blur)
            elif video_flag == True or webcam_flag == True:
                while (cap.isOpened()):
                    ret, img = cap.read()
                    if cv.waitKey(1) & 0xFF == ord('q') or ret == False:
                        break
                    gauss_blur = cv.GaussianBlur(img, (11, 11), 10)
                    cv.imshow('Source image', img)
                    cv.imshow('blur', gauss_blur)
                cap.release()  # освобождает оперативную память, занятую переменной cap.
                cv.destroyAllWindows()  # (закрывает все открытые в скрипте окна).
            else:
                messagebox.showinfo("ОШИБКА", "Загрузите фото/видео или включите веб камеру!!!")
        elif res == "Black and white filter":
            if image_flag == True:
                gray_image = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                self.location(gray_image)
            elif video_flag == True or webcam_flag == True:
                while (cap.isOpened()):
                    ret, img = cap.read()
                    if cv.waitKey(1) & 0xFF == ord('q') or ret == False:
                        break
                    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    cv.imshow('source video', img)
                    cv.imshow('gray video', gray_img)
                cap.release()
                cv.destroyAllWindows()
            else:
                messagebox.showinfo("ОШИБКА", "Загрузите фото/видео или включите веб камеру!!!")


class Child_2(tk.Toplevel):
    def __init__(self):
        super().__init__(root)
        self.inf_about_project()

    def inf_about_project(self):
        self.title('Информация о проекте/команде')
        self.geometry('800x550+300+50')
        self.resizable(False, False)

        text_ = tk.Label(self,
                         text="Наша команда Orange Duck разработала приложение способное обрабатывать изображения,",
                         font="times 13")
        text_.place(x=0, y=0)
        text_1 = tk.Label(self,
                          text="видео ролики, а также изображения, полученные с web-камеры. Функционалом программы являются:",
                          font="times 13")
        text_1.place(x=0, y=25)
        text_2 = tk.Label(self, text="1. Изменение размера", font="times 13")
        text_2.place(x=0, y=50)
        text_3 = tk.Label(self, text="2. Поворот изображения по разным осям", font="times 13")
        text_3.place(x=0, y=75)
        text_4 = tk.Label(self, text="3. Размытие по Гауссу", font="times 13")
        text_4.place(x=0, y=100)
        text_5 = tk.Label(self, text="4. Черно-белый фильтр", font="times 13")
        text_5.place(x=0, y=125)
        text_6 = tk.Label(self, text="5. Увеличение изображения без потери качества", font="times 13")
        text_6.place(x=0, y=150)
        text_7 = tk.Label(self, text="6. Распознавание на изображении человека/машины/велосипеда", font="times 13")
        text_7.place(x=0, y=175)
        text_8 = tk.Label(self, text="7. Идентификация человека", font="times 13")
        text_8.place(x=0, y=200)
        text_9 = tk.Label(self, text="8. Выделение пяти точек лица", font="times 13")
        text_9.place(x=0, y=225)
        Text_ = tk.Label(self, text="9. Определение возраста/пола человека", font="times 13")
        Text_.place(x=0, y=250)
        Text_1 = tk.Label(self, text="10. Определение эмоций человека", font="times 13")
        Text_1.place(x=0, y=275)
        Text_2 = tk.Label(self, text="11. Распознавание лица человека", font="times 13")
        Text_2.place(x=0, y=300)
        Text_3 = tk.Label(self,
                          text="Все эти методы были реализованы с использованием стандартных библиотек OpenCV и обученных",
                          font="times 13")
        Text_3.place(x=0, y=325)
        Text_4 = tk.Label(self,
                          text="нейросетей на платформе OpenVINO (версия 2020.4) от компании Intel. Программа была написана на языке",
                          font="times 13")
        Text_4.place(x=0, y=350)
        Text_5 = tk.Label(self, text="программирования Python (версия 3.7.4).", font="times 13")
        Text_5.place(x=0, y=375)
        Text_6 = tk.Label(self, text="В состав команды входят:", font="times 13")
        Text_6.place(x=0, y=400)
        Text_6 = tk.Label(self, text="Артём Савкин, Егор Тимофеев, Божко Мария, Ксения Сторожева", font="times 13")
        Text_6.place(x=0, y=425)
        btn_cancel = ttk.Button(self, text='Закрыть', command=self.destroy)
        btn_cancel.place(x=300, y=470)
        self.grab_set()
        self.focus_set()
        self.wait_window()


class Child_1(tk.Toplevel):
    def __init__(self):
        super().__init__(root)
        self.inf_about_models()

    def inf_about_models(self):
        self.title('Информация о моделях')
        self.geometry('1200x500+200+200')
        self.resizable(False, False)

        text_4 = tk.Label(self, text="О моделях:", font="times 15")
        text_4.place(x=0, y=0)
        text_3 = tk.Label(self, text="1. age-gender-recognition-retail-0013:", font="times 13")
        text_3.place(x=0, y=30)
        lab = tk.Label(self,
                       text="Сеть для одновременного распознавания возраста/пола, способна установить  возраст людей в диапазоне [18, 75] лет.",
                       font="times 13")
        lab.place(x=0, y=52)
        label = tk.Label(self, text="Она не применима для детей, поскольку их лиц не было в обучающей выборке.",
                         font="times 13")
        label.place(x=0, y=72)
        text_5 = tk.Label(self, text="2. face-detection-adas-0001:", font="times 13")
        text_5.place(x=0, y=102)
        text_6 = tk.Label(self, text="Детектирование лиц", font="times 13")
        text_6.place(x=0, y=122)
        text_8 = tk.Label(self, text="3. emotions-recognition-retail-0003:", font="times 13")
        text_8.place(x=0, y=152)
        text_9 = tk.Label(self,
                          text="Полностью сверточная сеть для распознавания пяти эмоций: «нейтральный», «счастливый», «грустный», «удивленый», «гнев».",
                          font="times 13")
        text_9.place(x=0, y=172)
        text_1 = tk.Label(self, text="4. landmarks-regression-retail-0009:", font="times 13")
        text_1.place(x=0, y=202)
        text_2 = tk.Label(self, text="Модель предсказывает пять ориентиров на лице: два глаза, нос и два уголка губ.",
                          font="times 13")
        text_2.place(x=0, y=222)
        text_11 = tk.Label(self, text="5. person-vehicle-bike-detection-crossroad-1016:", font="times 13")
        text_11.place(x=0, y=252)
        text_10 = tk.Label(self,
                           text="Сеть на базе SSD предназначена для обнаружения людей/транспортных средств/велосипедов в приложениях охранного видеонаблюдения.",
                           font="times 13")
        text_10.place(x=0, y=272)
        TEXT_ = tk.Label(self, text="Работает в различных сценах и погодных условиях/условиях освещения.",
                         font="times 13")
        TEXT_.place(x=0, y=292)
        TEXT_1 = tk.Label(self, text="6. single-image-super-resolution-1033:", font="times 13")
        TEXT_1.place(x=0, y=322)
        TEXT_2 = tk.Label(self,
                          text="Подход, основанный на внимании к сверхразрешению одного изображения, но с уменьшенным числом каналов и изменениями в сетевой архитектуре.",
                          font="times 13")
        TEXT_2.place(x=0, y=348)
        TEXT_3 = tk.Label(self, text="Он увеличивает разрешение входного изображения в 3 раза.", font="times 13")
        TEXT_3.place(x=0, y=368)
        TEXT_4 = tk.Label(self, text="7. person-reidentification-retail-0107:", font="times 13")
        TEXT_4.place(x=0, y=392)
        TEXT_5 = tk.Label(self,
                          text="Это модель повторной идентификации человека. Найденные объекты регестрируются в базе данных.",
                          font="times 13")
        TEXT_5.place(x=0, y=416)
        btn_cancel = ttk.Button(self, text='Закрыть', command=self.destroy)
        btn_cancel.place(x=1000, y=440)
        self.grab_set()
        self.focus_set()
        self.wait_window()


if __name__ == "__main__":
    root = tk.Tk()
    app = Main(root)
    app.pack()
    root.title("Orange Ducks Team")
    root.geometry("900x300+300+200")
    root.resizable(False, False)
    root.mainloop()
