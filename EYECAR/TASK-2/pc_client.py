import socket
import cv2
import beholder
import numpy as np
import threading
from enum import IntEnum
import time

HOME_TEST = True
DRAW_DETECTION = True

# ===================== Детектирование Светофоров =====================

def draw_boxes(image, boxes, labels, confidences, class_ids, idxs, color = (0, 255, 0)):
    if len(idxs) > 0:
        for i in idxs.flatten():
            left, top = boxes[i][0], boxes[i][1]
            width, height = boxes[i][2], boxes[i][3]

            cv2.rectangle(image, (left, top), (left + width, top + height), color)
            label = "%s: %.2f" % (labels[class_ids[i]], confidences[i])
            cv2.putText(image, label, (left, top + height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
    return image

def make_prediction(net, layer_names, frame, \
        conf_threshold = 0.8, nms_threshold = 0.3, inp_size = (256, 256)):
    boxes = []
    confidences = []
    class_ids = []
    frame_height, frame_width = frame.shape[:2]
   
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, inp_size, swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    for output in outputs:
        for detection in output:            
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
           
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)

                left = int(center_x - (width / 2))
                top = int(center_y - (height / 2))
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return boxes, confidences, class_ids, idxs

# =====================================================================

if not HOME_TEST:
    HOST = "192.168.4.1"  # адрес беспилотного автомобиля, на нём запущен сервер, передающий кадры
    #  HOST = 'localhost'
    PORT = 1080  # порт для передачи команд Raspberry Pi

    beholder_client = beholder.Client(zmq_host=HOST, # создаём и настраиваем клиент, принимающий кадры от Raspberry Pi
                            # zmq_host="192.168.1.145",
                            zmq_port=12345,
                            rtp_host="192.168.4.4",  # Адрес ПК в сети беспилотника. Адрес где мы будем принимать кадры
                            # rtp_host="10.205.1.185",
                            rtp_port=5000,
                            rtcp_port=5001,
                            device="/dev/video0",  # видеокамера, с которой мы принимаем кадры
                            # width=1920,
                            # height=1080,
                            width=640,  # ширина кадра
                            height=480,  # высот кадра
                            # width=640,
                            # height=480,
                            framerate=30,  # частота кадров
                            encoding=beholder.Encoding.MJPEG,  #MJPEG,    #H264
                            limit=1)  # длина очереди кадров на ПК, рекомендуется установить значение 1

    beholder_client.start()  # Запускаем приём кадров, очередь кадров начинает наполнятся
    # Если вы собираетесь выполнять длинные операции, например, чтение нейронной сети с диска, выполните их до старта клиента
    # Если клиент не будет читать кадры в течении 30 секунд, то сервер прервёт передачу видеопотока.

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # определяем сокет, на который будем передавть команды
    sock.connect((HOST, PORT))  # подключаемсся к нему

def send_msg(msg):  # функция отправляющая строку на Raspberri Pi
    print(f'Sending message ["{msg}"]...', end='')
    msg = msg.encode('utf-8')
    print('done')
    sock.sendall(msg)

def set_speed(speed):  # Функция отправляющая Raspberri Pi значение скорости
    msg = f'SPEED:{speed}'
    send_msg(msg)


def set_angle(angle):  # Функция отправляющая Raspberri Pi значение угла поворота
    msg = f'ANGLE:{angle}'
    send_msg(msg)


def drop_cargo():  # Функция отправляющая Raspberri Pi команду "опрокинуть кузов"
    msg = f'CARGO:DROP'
    send_msg(msg)


def receive_msg(symbols=100):  # Функция читающая данные от Raspberri Pi;  !!!В программе не используется!!!
    print('Receiving message...', end='')
    data = sock.recv(symbols)
    print('done.')
    data = data.decode('utf-8')
    return data

STOP_SPEED = 1500
STD_ANGLE = 90

def stop():
    set_speed(STOP_SPEED)
    set_angle(STD_ANGLE)

ESCAPE = 27
SPASE = 32

STOP_SPEED = 1500
STD_ANGLE = 90

key = 1
speed = 1435  # 1500 - стоп

if not HOME_TEST:
    stop()

CONFIG_FILE = '../yolov4-tiny-usr.cfg'
WEIGHT_FILE = '../yolov4-tiny_last.weights'
FPS = False

trl_image = []
trl_output_image = []
trl_has = True

class TRLSignal(IntEnum):
    flashing_green = 6
    red_yellow = 5
    yellow = 4
    red = 3
    off = 2
    green = 1
    none = 0
trl_signal = TRLSignal.none
timing_signals = { TRLSignal.none: 0, TRLSignal.green: 0, TRLSignal.off: 0, \
        TRLSignal.red: 0, TRLSignal.yellow: 0, TRLSignal.red_yellow: 0, TRLSignal.flashing_green: 0}

def print_timing():
    global timing_signals
    names_signal = { TRLSignal.red: 'red', TRLSignal.yellow: 'yellow', \
        TRLSignal.green: 'green', TRLSignal.red_yellow: 'red and yellow', TRLSignal.flashing_green: 'flashing green', }

    for key in [TRLSignal.red,  TRLSignal.yellow, \
            TRLSignal.green, TRLSignal.red_yellow, TRLSignal.flashing_green]:
        if not (key in timing_signals): 
            continue

        print("{:<20} {}".format(names_signal[key], '--> ' + str(timing_signals[key])))

labels = ['green', 'off', 'red', 'yellow', 'red+yellow', 'flashing green']

net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHT_FILE)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

frame_time = 0
timer_start = 0
flashing_green = False

start_signal = TRLSignal.none
change_start = 0

if HOME_TEST:
    cap = cv2.VideoCapture('../tr4.mp4') 

while key != ESCAPE and trl_has:
    status = True
    frame = None
    if not HOME_TEST:
        status, frame = beholder_client.get_frame(0.25)  # читаем кадр из очереди
    else:
        ret, frame = cap.read()
        if not ret: break

    if (not HOME_TEST and status == beholder.Status.OK) or HOME_TEST:  # Если кадр прочитан успешно ...
        if FPS:
            frame_time = time.time()

        frame = cv2.resize(frame, (256, 256))
        boxes, confidences, class_ids, idxs = make_prediction(net, ln, frame)
        if DRAW_DETECTION:
            frame = draw_boxes(frame, boxes, labels, confidences, class_ids, idxs)

        if len(confidences) > 0:
            conf_id = np.argmax(confidences)
            signal = class_ids[conf_id] + 1

            if start_signal == TRLSignal.none:
                start_signal = TRLSignal(signal)
                if start_signal == TRLSignal.off:
                    start_signal = TRLSignal.green

            if signal != trl_signal:
                now = time.time()

                if signal != trl_signal and trl_signal == start_signal and not flashing_green:
                    change_start += 1

                if not flashing_green:
                    if signal == TRLSignal.off and trl_signal == TRLSignal.green:
                        flashing_green = True

                    timing_signals[TRLSignal(trl_signal)] = max(round(now - timer_start, 2), timing_signals[TRLSignal(trl_signal)])
                    timer_start = now
                    
                elif flashing_green and signal == TRLSignal.yellow:
                    timing_signals[TRLSignal.flashing_green] = max(round(now - timer_start, 2), timing_signals[TRLSignal.flashing_green])
                    flashing_green = False

                    timer_start = now

                if change_start > 2:
                    trl_has = False

                trl_signal = signal

        if FPS:
            print(int(1 / (time.time() - frame_time)))

        cv2.imshow("TRL detection", cv2.resize(frame, \
                (512, 512)))

        timing = 1
        if HOME_TEST:
            timing = int(1000/14)
        key = cv2.waitKey(timing)

    if not HOME_TEST:
        if status == beholder.Status.EOS:  # Если сервер прервал передачу
            print("End of stream")
            break
        elif status == beholder.Status.Error:  # Если кадр пришёл повреждённым
            print("Error")
            break
        elif status == beholder.Status.Timeout:   # Если очередь кадров пуста.
            # Do nothing
            pass

if not HOME_TEST:
    sock.close()
    beholder_client.stop()

print_timing()
cv2.destroyAllWindows()