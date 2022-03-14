import socket
import cv2
import beholder
import numpy as np
from func_No300x400 import *
import threading
from enum import IntEnum
import time

HOME_TEST = True
LINE_DEBUG = False

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


trl_image = []
trl_output_image = []

class TRLSignal(IntEnum):
    flashing_green = 6
    red_yellow = 5
    yellow = 4
    red = 3
    off = 2
    green = 1
    none = 0
trl_signal = TRLSignal.none
start_robot = False

def trl_detect():
    global trl_image, trl_output_image, trl_signal, start_robot

    CONFIG_FILE = '../yolov4-tiny-usr.cfg'
    WEIGHT_FILE = '../yolov4-tiny_last.weights'
    DRAW_DETECTION = True
    FPS = False

    GREEN_TIME = 0.2

    labels = ['green', 'off', 'red', 'yellow', 'red+yellow', 'flashing green']

    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHT_FILE)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    frame_time = 0
    green_time_start = 0

    while not start_robot:
        if FPS:
            frame_time = time.time()
        
        if len(trl_image) == 0:
            continue

        frame = cv2.resize(trl_image, (256, 256))
        boxes, confidences, class_ids, idxs = make_prediction(net, ln, frame)
        if DRAW_DETECTION:
            frame = draw_boxes(frame, boxes, labels, confidences, class_ids, idxs)
        trl_output_image = frame.copy()

        if len(confidences) > 0:
            conf_id = np.argmax(confidences)
            signal = class_ids[conf_id] + 1

            if signal != trl_signal and TRLSignal(signal) == TRLSignal.green:
                green_time_start = time.time()

            now = time.time()
            if TRLSignal(signal) == TRLSignal.green and signal == trl_signal \
                    and (now - green_time_start) >= GREEN_TIME:
                start_robot = True
            trl_signal = signal

        if FPS:
            print(int(1 / (time.time() - frame_time)))

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
                            width=1280,  # ширина кадра
                            height=720,  # высот кадра
                            # width=640,
                            # height=480,
                            framerate=30,  # частота кадров
                            encoding=beholder.Encoding.MJPEG,  #MJPEG,    #H264
                            limit=20)  # длина очереди кадров на ПК, рекомендуется установить значение 1

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
STD_SPEED = 1435
STD_ANGLE = 90

def stop():
    set_speed(STOP_SPEED)
    set_angle(STD_ANGLE)

KP = 0.32  #0.22   0.32
KD = 0.17
last = 0

SIZE = (533, 300)

RECT = np.float32([[0, SIZE[1]],
                   [SIZE[0], SIZE[1]],
                   [SIZE[0], 0],
                   [0, 0]])

TRAP = np.float32([[10, 299],
                   [523, 299],
                   [440, 200],
                   [93, 200]])
src_draw = np.array(TRAP, dtype=np.int32)

ESCAPE = 27
SPASE = 32

cv2.namedWindow("Frame")

traffic_thread = threading.Thread(target=trl_detect, name="Traffic_light_detector", daemon=True)
traffic_thread.start()

DISTANCE = 4 * 100 # в сантиметрах
V_CALC = 20 # сантиметры в секунду
TIME_GOING = DISTANCE / V_CALC

key = 1
speed = STD_SPEED

if not HOME_TEST:
    #  меньше 1500 - ехать вперёд, чем меньше значение, тем быстрее; рабочий диапазон от 1410 до 1455
    #  больше 1500 - ехать назад, чем больше значение, тем быстрее; рабочий диапазон от 1550 до 1570
    set_speed(speed)  # отправляем Raspberry значение скорости
    set_angle(STD_ANGLE)  # отправляем Raspberry значение угла поворота колёс

if HOME_TEST:
    cap = cv2.VideoCapture('../tr4.mp4') 

start_timer = 0
while key != ESCAPE:
    if start_robot and start_timer == 0:
        start_timer = time.time()

    now = time.time()
    if start_robot and (now - start_timer) >= TIME_GOING:
        #stop_fast()
        if not HOME_TEST:
            stop()
        break

    status = True
    frame = None
    if not HOME_TEST:
        status, frame = beholder_client.get_frame(0.25)  # читаем кадр из очереди
    else:
        ret, frame = cap.read()
        if not ret: 
            break

    if (not HOME_TEST and status == beholder.Status.OK) or HOME_TEST:  # Если кадр прочитан успешно ...
        if len(trl_output_image) != 0 and not start_robot:
            cv2.imshow("TRL detection", cv2.resize(trl_output_image, \
                    (512, 512)))

        cv2.imshow("Frame", frame)  # выводим его на экран
        img = cv2.resize(frame, SIZE)
        trl_image = img.copy()

        binary = binarize(img, d=LINE_DEBUG)  # бинаризуем изображение

        perspective = trans_perspective(binary, TRAP, RECT, SIZE)
        # извлекаем область изображения перед колёсами автомобиля

        left, right = centre_mass(perspective, d=LINE_DEBUG)  # находим левую и правую линии размтки
        err = 0 - ((left + right) // 2 - SIZE[0]//2)  # вычисляем отклонение середины дороги от центра кадра
        if abs(right - left) < 100:
            err = last
            print("LAST")

        angle = int(STD_ANGLE + KP * err + KD * (err - last))  # Вычисляем угол поворота колёс
        if angle < 72:
            angle = 72
        elif angle > 108:
            angle = 108

        last = err

        # set_speed(speed)
        if not HOME_TEST:
            set_angle(angle)
        # 14

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
    stop()

    sock.close()
    beholder_client.stop()
cv2.destroyAllWindows()