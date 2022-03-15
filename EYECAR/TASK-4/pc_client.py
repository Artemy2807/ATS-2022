import socket
import cv2
import beholder
import numpy as np
from func_No300x400 import *
import threading
from enum import IntEnum
import time

HOME_TEST = True
LINE_DEBUG = True

if not HOME_TEST:
    HOST = "192.168.4.1"  # адрес беспилотного автомобиля, на нём запущен сервер, передающий кадры
    #  HOST = 'localhost'
    PORT = 1080  # порт для передачи команд Raspberry Pi

    beholder_client = beholder.Client(zmq_host=HOST, # создаём и настраиваем клиент, принимающий кадры от Raspberry Pi
                            # zmq_host="192.168.1.145",
                            zmq_port=12345,
                            rtp_host="192.168.4.15",  # Адрес ПК в сети беспилотника. Адрес где мы будем принимать кадры
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
STD_SPEED = 1435
STD_ANGLE = 90

KP = 0.32  #0.22   0.32
KD = 0.17
last = 0

def stop():
    set_speed(STOP_SPEED)
    time.sleep(0.5)

    set_angle(STD_ANGLE)
    # НЕОБХОДИМА ЗАДЕРЖКА, ЧТОБЫ КОД НА RASPBERRY PI УСПЕВАЛ ОБРАБОТАТЬ ДАННЫЕ!!!
    time.sleep(0.5)

def start():
    set_angle(STD_ANGLE)
    # НЕОБХОДИМА ЗАДЕРЖКА, ЧТОБЫ КОД НА RASPBERRY PI УСПЕВАЛ ОБРАБОТАТЬ ДАННЫЕ!!!
    time.sleep(0.5)
    set_speed(STD_SPEED)
    time.sleep(0.1)

SIZE = (533, 300)
SIZE_PED = (533, 104)
RECT = np.float32([[0, SIZE[1]],
                   [SIZE[0], SIZE[1]],
                   [SIZE[0], 0],
                   [0, 0]])
RECT_PED = np.float32([[0, SIZE_PED[1]],
                   [SIZE_PED[0], SIZE_PED[1]],
                   [SIZE_PED[0], 0],
                   [0, 0]])

TRAP = np.float32([[0, SIZE[1]],
                   [SIZE[0], SIZE[1]],
                   [440, 190],
                   [93, 190]])
TRAP_PED = np.float32([[78, 218],
                   [455, 218],
                   [392, 114],
                   [141, 114]])
src_draw = np.array(TRAP, dtype=np.int32)

ESCAPE = 27
SPASE = 32

cv2.namedWindow("Frame")

key = 1
ped_exist = False
ped_timer = 0

if not HOME_TEST:
    #  меньше 1500 - ехать вперёд, чем меньше значение, тем быстрее; рабочий диапазон от 1410 до 1455
    #  больше 1500 - ехать назад, чем больше значение, тем быстрее; рабочий диапазон от 1550 до 1570
    start()

if HOME_TEST:
    cap = cv2.VideoCapture('../Loop.mkv') 

while key != ESCAPE:
    status = True
    frame = None
    if not HOME_TEST:
        status, frame = beholder_client.get_frame(0.25)  # читаем кадр из очереди
    else:
        ret, frame = cap.read()
        frame = cv2.imread('../ped.png')
        if not ret: 
            break

    if (not HOME_TEST and status == beholder.Status.OK) or HOME_TEST:  # Если кадр прочитан успешно ...
        img = cv2.resize(frame, SIZE)
        
        for p in TRAP:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 255, 0), 2)
        for p in TRAP_PED:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 0, 255), 2)
        cv2.imshow("Frame", img)  # выводим его на экран
        

        binary = binarize(img, d=LINE_DEBUG)  # бинаризуем изображение
        binary_ped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_ped = cv2.inRange(binary_ped, 100, 200)

        #bgr_ped = cv2.inRange(img, (100, 100, 100), (200, 200, 200))
        #binary_ped = cv2.bitwise_and(bgr_ped, binary_ped)

        perspective = trans_perspective(binary, TRAP, RECT, SIZE)
        perspective_ped = trans_perspective(binary_ped, TRAP_PED, RECT_PED, SIZE_PED)
        cv2.imshow("Frame ped", perspective_ped)

        left, right = centre_mass(perspective, d=LINE_DEBUG)  # находим левую и правую линии размтки
        err = 0 - ((left + right) // 2 - SIZE[0]//2)  # вычисляем отклонение середины дороги от центра кадра

        ped_detection = 0
        for i in range(left + 80, right - 80):
            ped_detection += int(np.sum(perspective_ped[:, i], axis=0) // 255)
        
        if ped_detection >= 750 and (not ped_exist):
            ped_exist = True
            print('Ped exist!!')
            if not HOME_TEST:
                stop()

        now = time.time()
        if ped_detection < 750 and ped_exist and ped_timer == 0:
            ped_timer = now

        if ped_detection < 750 and ped_exist and (now - ped_timer) > 1.0:
            ped_exist = False
            ped_timer = 0
            print('Starting...')
            if not HOME_TEST:
                start()
        print(ped_detection)

        if abs(right - left) < 100:
            err = last
            print("LAST")

        angle = int(STD_ANGLE + KP * err + KD * (err - last))  # Вычисляем угол поворота колёс
        if angle < 72:
            angle = 72
        elif angle > 108:
            angle = 108

        last = err

        if not HOME_TEST:
            set_angle(angle)

        timing = 1
        if HOME_TEST:
            timing = int(1000/1)
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
    time.sleep(1)

    sock.close()
    beholder_client.stop()
cv2.destroyAllWindows()