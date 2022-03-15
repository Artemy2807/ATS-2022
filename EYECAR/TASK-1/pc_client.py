import socket
import cv2
import beholder
import numpy as np
from func_No300x400 import *
import threading
import time

LINE_DEBUG = True

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
STD_SPEED = 1435
STD_DOWNWARD_SPEED = 1550
STD_ANGLE = 90

KP = 0.32  #0.22   0.32
KD = 0.17
last = 0


def stop():
    set_speed(STOP_SPEED)
    set_angle(STD_ANGLE)

def stop_fast():
    set_speed(STD_DOWNWARD_SPEED)
    set_angle(STD_ANGLE)
    time.sleep(0.5)
    stop()

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

DISTANCE = 3 * 100 # в сантиметрах
#V_CALC = 10 # сантиметры в секунду
#TIME_GOING = DISTANCE / V_CALC
DISTANCE_MARK_CNT = DISTANCE // 25

key = 1
speed = STOP_SPEED

is_empty = True
is_once_empty = True
dst_mark_count = 0
dst_mark_timer = 0

#  меньше 1500 - ехать вперёд, чем меньше значение, тем быстрее; рабочий диапазон от 1410 до 1455
#  больше 1500 - ехать назад, чем больше значение, тем быстрее; рабочий диапазон от 1550 до 1570
set_speed(speed)  # отправляем Raspberry значение скорости
set_angle(STD_ANGLE)  # отправляем Raspberry значение угла поворота колёс

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
rec = cv2.VideoWriter("output.avi", fourcc, 30, (640, 480))

while key != ESCAPE:
    if dst_mark_count >= DISTANCE_MARK_CNT:
        stop()
        break

    status, frame = beholder_client.get_frame(0.25)  # читаем кадр из очереди
    if status == beholder.Status.OK:  # Если кадр прочитан успешно ...

        cv2.imshow("Frame", frame)  # выводим его на экран
        img = cv2.resize(frame, SIZE)

        binary = binarize(img, d=LINE_DEBUG)  # бинаризуем изображение

        perspective = trans_perspective(binary, TRAP, RECT, SIZE)
        # извлекаем область изображения перед колёсами автомобиля
        dst_mark = detect_distance_mark(perspective)

        if dst_mark == False:
            is_once_empty = True

        if dst_mark and is_empty:
            is_empty = False
            is_once_empty = False
            dst_mark_timer = time.time()

            dst_mark_count += 1
            print('new mark number', dst_mark_count)

        now = time.time()
        if (not is_empty) and (now - dst_mark_timer) > 0.5 and is_once_empty:
            is_empty = True
            print('ready to detect new distance mark...')

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

        set_angle(angle)
        rec.write(frame)
        key = cv2.waitKey(1)

    elif status == beholder.Status.EOS:  # Если сервер прервал передачу
        print("End of stream")
        break
    elif status == beholder.Status.Error:  # Если кадр пришёл повреждённым
        print("Error")
        break
    elif status == beholder.Status.Timeout:   # Если очередь кадров пуста.
        # Do nothing
        pass

stop()
rec.release()
sock.close()
beholder_client.stop()

cv2.destroyAllWindows()
