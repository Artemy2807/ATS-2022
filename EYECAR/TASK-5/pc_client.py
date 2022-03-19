import socket
import cv2
import beholder
import numpy as np
from func_No300x400 import *
import threading
import time

HOME_TEST = True
LINE_DEBUG = True

if not HOME_TEST:
    HOST = "192.168.9.196"  # адрес беспилотного автомобиля, на нём запущен сервер, передающий кадры
    #  HOST = 'localhost'
    PORT = 1080  # порт для передачи команд Raspberry Pi

    beholder_client = beholder.Client(zmq_host=HOST,  # создаём и настраиваем клиент, принимающий кадры от Raspberry Pi
                                      # zmq_host="192.168.1.145",
                                      zmq_port=12345,
                                      rtp_host="192.168.9.222",
                                      # Адрес ПК в сети беспилотника. Адрес где мы будем принимать кадры
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
                                      encoding=beholder.Encoding.MJPEG,  # MJPEG,    #H264
                                      limit=1)  # длина очереди кадров на ПК, рекомендуется установить значение 1

    beholder_client.start()  # Запускаем приём кадров, очередь кадров начинает наполнятся
    # Если вы собираетесь выполнять длинные операции, например, чтение нейронной сети с диска, выполните их до старта клиента
    # Если клиент не будет читать кадры в течении 30 секунд, то сервер прервёт передачу видеопотока.

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # определяем сокет, на который будем передавть команды
    sock.connect((HOST, PORT))  # подключаемсся к нему


def send_msg(msg):  # функция отправляющая строку на Raspberri Pi
    # print(f'Sending message ["{msg}"]...', end='')
    msg = '<MSG>' + msg + "</MSG>"
    msg = msg.encode('utf-8')
    # print('done')
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

KP = 0.4  # 0.22   0.32
KD = 0.17
last = 0


def stop():
    set_speed(STOP_SPEED)
    set_angle(STD_ANGLE)
    time.sleep(0.1)


SIZE = (533, 300)
RECT = np.float32([[0, SIZE[1]],
                   [SIZE[0], SIZE[1]],
                   [SIZE[0], 0],
                   [0, 0]])

TRAP = np.float32([[0, SIZE[1]],
                   [SIZE[0], SIZE[1]],
                   [440, 190],
                   [93, 190]])
src_draw = np.array(TRAP, dtype=np.int32)

ESCAPE = 27
SPASE = 32

cv2.namedWindow("Frame")

key = 1

if HOME_TEST:
    cap = cv2.VideoCapture('../my.avi')

if not HOME_TEST:
    #  меньше 1500 - ехать вперёд, чем меньше значение, тем быстрее; рабочий диапазон от 1410 до 1455
    #  больше 1500 - ехать назад, чем больше значение, тем быстрее; рабочий диапазон от 1550 до 1570
    set_angle(STD_ANGLE)  # отправляем Raspberry значение угла поворота колёс
    set_speed(STD_SPEED)  # отправляем Raspberry значение скорости

# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# rec = cv2.VideoWriter("output.avi", fourcc, 30, (640, 480))
povor = False
# road_lines = False
cross_road_cnt = 0
timer_povor = 0

RIGHT_CROSS_ROAD = 65
center = SIZE[0] // 2 - 50
while key != ESCAPE:
    status = True
    frame = None
    if not HOME_TEST:
        status, frame = beholder_client.get_frame(0.25)  # читаем кадр из очереди
    else:
        ret, frame = cap.read()
        if not ret:
            break

    if (not HOME_TEST and status == beholder.Status.OK) or HOME_TEST:  # Если кадр прочитан успешно ...
        img = cv2.resize(frame, SIZE)
        # for p in TRAP:
        #    cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 255, 0), 2)
        cv2.imshow("Frame", img)  # выводим его на экран

        binary = binarize(img, d=LINE_DEBUG)  # бинаризуем изображение
        perspective = trans_perspective(binary, TRAP, RECT, SIZE)

        left, right = centre_mass(perspective, d=LINE_DEBUG)  # находим левую и правую линии размтки
        err = 0 - ((left + right) // 2 - center)  # вычисляем отклонение середины дороги от центра кадра

        if abs(right - left) < 120:
            err = last
            # print("LAST")

        stopline = detect_stop(perspective, cross_road_cnt)

        if stopline:
            cross_road_cnt += 1
            povor = True
            timer_povor = time.time()

        if povor:
            now = time.time()
            if cross_road_cnt <= 2:
                pspeed = 1440
                pangle = 90

                print((now - timer_povor))
                if (now - timer_povor) > 0.7:
                    pangle = RIGHT_CROSS_ROAD
                if (now - timer_povor) > (0.7 + 6.0):
                    pspeed = STD_SPEED
                    pangle = 90
                    povor = False
                    timer_povor = 0
                    print('exit')
                    if cross_road_cnt == 2:
                        center = SIZE[0] // 2

                print(pspeed, pangle)
                if not HOME_TEST:
                    set_speed(pspeed)
                    set_angle(pangle)
            else:
                if not HOME_TEST:
                    stop()
                    time.sleep(0.4)
                    set_speed(1435)
                    set_angle(115)
                    time.sleep(2.0)
                    set_speed(1550)
                    set_angle(90)
                    time.sleep(1.0)
                    stop()
                    time.sleep(0.4)
                    drop_cargo()
                    time.sleep(4)
                break

        if not povor:
            angle = int(STD_ANGLE + KP * err + KD * (err - last))  # Вычисляем угол поворота колёс
            if angle < 65:
                angle = 65
            elif angle > 115:
                angle = 115

            last = err

            if not HOME_TEST:
                set_angle(angle)

        timing = 1
        if HOME_TEST:
            timing = int(1000 / 10)
        key = cv2.waitKey(timing)

    if not HOME_TEST:
        if status == beholder.Status.EOS:  # Если сервер прервал передачу
            print("End of stream")
            break
        elif status == beholder.Status.Error:  # Если кадр пришёл повреждённым
            print("Error")
            break
        elif status == beholder.Status.Timeout:  # Если очередь кадров пуста.
            # Do nothing
            pass

if not HOME_TEST:
    stop()
    # rec.release()
    sock.close()
    beholder_client.stop()

cv2.destroyAllWindows()
