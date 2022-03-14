import socket
import time
import os
import serial


class Arduino:  # класс ардуино! При создании объекта, задаёт порт и скорость обмена данными, для общения по UART
    def __init__(self, port, baudrate=9600, timeout=1):
        self.serial = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self.serial.flush()

    def send_data(self, data: str):  # метод класса для отправки данных через UART
        data += '\n'
        self.serial.write(data.encode('utf-8'))

    def read_data(self):  # метод класса для чтения данных через UART !!! Не используется в программе. !!!
        line = self.serial.readline().decode('utf-8').strip()
        return line


def setup_socket(port):  # создаёт TCP сокет и начинает слушать его. Для приёма данных от компьютера оператора
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('', port)
    sock.bind(server_address)
    sock.listen(1)
    print('Listening...')
    return sock


def main():
    arduino = Arduino('/dev/ttyACM0')  # создаём экземпляр класса Arduino, будем вызывать его методы для общения через UART
    port = 1080
    sock = setup_socket(port)
    while True:  # Цикл в котором ждём подключения компьютер оператора (ПК) к сокету
        print('Waiting for new incoming connection...')
        conn, address = sock.accept()  # Ждём подключения
        print(address, 'is connected!')  # Сообщаем о том, что к сокету кто-то подключился
        try:
            while True:  # Цикл приёма сообщений
                data = conn.recv(100)  # читаем символы, которые нам прислали с ПК, но не больше 100
                data = data.decode('utf-8')  # декодируем данные из битового представления в символы
                print('Received on server:', data)  # выводим полученные данные
                if not data:  # если данных нет, считаем, что подключение оборвалось
                    conn.close()  # закрываем соединение
                    break  # выходим во внешний цикл, ждать повторного подключения

                arduino.send_data(data)  # отправляем VOSTOK_UNO данные, полученные от ПК
        except Exception as e:
            print('\nConnection was broken!')
            print('[ERROR]', e)


if __name__ == '__main__':
    main()
