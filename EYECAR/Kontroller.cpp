#include <Arduino.h>  //# это строка нужна только на VOSTOK-UNO
#include <Servo.h>  //# подключение библиотеки для управления сервоприводами

Servo ESC;  //# создание объекта связанного с драйвером двигателя
Servo SRV;  //# создание объекта связанного с рулевым сервоприводом
Servo CARGO;  //# создание объекта связанного с сервоприводом кузова

int Speed, Angle;

char del = ':';
String angle_cmd = "ANGLE",
    speed_cmd = "SPEED",
    cargo_cmd = "CARGO";

int width_impulse(int angle){
  return angle * 11.1 + 500;  //# Пересчёт градусов в длину импульса для сервопривода
}

void dropCargo(){  //# Опрокидывает кузов и возвращает его в исходное положение через 1,5 секунды
  setSpeed_(1500);  // останавливаемся! опрокидывать кузов на ходу - плохая идея
  CARGO.write(90); //# принять положение 90 градусов, для сервопривода кузова // кузов опрокинут
  delay(1500);     //# тут, delay можно оставить
  CARGO.write(160);  //# принять положение 160 градусов, для сервопривода кузова  // кузов не опрокинут
}

void setAngle(int angle){ //повернуть рулевые колёса в положение соответствующее углу, преданному в качестве аргумента
    // Angle = angle;
    if(angle > 110){angle = 110;}
    if (angle < 70){angle = 70;}
    SRV.write(width_impulse(Angle));
}

void setSpeed_(int _speed){  //ехать с новой соростью, значение скорости в аргументе
    // Speed = _speed;
    ESC.writeMicroseconds(_speed);
}

void range(int& val, int mn, int mx) {
  val = (val < mn ? mn : (val > mx ? mx : val));
}

void setup() {  //Задать настройки для приёма сообщений от Raspberry Pi через UART
  Serial1.begin(9600); // назначаем скорость общения
  ESC.attach(9);  // указываем пин драйвера двигателя
  SRV.attach(8);  // указываем пин рулевого сервопривода
  CARGO.attach(10);  // указываем пин сервопривода кузова
  ESC.writeMicroseconds(1500); // отправляем стартовый сигнал драйверу двигателя, он запомнит его как сигнал "СТОП"
  CARGO.write(160);  // возвращаем сервопривод кузова в положение "кузов не опрокинут"
  SRV.write(90);  // устанавливаем рулевые колёса прямо
  delay(2000);  // ждём 2с, чтобы драйвер двигателя "уловил" сигнал "СТОП"
}

void loop() {
  if(Serial1.available() > 0){
    String data = Serial1.readStringUntil('\n'); // читаем символы, пока не прочитаем симыол конца строки

    int idx = data.indexOf(del);
    if (idx >= 0) {

      String cmd = data.substring(0, idx);
      if (cmd.equals(cargo_cmd))
        dropCargo();  // выполняем если получили команду "опрокинуть кузов"
      else {
        int val = data.substring(idx + 1).toInt();
        if (cmd.equals(speed_cmd)) {
          range(val, 1410, 1570);
          setSpeed_(val);
        }else if(cmd.equals(angle_cmd)) {
          range(val, 72, 108);
          setAngle(val);
        }
      }

    }
  }
}