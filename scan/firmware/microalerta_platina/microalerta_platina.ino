// MicroAlerta: platina con dos motores 28BYJ-48 (drivers ULN2003) en las perillas X e Y
// y dos luces (LED azul para el Rojo de Nilo y LED blanco), controlada por el computador.
//
// Conexiones (Arduino Nano):
//   Motor X  -> ULN2003 IN1..IN4 = D2, D3, D4, D5
//   Motor Y  -> ULN2003 IN1..IN4 = D6, D7, D8, D9
//   LED azul -> D10 (PWM) a través de un transistor o MOSFET (no directo al pin si consume más de 20 mA)
//   LED blanco -> D11 (PWM), igual
//   Fuente de 5 V 2 A para los motores; GND común con el Arduino.
//
// Órdenes por el puerto serie (115200 baudios, una por línea). Responde una línea por orden:
//   ?            -> MICROALERTA 1
//   M x y        -> va a la posición absoluta (pasos) y responde OK x y
//   R dx dy      -> se mueve en pasos relativos y responde OK x y
//   Z            -> la posición actual pasa a ser 0 0 (OK 0 0)
//   P            -> POS x y
//   L azul blanca-> brillo de las luces, 0 a 255 (OK)
//   V us         -> microsegundos entre pasos (por defecto 2500; menos = más rápido, puede perder pasos)
// Después de cada movimiento apaga las bobinas: la caja reductora del motor sostiene la perilla
// y así los motores no se calientan.

const byte PX[4] = {2, 3, 4, 5};
const byte PY[4] = {6, 7, 8, 9};
const byte LED_AZUL = 10;
const byte LED_BLANCA = 11;

// paso completo con dos bobinas: 2048 pasos por vuelta del eje de salida
const byte SECUENCIA[4][4] = {{1, 1, 0, 0}, {0, 1, 1, 0}, {0, 0, 1, 1}, {1, 0, 0, 1}};

long posX = 0, posY = 0;
byte faseX = 0, faseY = 0;
unsigned int esperaUs = 2500;
String linea = "";

void bobinas(const byte *pines, byte fase) {
  for (byte i = 0; i < 4; i++) digitalWrite(pines[i], SECUENCIA[fase][i]);
}

void soltar(const byte *pines) {
  for (byte i = 0; i < 4; i++) digitalWrite(pines[i], LOW);
}

void moverEje(const byte *pines, byte &fase, long &pos, long destino) {
  while (pos != destino) {
    if (destino > pos) { fase = (fase + 1) & 3; pos++; }
    else               { fase = (fase + 3) & 3; pos--; }
    bobinas(pines, fase);
    delayMicroseconds(esperaUs);
  }
  delay(20);
  soltar(pines);
}

void irA(long x, long y) {
  moverEje(PX, faseX, posX, x);
  moverEje(PY, faseY, posY, y);
}

void responderPos(const char *pre) {
  Serial.print(pre); Serial.print(' ');
  Serial.print(posX); Serial.print(' ');
  Serial.println(posY);
}

void ejecutar(String s) {
  s.trim();
  if (s.length() == 0) return;
  char c = s.charAt(0);
  String resto = s.substring(1);
  resto.trim();
  int esp = resto.indexOf(' ');
  long a = resto.toInt();
  long b = esp >= 0 ? resto.substring(esp + 1).toInt() : 0;
  switch (c) {
    case '?': Serial.println("MICROALERTA 1"); break;
    case 'M': irA(a, b); responderPos("OK"); break;
    case 'R': irA(posX + a, posY + b); responderPos("OK"); break;
    case 'Z': posX = 0; posY = 0; responderPos("OK"); break;
    case 'P': responderPos("POS"); break;
    case 'L': analogWrite(LED_AZUL, constrain(a, 0, 255)); analogWrite(LED_BLANCA, constrain(b, 0, 255));
              Serial.println("OK"); break;
    case 'V': esperaUs = constrain(a, 800, 20000); Serial.println("OK"); break;
    default:  Serial.println("ERR orden desconocida");
  }
}

void setup() {
  for (byte i = 0; i < 4; i++) { pinMode(PX[i], OUTPUT); pinMode(PY[i], OUTPUT); }
  pinMode(LED_AZUL, OUTPUT);
  pinMode(LED_BLANCA, OUTPUT);
  soltar(PX); soltar(PY);
  analogWrite(LED_AZUL, 0); analogWrite(LED_BLANCA, 0);
  Serial.begin(115200);
}

void loop() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (linea.length()) ejecutar(linea);
      linea = "";
    } else if (linea.length() < 40) {
      linea += ch;
    }
  }
}
