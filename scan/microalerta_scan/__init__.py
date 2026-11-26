"""MicroAlerta Scan — pipeline de visión para partículas compatibles con microplásticos (PCM).

Fase 1: visión clásica + rasgos morfológicos/ópticos + clasificador tabular (Random Forest).
Fase 2: detección por fluorescencia (Rojo de Nilo), captura en cuadrícula con dos motores en las
perillas de la platina, cuadrícula del disco y prueba simulada sin equipo.
No asigna polímeros por color: clasifica forma (fibra/fragmento/película) y, si hay un
modelo entrenado con etiquetas validadas, la probabilidad de que la partícula sea PCM.
"""
__version__ = "0.2.3"
