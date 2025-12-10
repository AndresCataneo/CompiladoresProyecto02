"""
    Programa que permite clasificar la optimización de un programa con ciclos enrrollados y desenrrollados
"""
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import re

def ejecucion_versiones(iteraciones, op_por_iteracion):
    """
        Funcion que simula la ejecución de dos versiones de un ciclo:
        - Versión normal (sin optimización)
        - Versión con ciclo desenrrollado 

        Parametros:
            - iteraciones: número de iteraciones del ciclo
            - op_por_iteracion: número de operaciones por iteración
        Regresa:
            - métricas de la versión normal: [tiempo, memoria, código]
            - métricas de la versión desenrrollada: [tiempo, memoria, código]
            - etiqueta óptima: 0 si normal es mejor, 1 si desenrrollada es mejor

    """

    #Cuanto tiempo tarda el ciclo sin optimizar
    tiempo_b = iteraciones * op_por_iteracion * 0.001

    # Normal
    tiempo_nor = tiempo_b * (1 + random.uniform(-0.05, 0.05))
    memoria_nor = iteraciones * op_por_iteracion * 8
    tam_codigo_nor = 120 + op_por_iteracion * 2

    # Desenrrollado
        # Región de Compensación Crítica-----
    if 15 < iteraciones < 35 and 8 < op_por_iteracion < 15:
        ruido = random.uniform(-0.2, 0.2)
        valor_desenrrollado = 0.9 + ruido   # A veces ayuda, a veces no
    else:
        if iteraciones <= 20 and op_por_iteracion <= 12:
            valor_desenrrollado = 0.75
        else:
            valor_desenrrollado = 1.10
        #------------------------------------
    tiempo_des = tiempo_b * valor_desenrrollado * (1 + random.uniform(-0.05, 0.05))
    memoria_des = memoria_nor * 1.4
    tam_codigo_des = tam_codigo_nor * 2


    # Evaluación  (Modelo de costo con funcion de aptitud basado en MLGO)
    resultado_nor = 0.7*(1/tiempo_nor) + 0.2*(1/memoria_nor) + 0.1*(1/tam_codigo_nor)
    resultado_des = 0.7*(1/tiempo_des) + 0.2*(1/memoria_des) + 0.1*(1/tam_codigo_des)

    if resultado_des > resultado_nor + 0.0001:
        opcion_optima = 1
    else:
        opcion_optima = 0

    return [tiempo_nor, memoria_nor, tam_codigo_nor], [tiempo_des, memoria_des, tam_codigo_des], opcion_optima

#Esto esta simplificado, idealmente necesitamos mas datos y mejores caracteristicas donde la presición deberia ser mas baja
def generate_training_data(n=400):
    """
    Función que genera datos de entrenamiento simulados para el clasificador.
    Parámetros:
        - n: número de muestras a generar
    Regresa:
        - X: matriz de características
        - y: etiquetas óptimas
    """
    X, y = [], []

    for _ in range(n):
        iteraciones = random.randint(5, 120)
        operations = random.randint(1, 25)
        # m_des lo calculamos pero no lo usamos en las características porque eso es la respuesta
        m_nor, _, mejor = ejecucion_versiones(iteraciones, operations)

        caracteristicas = [
            iteraciones,
            operations,
            iteraciones * operations,
            math.log1p(operations),
            # ciclos pequeños y simples 
            1 if iteraciones < 20 else 0, 
            1 if operations < 8 else 0,
            # tiempo_nor, memoria_nor, tam_codigo_nor
            m_nor[0], m_nor[1], m_nor[2]
        ]

        X.append(caracteristicas)
        y.append(mejor)

    return np.array(X), np.array(y)


def entrenamiento_modelo():
    """
    Docstring for entrenamiento_modelo
    """
    print("="*50)
    print("Entrenemiento del modelo de clasificación de ciclos desenrrollados")

    X, y = generate_training_data(450)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    clf = DecisionTreeClassifier(max_depth=7, random_state=42)
    clf.fit(X_train, y_train)

    print("\nPrecisión del entrenamiento:", clf.score(X_train, y_train))
    print("Precisión de la prueba:",       clf.score(X_test, y_test))


    return clf, X_test, y_test

def evaluacion_modelo(clf, X_test, y_test):
    """
        Función para evaluar el modelo entrenado
        Parámetros:
            - clf: clasificador entrenado
            - X_test: datos de prueba
            - y_test: etiquetas reales de prueba
        Regresa:
            - None 
    """
    y_pred = clf.predict(X_test)

    print("\nREPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "desenrrollado"]))

    print("MATRIZ DE CONFUSIÓN:")
    print(confusion_matrix(y_test, y_pred))

def visualizar_arbol(clf):
    """
        Función para visualizar y guardar el árbol de decisión
        Parámetros: 
            - clf: clasificador entrenado
        Regresa:
            - None
    """
    feature_names = [
    "iteraciones", "ops", "ops_totales", "ops_log",
    "es_pequeno", "es_simple", "tiempo_nor", "mem_v1", "tam_codigo_nor"
    ]

    plt.style.use("dark_background")
    plt.figure(figsize=(22, 12))
    
    plot_tree(clf, feature_names=feature_names, class_names=["Normal", "desenrrollado"], filled=True, rounded=True, fontsize=8)
    ax = plt.gca()

    # Bordes de nodos blancos
    for artist in ax.findobj(plt.Rectangle):
        artist.set_edgecolor("white")
        artist.set_linewidth(1.2)

    # Texto más brillante
    for txt in ax.texts:
        txt.set_color("black")
    plt.title("Árbol de Decisión para selección de ciclos desenrrollados", fontsize=16)
    plt.savefig("Proyecto2/Imgs/arbolDec.png", dpi=300)
    plt.close()
    print("\nÁrbol de decisión guardado como arbolDec.png")


def extraccion_caracteristicas_prog(ruta):
    """
    Función para extraer características de un archivo de código .txt
    Parámetros:
        - ruta: ruta al archivo de código .txt
    Regresa:
        - caracteristicas: lista de características extraídas
        - tupla con (iteraciones, ops, m_nor, m_des)
    """
    with open(ruta, "r") as f:
        code = f.read()

    # Detección de iteraciones
    patrones = [
        r"range\((\d+)\)",
        r"<\s*(\d+)\)",
        r"<\s*(\d+)",
        r"hasta\s+(\d+)"
    ]

    iteraciones = None
    for p in patrones:
        m = re.search(p, code)
        if m:
            iteraciones = int(m.group(1))
            break

    if iteraciones is None:
        iteraciones = 20

    # Eliminamos comentarios
    lineas = [l for l in code.split("\n") if not l.strip().startswith("#")]
    codigo_dep = "\n".join(lineas)

    operadores = ['+=', '-=', '*=', '/=', '+', '-', '*', '/', '%']
    ops = sum(codigo_dep.count(op) for op in operadores)
    # Si no detecta, estimar por complejidad del cuerpo del ciclo
    if ops == 0:
        cuerpo_ciclo = re.findall(r'for.*?:\s*(.*?)(?=for|$)', code, re.DOTALL)
        if cuerpo_ciclo:
            ops = max(5, len(cuerpo_ciclo[0].split("\n")) * 2)
        else:
            ops = 5

    # Metricas sin ninguna optimización
    m_nor, m_des, _ = ejecucion_versiones(iteraciones, ops)

    caracteristicas = [
        iteraciones,
        ops,
        iteraciones * ops,
        math.log1p(ops),
        1 if iteraciones < 20 else 0,
        1 if ops < 8 else 0,
        m_nor[0], m_nor[1], m_nor[2]
    ]

    return caracteristicas, (iteraciones, ops, m_nor, m_des)

#Evitamos pasar el modelo cada vez a cada función
class LoopOptimizador:
    """
    Clase que realiza el  análisis y recomendacion sobre la optimización de ciclos
    Parámetros:
        - modelo entrenado

    Métodos:
    """
    def __init__(self,modelo):
        """
            Funcion que inicializa el optimizador con un modelo de clasificación entrenado.

            Parámetros:
                modelo (sklearn classifier):
                    Modelo ya entrenado que predice entre 'Normal' y 'desenrrollado'.

            Regresa:
                None
        """
        self.clf = modelo

    def analiza_carac(self, caracteristicas):
        """
            Funcion  que analiza las características de un programa y poredice si usar ciclo desenrrollado o no.

            Parámetros:
                caracteristicas: Lista de características extraídas del ciclo, lista que debe coincidir con el formato usado durante el entrenamiento.
            Regresa:
                None
        """
        pred = self.clf.predict([caracteristicas])[0]
        prob = self.clf.predict_proba([caracteristicas])[0]

        print("\nRecomendación:")
        print("→ Usar desenrrollado" if pred == 1 else "→ Usar NORMAL")
        print(f"Probabilidades: Normal={prob[0]:.2f}, desenrrollado={prob[1]:.2f}")

    def analiza_archivo(self, ruta):
        """
            Función que analiza un archivo de código .txt que contiene un ciclo, extrae sus características y recomienda optimización.
            Parámetros:
                - ruta: ruta al archivo de código .txt
            Regresa:
                - caracteristicas: lista de características extraídas o None en caso de error
                - None en caso de error
        """
        try:
            caracteristicas, (iteraciones, ops, m_nor, m_des) = extraccion_caracteristicas_prog(ruta)

            print(f"\nLoop detectado: iter={iteraciones}, ops={ops}")
            print(f"Métricas normal:   {m_nor}")
            print(f"Métricas desenrrollado: {m_des}")
            self.analiza_carac(caracteristicas)

            return caracteristicas

        except FileNotFoundError:
            print(f"\nError: El archivo '{ruta}' no existe o la ruta es incorrecta.")
            return None

        except Exception as e:
            print(f"\nError inesperado al procesar el archivo:")
            print(f"   → {type(e).__name__}: {e}")
            return None    



def compara_programas(arch1, arch2, optimizador):
    """
    Función para comparar dos programas y mostrar recomendaciones de optimización.
    Parámetros:
        - arch1: ruta al primer archivo de código .txt
        - arch2: ruta al segundo archivo de código .txt 
    Regresa:
        - None
    """
    print("\n=================================================")

    print(f"Comparación de programas:  {arch1} con  {arch2}")
    print("==================================================")

    print("\n---- PROGRAMA 1 ----")
    optimizador.analiza_archivo(arch1)

    print("\n---- PROGRAMA 2 ----")
    optimizador.analiza_archivo(arch2)

