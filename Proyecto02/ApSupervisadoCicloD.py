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

    #Cuanto tiempo tarda el ciclo sin optimizar. 0.001 = 1ms por operación, 1000 operaciones = 1s
    tiempo_b = iteraciones * op_por_iteracion * 0.001

    # Normal
    tiempo_nor = tiempo_b * (1 + random.uniform(-0.05, 0.05))
    #Asumimos 8 bytes por operación
    memoria_nor = iteraciones * op_por_iteracion * 8
    # 16 para overhead tipico
    tam_codigo_nor = 16 + op_por_iteracion * 2

    # Desenrrollado
        # Región de Compensación Crítica-----
    if 15 < iteraciones < 35 and 8 < op_por_iteracion < 15:
        ruido = random.uniform(-0.2, 0.2)
        valor_desenrrollado = 0.9 + ruido   
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
    #Notemos que estamos asignandole más peso al tiempo, ya que es lo que suele importar más en las optimizaciones.
    resultado_nor = 0.8*(1/tiempo_nor) + 0.19*(1/memoria_nor) + 0.01*(1/tam_codigo_nor)
    resultado_des = 0.8*(1/tiempo_des) + 0.19*(1/memoria_des) + 0.01*(1/tam_codigo_des)

    # Decisión con algo de aleatoriedad para evitar sobreajuste, sobre todo que tenemos datos sinteticos. Con esto el modelo puede generalizar mejor.
    umbral_decision = 0.9 
    if random.random() < umbral_decision:
        if resultado_nor > resultado_des + 0.0001:
            opcion_optima = 0  
        else:
            opcion_optima = 1
    else:
        opcion_optima = random.randint(0, 1)  
        

    return [tiempo_nor, memoria_nor, tam_codigo_nor], [tiempo_des, memoria_des, tam_codigo_des], opcion_optima

#Esto esta simplificado, idealmente necesitamos mas datos y mejores caracteristicas donde la presición deberia ser mas baja
def generar_datos_entrenamiento(n=400):
    """
    Función que genera datos de entrenamiento simulados para el clasificador.
    Parámetros:
        - n: número de muestras a generar
    Regresa:
        - x: matriz de características
        - y: etiquetas óptimas
    """
    x, y = [], []

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

        x.append(caracteristicas)
        y.append(mejor)

    return np.array(x), np.array(y)

def entrenamiento_modelo():
    """
        Funcion para entrenar el modelo de clasificación de ciclos desenrrollados.
        Parámetros:
            - None
        Regresa:
            - clf: clasificador entrenado
            - datos_prueba: datos de prueba
            - et_prueba: etiquetas reales de prueba
    """
    print("="*50)
    print("Entrenemiento del modelo de clasificación de ciclos desenrrollados")

    x, y = generar_datos_entrenamiento(450)

    datos_entr, datos_prueba, et_entr, et_prueba = train_test_split(
        x, y, test_size=0.25, random_state=42
    )
    #Con una profundidad de 4 el modelo generaliza bastante bien. Si aumentamos la profundidad como 6, sigue fucnionando, pero ya se empieza a sobreajustar
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(datos_entr, et_entr)

    print("\nPrecisión del entrenamiento:", clf.score(datos_entr, et_entr))
    print("Precisión de la prueba:",       clf.score(datos_prueba, et_prueba))


    return clf, datos_prueba, et_prueba

def evaluacion_modelo(clf, datos_prueba, et_prueba):
    """
        Función para evaluar el modelo entrenado
        Parámetros:
            - clf: clasificador entrenado
            - datos_prueba: datos de prueba
            - et_prueba: etiquetas reales de prueba
        Regresa:
            - None 
    """
    y_pred = clf.predict(datos_prueba)

    print("\nREPORTE DE CLASIFICACIÓN:")
    print(classification_report(et_prueba, y_pred, target_names=["Normal", "desenrrollado"]))

    print("MATRIZ DE CONFUSIÓN:")
    print(confusion_matrix(et_prueba, y_pred))

def extraccion_caracteristicas_prog(ruta):
    """
    Función para extraer características de un archivo de código .txt. 
    Limitaciones: No Funciona con ciclos donde el límite depende de una variable de entrada o una condición compleja que cambia dinámicamente
    Parámetros:
        - ruta: ruta al archivo de código .txt
    Regresa:
        - caracteristicas: lista de características extraídas
        - tupla con (iteraciones, ops, m_nor, m_des)
    """
    with open(ruta, "r") as f:
        code = f.read()

    # Detección de iteraciones con estructura de python, C, pseudocodigo
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

def funcion_costo(w):
    """
        Función para calcular el costo basado en las metricas w = [tiempo, memoria, tam_codigo]
        Parámetros:
            - w: lista con las métricas [tiempo, memoria, tam_codigo]
        Regresa:
            - costo calculado
    """
    tiempo, memoria, codigo = w
    return 0.8*(1/tiempo) + 0.19*(1/memoria) + 0.01*(1/codigo)
#Evitamos pasar el modelo cada vez a cada función
class LoopOptimizador:
    """
    Clase que realiza el  análisis y recomendacion sobre la optimización de ciclos
    Parámetros:
        - modelo entrenado
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
        #El árbol aprende reglas perfectas sin ambigüedad.  Así que la probabilidad será siempre alta para una clase y baja para la otra debido a datos deterministas.
        #Esto es para facilitar la implementación y poder explicar mejor las decisiones.
        print(f"Probabilidades: Normal={prob[0]:.2f}, desenrrollado={prob[1]:.2f}")

        return pred, prob

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
            pred, prob = self.analiza_carac(caracteristicas)
            
            return {
                "iter": iteraciones,
                "ops": ops,
                "m_nor": m_nor,
                "m_des": m_des,
                "costo_nor": funcion_costo(m_nor),
                "costo_des": funcion_costo(m_des),
                "pred": pred,
                "prob": prob
            }

        except FileNotFoundError:
            print(f"\nError: El archivo '{ruta}' no existe o la ruta es incorrecta.")
            return None

        except Exception as e:
            print(f"\nError inesperado al procesar el archivo:")
            print(f"   → {type(e).__name__}: {e}")
            return None    

def visualizar_arbol(clf):
    """
        Función para visualizar y guardar el árbol de decisión
        Parámetros: 
            - clf: clasificador entrenado
        Regresa:
            - None

        NOTA: 
            -Cada nodo representa una partición del espacio de características, y las ramas etiquetadas como True o False indican si la condición se cumple respecto al umbral.
            - Las hojas muestran la clase predicha (Normal o desenrrollado) junto con la proporción de muestras de cada clase en esa hoja. 
    """
    #Puede que no aparezcan todas las características en los nodos, pero si se aumenta la profundidad del arbol y hay mas datos y se necesitan mas metricas para decidir, apareceran mas
    nombre_carac = [
    "iteraciones", "ops", "ops_totales", "ops_log",
    "es_pequeno", "es_simple", "tiempo_nor", "mem_v1", "tam_codigo_nor"
    ]

    plt.style.use("dark_background")
    plt.figure(figsize=(22, 12))
    
    plot_tree(clf, feature_names=nombre_carac, class_names=["Normal", "desenrrollado"], filled=True, rounded=True, fontsize=8)
    ax = plt.gca()

    # Colores de nodos
    for artist in ax.findobj(plt.Rectangle):
        artist.set_edgecolor("white")
        artist.set_linewidth(1.2)
    # Colores de texto
    for txt in ax.texts:
        if txt.get_bbox_patch() is not None:
            txt.set_color("black")
        else:
            # Texto de aristas 
            txt.set_color("white")
            txt.set_fontweight("bold")
    plt.title("Árbol de Decisión para selección de ciclos desenrrollados", fontsize=16)
    plt.savefig("Imgs/arbolDec.png", dpi=300)
    plt.close()
    print("\nÁrbol de decisión guardado como arbolDec.png")

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
    p1 = optimizador.analiza_archivo(arch1)

    print("\n---- PROGRAMA 2 ----")
    p2 = optimizador.analiza_archivo(arch2)

    if p1 is None or p2 is None:
        print("\nError en análisis")
        return

    print("\n================ DECISIÓN FINAL =================")

    mejor1 = max(p1["costo_nor"], p1["costo_des"])
    mejor2 = max(p2["costo_nor"], p2["costo_des"])

    if mejor1 > mejor2:
        print(f"Se recomienda usar el PROGRAMA 1 ({arch1})")
    else:
        print(f"Se recomienda usar el PROGRAMA 2 ({arch2})")

    print("\nJustificación:")
    print(f"Programa 1 - mejor costo: {mejor1:.6f}")
    print(f"Programa 2 - mejor costo: {mejor2:.6f}")

