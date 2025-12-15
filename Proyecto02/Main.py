import ApSupervisadoCicloD as apSupC
import sys
def uso_programa():
    """
    Función para mostrar el uso del programa desde línea de comandos.

    Parámetros:
        - None
    Regresa:
        - None
    """
    print("""
    Uso:
        python Main.py --analizar archivo.txt
        python Main.py --comparar archivo1.txt archivo2.txt""")


if __name__ == "__main__":
    args = sys.argv

    if len(args) == 1:
        uso_programa()
        sys.exit(0)
    
    if(args[1] == "--analizar" or args[1] == "--comparar") and len(args) <= 4:
        clf, datos_prueba, et_prueba = apSupC.entrenamiento_modelo()

        apSupC.evaluacion_modelo(clf, datos_prueba, et_prueba)

        apSupC.visualizar_arbol(clf)

        optimizador = apSupC.LoopOptimizador(clf)

        if args[1] == "--analizar":
            if len(args) != 3:
                uso_programa()
                sys.exit(1)
            optimizador.analiza_archivo(args[2])
            sys.exit(0)

        if args[1] == "--comparar":
            if len(args) != 4:
                uso_programa()
                sys.exit(1)
            apSupC.compara_programas(args[2], args[3], optimizador)
            sys.exit(0)
    else:
        print("Error: comando desconocido.")
        uso_programa()
