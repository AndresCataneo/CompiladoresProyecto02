# CompiladoresProyecto02
Aprendizaje automático para optimización de código

Instalar sklearn
-pip install scikit-learn numpy

Instalar matplotlib  
-pip install matplotlib   

Ejecutar con 
- python .\Proyecto2\Main.py --compare .\Archivos_txt\[archivo1.txt] .\Archivos_txt\[archivo2.txt]
- python .\Proyecto2\Main.py --analyze .\Archivos_txt\[archivo.txt]  


NOTA: Es importante ver que esté programa analiza lso archivos que se les pase y verá que tipo de optimización conviene o si es mejor dejarlo así en cuanto a los ciclos desenrrollados.
Por lo que si queremos comparar si un programa con ciclo desenrrollado es mejor con uno o sin uno, utilizamos --analyze. Sin, embargo, si queremos también compar entro dos programas distintos, usamos --compare.

Esto es util por que puede que ambos programas hagan lo mismo, incluso en el mismo tiempo de complejidad, pero uno sea más eficiente.
