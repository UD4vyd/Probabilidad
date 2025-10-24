import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, hypergeom

class SolucionadorProblemas:
    def __init__(self):
        self.problemas = {
            1: "Selección de estudiantes",
            2: "Ordenamiento de libros", 
            3: "Ordenamiento de estudiantes por carrera",
            4: "Probabilidad con dados",
            5: "Probabilidad de memorias defectuosas"
        }
    
    def mostrar_menu(self):
        print("\n" + "="*50)
        print("          SOLUCIONADOR DE PROBLEMAS")
        print("="*50)
        for key, value in self.problemas.items():
            print(f"{key}. {value}")
        print("6. Salir")
        print("="*50)
    
    def ejecutar(self):
        while True:
            self.mostrar_menu()
            try:
                opcion = int(input("\nSeleccione el problema a resolver (1-6): "))
                
                if opcion == 1:
                    self.problema_1()
                elif opcion == 2:
                    self.problema_2()
                elif opcion == 3:
                    self.problema_3()
                elif opcion == 4:
                    self.problema_4()
                elif opcion == 5:
                    self.problema_5()
                elif opcion == 6:
                    print("¡Hasta luego!")
                    break
                else:
                    print("Opción no válida. Intente nuevamente.")
                    
            except ValueError:
                print("Por favor ingrese un número válido.")
    
    def problema_1(self):
        print("\n" + "="*50)
        print("PROBLEMA 1: Selección de estudiantes")
        print("="*50)
        
        # Datos del problema
        E = 8  # Electrónica
        S = 3  # Sistemas  
        I = 9  # Industrial
        total = E + S + I
        
        print(f"Total de estudiantes: {total}")
        print(f"Electrónica (E): {E}, Sistemas (S): {S}, Industrial (I): {I}")
        
        # Cálculo de combinaciones totales
        C_total = math.comb(total, 3)
        print(f"\nCombinaciones totales de 3 estudiantes: C({total},3) = {C_total}")
        
        # a) 3 estudiantes de Electrónica
        C_3E = math.comb(E, 3)
        P_3E = C_3E / C_total
        print(f"\na) Probabilidad de 3 estudiantes de Electrónica:")
        print(f"   C(8,3) = {C_3E}")
        print(f"   P = {C_3E}/{C_total} = {P_3E:.4f} ({P_3E*100:.2f}%)")
        
        # b) 3 estudiantes de Sistemas
        C_3S = math.comb(S, 3)
        P_3S = C_3S / C_total
        print(f"\nb) Probabilidad de 3 estudiantes de Sistemas:")
        print(f"   C(3,3) = {C_3S}")
        print(f"   P = {C_3S}/{C_total} = {P_3S:.4f} ({P_3S*100:.2f}%)")
        
        # c) 2E y 1S
        C_2E = math.comb(E, 2)
        C_1S = math.comb(S, 1)
        C_2E1S = C_2E * C_1S
        P_2E1S = C_2E1S / C_total
        print(f"\nc) Probabilidad de 2E y 1S:")
        print(f"   C(8,2) × C(3,1) = {C_2E} × {C_1S} = {C_2E1S}")
        print(f"   P = {C_2E1S}/{C_total} = {P_2E1S:.4f} ({P_2E1S*100:.2f}%)")
        
        # d) Al menos 1S
        # Probabilidad complementaria: 1 - P(ningún S)
        C_ningun_S = math.comb(total - S, 3)
        P_ningun_S = C_ningun_S / C_total
        P_al_menos_1S = 1 - P_ningun_S
        print(f"\nd) Probabilidad de al menos 1 estudiante de Sistemas:")
        print(f"   P(ningún S) = C(17,3)/C(20,3) = {C_ningun_S}/{C_total} = {P_ningun_S:.4f}")
        print(f"   P(al menos 1S) = 1 - {P_ningun_S:.4f} = {P_al_menos_1S:.4f} ({P_al_menos_1S*100:.2f}%)")
        
        # e) Escoger 1 de cada carrera
        C_1E = math.comb(E, 1)
        C_1S = math.comb(S, 1) 
        C_1I = math.comb(I, 1)
        C_1cada = C_1E * C_1S * C_1I
        P_1cada = C_1cada / C_total
        print(f"\ne) Probabilidad de escoger 1 de cada carrera:")
        print(f"   C(8,1) × C(3,1) × C(9,1) = {C_1E} × {C_1S} × {C_1I} = {C_1cada}")
        print(f"   P = {C_1cada}/{C_total} = {P_1cada:.4f} ({P_1cada*100:.2f}%)")
        
        # f) Escoger en orden E-S-I
        # Para orden específico, usamos permutaciones
        P_ESI = (E/total) * (S/(total-1)) * (I/(total-2))
        print(f"\nf) Probabilidad de escoger en orden E-S-I:")
        print(f"   P = (8/20) × (3/19) × (9/18) = {P_ESI:.4f} ({P_ESI*100:.2f}%)")
        
        # Gráfica
        self.graficar_problema_1([P_3E, P_3S, P_2E1S, P_al_menos_1S, P_1cada, P_ESI])
    
    def graficar_problema_1(self, probabilidades):
        categorias = ['3E', '3S', '2E1S', 'Al menos 1S', '1 de cada', 'Orden E-S-I']
        colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(categorias, probabilidades, color=colores, alpha=0.7)
        plt.title('Probabilidades - Selección de Estudiantes', fontsize=14, fontweight='bold')
        plt.ylabel('Probabilidad', fontsize=12)
        plt.xticks(rotation=45)
        
        # Añadir valores en las barras
        for bar, prob in zip(bars, probabilidades):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def problema_2(self):
        print("\n" + "="*50)
        print("PROBLEMA 2: Ordenamiento de libros")
        print("="*50)
        
        ing = 4  # Ingeniería
        ingle = 6  # Inglés
        fis = 2   # Física
        total = ing + ingle + fis
        
        print(f"Total de libros: {total}")
        print(f"Ingeniería: {ing}, Inglés: {ingle}, Física: {fis}")
        
        # a) Libros de cada asignatura juntos
        # Consideramos cada grupo como un bloque
        perm_bloques = math.factorial(3)  # Permutaciones de los 3 bloques
        perm_ing = math.factorial(ing)    # Permutaciones dentro de ingeniería
        perm_ingle = math.factorial(ingle) # Permutaciones dentro de inglés
        perm_fis = math.factorial(fis)    # Permutaciones dentro de física
        
        formas_a = perm_bloques * perm_ing * perm_ingle * perm_fis
        print(f"\na) Libros de cada asignatura juntos:")
        print(f"   Permutaciones de bloques: 3! = {perm_bloques}")
        print(f"   Permutaciones dentro de ingeniería: {ing}! = {perm_ing}")
        print(f"   Permutaciones dentro de inglés: {ingle}! = {perm_ingle}")
        print(f"   Permutaciones dentro de física: {fis}! = {perm_fis}")
        print(f"   Total de formas: {perm_bloques} × {perm_ing} × {perm_ingle} × {perm_fis} = {formas_a:,}")
        
        # b) Solo los libros de ingeniería juntos
        # Consideramos ingeniería como un bloque + los otros 8 libros individuales
        elementos = 1 + ingle + fis  # 1 bloque + 8 libros individuales
        perm_elementos = math.factorial(elementos)
        perm_dentro_ing = math.factorial(ing)
        
        formas_b = perm_elementos * perm_dentro_ing
        print(f"\nb) Solo los libros de ingeniería juntos:")
        print(f"   Elementos a permutar: 1 bloque + {ingle + fis} libros = {elementos} elementos")
        print(f"   Permutaciones de elementos: {elementos}! = {perm_elementos:,}")
        print(f"   Permutaciones dentro de ingeniería: {ing}! = {perm_dentro_ing}")
        print(f"   Total de formas: {perm_elementos:,} × {perm_dentro_ing} = {formas_b:,}")
        
        # Gráfica comparativa
        self.graficar_problema_2(formas_a, formas_b)
    
    def graficar_problema_2(self, formas_a, formas_b):
        categorias = ['Cada asignatura junta', 'Solo ingeniería junta']
        valores = [formas_a, formas_b]
        colores = ['#6A89CC', '#B8E994']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categorias, valores, color=colores, alpha=0.7)
        plt.title('Número de Formas de Ordenar Libros', fontsize=14, fontweight='bold')
        plt.ylabel('Número de Formas', fontsize=12)
        
        # Añadir valores en las barras
        for bar, valor in zip(bars, valores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1*height,
                    f'{valor:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def problema_3(self):
        print("\n" + "="*50)
        print("PROBLEMA 3: Ordenamiento de estudiantes")
        print("="*50)
        
        E = 5  # Electrónica
        S = 2  # Sistemas
        I = 3  # Industrial
        total = E + S + I
        
        print(f"Total de estudiantes: {total}")
        print(f"Electrónica: {E}, Sistemas: {S}, Industrial: {I}")
        print("Los estudiantes de la misma carrera no se distinguen entre sí")
        
        # Usamos permutaciones con repetición
        formas = math.factorial(total) / (math.factorial(E) * math.factorial(S) * math.factorial(I))
        
        print(f"\nFormas de ordenarlos:")
        print(f"   Total de permutaciones sin restricción: {total}! = {math.factorial(total):,}")
        print(f"   Como los de misma carrera son indistinguibles, dividimos por:")
        print(f"   {E}! × {S}! × {I}! = {math.factorial(E)} × {math.factorial(S)} × {math.factorial(I)} = {math.factorial(E) * math.factorial(S) * math.factorial(I)}")
        print(f"   Total de formas: {math.factorial(total):,} / ({math.factorial(E)}×{math.factorial(S)}×{math.factorial(I)}) = {formas:,.0f}")
        
        # Gráfica
        self.graficar_problema_3(E, S, I, formas)
    
    def graficar_problema_3(self, E, S, I, formas):
        carreras = ['Electrónica', 'Sistemas', 'Industrial']
        cantidades = [E, S, I]
        colores = ['#FF9F43', '#54A0FF', '#00D2D3']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfica de distribución por carrera
        ax1.pie(cantidades, labels=carreras, colors=colores, autopct='%1.0f%%', startangle=90)
        ax1.set_title('Distribución de Estudiantes por Carrera')
        
        # Gráfica de resultado
        ax2.bar(['Formas de ordenar'], [formas], color='#A29BFE', alpha=0.7)
        ax2.set_title('Número de Formas de Ordenar')
        ax2.set_ylabel('Cantidad')
        ax2.text(0, formas/2, f'{formas:,.0f}', ha='center', va='center', fontsize=20, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def problema_4(self):
        print("\n" + "="*50)
        print("PROBLEMA 4: Probabilidad con dados")
        print("="*50)
        
        # a) No obtener 7 u 11 en 2 lanzamientos
        print("a) No obtener un total de 7 u 11 en 2 lanzamientos")
        
        # Espacio muestral para 2 dados
        espacio_muestral = 6 * 6
        print(f"   Espacio muestral: 6 × 6 = {espacio_muestral}")
        
        # Combinaciones que suman 7: (1,6),(2,5),(3,4),(4,3),(5,2),(6,1)
        combinaciones_7 = 6
        # Combinaciones que suman 11: (5,6),(6,5)
        combinaciones_11 = 2
        
        combinaciones_favorables = combinaciones_7 + combinaciones_11
        P_favorable = combinaciones_favorables / espacio_muestral
        P_no_7_11 = 1 - P_favorable
        
        print(f"   Combinaciones que suman 7: {combinaciones_7}")
        print(f"   Combinaciones que suman 11: {combinaciones_11}")
        print(f"   Total combinaciones 7 u 11: {combinaciones_favorables}")
        print(f"   P(7 u 11) = {combinaciones_favorables}/{espacio_muestral} = {P_favorable:.4f}")
        print(f"   P(no 7 u 11) = 1 - {P_favorable:.4f} = {P_no_7_11:.4f} ({P_no_7_11*100:.2f}%)")
        
        # b) Obtener 3 veces el 6 en 5 lanzamientos
        print(f"\nb) Obtener 3 veces el número 6 en 5 lanzamientos")
        
        n = 5  # lanzamientos
        k = 3  # éxitos (obtener 6)
        p = 1/6  # probabilidad de éxito
        
        # Usamos distribución binomial
        P_3seis = math.comb(n, k) * (p**k) * ((1-p)**(n-k))
        
        print(f"   Distribución binomial: n={n}, k={k}, p={p:.4f}")
        print(f"   P(X=3) = C(5,3) × (1/6)³ × (5/6)²")
        print(f"          = {math.comb(n, k)} × {p**k:.6f} × {((1-p)**(n-k)):.4f}")
        print(f"          = {P_3seis:.6f} ({P_3seis*100:.4f}%)")
        
        # Gráficas
        self.graficar_problema_4(P_no_7_11, P_3seis)
    
    def graficar_problema_4(self, P_no_7_11, P_3seis):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfica para parte a)
        labels_a = ['No 7 u 11', 'Sí 7 u 11']
        sizes_a = [P_no_7_11, 1-P_no_7_11]
        colors_a = ['#55EFC4', '#FF7675']
        
        ax1.pie(sizes_a, labels=labels_a, colors=colors_a, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Probabilidad en 2 Lanzamientos\n(No obtener 7 u 11)')
        
        # Gráfica para parte b)
        n = 5
        p = 1/6
        k_values = range(n+1)
        prob_values = [math.comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in k_values]
        
        bars = ax2.bar(k_values, prob_values, color='#74B9FF', alpha=0.7)
        ax2.set_title('Distribución Binomial - 5 Lanzamientos\n(Probabilidad de obtener k veces 6)')
        ax2.set_xlabel('Número de veces que sale 6')
        ax2.set_ylabel('Probabilidad')
        ax2.grid(axis='y', alpha=0.3)
        
        # Resaltar k=3
        bars[3].set_color('#E17055')
        
        plt.tight_layout()
        plt.show()
    
    def problema_5(self):
        print("\n" + "="*50)
        print("PROBLEMA 5: Probabilidad de memorias defectuosas")
        print("="*50)
        
        total_diario = 12000
        p_defectuosa = 0.03
        muestra = 600
        defectuosas_deseadas = 12
        
        print(f"Producción diaria: {total_diario:,} memorias")
        print(f"Tasa de defectuosas: {p_defectuosa*100}%")
        print(f"Muestra: {muestra} memorias")
        print(f"Defectuosas deseadas en muestra: {defectuosas_deseadas}")
        
        # Como la muestra es grande respecto a la población, usamos distribución hipergeométrica
        # o aproximación binomial
        
        # Método exacto: distribución hipergeométrica
        N = total_diario
        K = int(total_diario * p_defectuosa)  # Total defectuosas
        n = muestra
        k = defectuosas_deseadas
        
        # P(X = k) = [C(K,k) × C(N-K, n-k)] / C(N,n)
        numerador = math.comb(K, k) * math.comb(N - K, n - k)
        denominador = math.comb(N, n)
        P_exacta = numerador / denominador
        
        print(f"\nSolución exacta (distribución hipergeométrica):")
        print(f"   P(X=12) = [C({K},12) × C({N-K},{n-k})] / C({N},{n})")
        print(f"           = {numerador:.2e} / {denominador:.2e}")
        print(f"           = {P_exacta:.6f} ({P_exacta*100:.4f}%)")
        
        # Aproximación binomial (pues N es grande)
        P_binomial = math.comb(n, k) * (p_defectuosa**k) * ((1-p_defectuosa)**(n-k))
        
        print(f"\nAproximación binomial:")
        print(f"   P(X=12) = C(600,12) × (0.03)¹² × (0.97)⁵⁸⁸")
        print(f"           = {P_binomial:.6f} ({P_binomial*100:.4f}%)")
        
        # Gráfica de distribución
        self.graficar_problema_5(n, p_defectuosa, defectuosas_deseadas)
    
    def graficar_problema_5(self, n, p, k_deseado):
        k_values = range(0, 31)  # De 0 a 30 defectuosas
        prob_values = [math.comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in k_values]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(k_values, prob_values, color='#A3CB38', alpha=0.7)
        plt.title('Distribución de Probabilidad - Memorias Defectuosas\n(600 memorias, 3% tasa de defectos)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Número de memorias defectuosas', fontsize=12)
        plt.ylabel('Probabilidad', fontsize=12)
        
        # Resaltar el valor deseado (k=12)
        bars[k_deseado].set_color('#EA2027')
        plt.axvline(x=k_deseado, color='red', linestyle='--', alpha=0.8)
        
        # Añadir anotación
        prob_k_deseado = prob_values[k_deseado]
        plt.annotate(f'P(X={k_deseado}) = {prob_k_deseado:.6f}', 
                    xy=(k_deseado, prob_k_deseado), 
                    xytext=(k_deseado+2, prob_k_deseado+0.005),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontweight='bold', color='red')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

# Ejecutar el programa
if __name__ == "__main__":
    solucionador = SolucionadorProblemas()
    solucionador.ejecutar()