
from pgmpy.models import DiscreteBayesianNetwork 
from pgmpy.factors.discrete import TabularCPD  
from pgmpy.inference import VariableElimination

# 1. CREAR LA ESTRUCTURA DE LA RED BAYESIANA
print("=== CREANDO RED BAYESIANA ===") 

# Definir las relaciones entre variables
# Batería -> Motor arranca -> Radio funciona
modelo = DiscreteBayesianNetwork([  
    ('Bateria', 'Motor_Arranca'),      # La batería afecta si el motor arranca
    ('Motor_Arranca', 'Radio_Funciona') # Si el motor arranca afecta a la radio
])

print("Estructura de la red creada:")
print(f"Nodos: {modelo.nodes()}")
print(f"Aristas: {modelo.edges()}")
print()

# 2. DEFINIR LAS PROBABILIDADES (CPD - Conditional Probability Distributions)
print("=== DEFINICIÓN DE PROBABILIDADES ===")

# Probabilidad de la Batería (variable padre)
# 0 = Batería descargada, 1 = Batería cargada
cpd_bateria = TabularCPD(
    variable='Bateria',
    variable_card=2,  # 2 estados posibles
    values=[[0.1], [0.9]]  # 10% descargada, 90% cargada
)

# Probabilidad de que el Motor arranque DADA la Batería
# P(Motor_Arranca | Bateria)
cpd_motor = TabularCPD(
    variable='Motor_Arranca',
    variable_card=2,  # 0 = no arranca, 1 = arranca
    values=[
        [0.99, 0.01],  # P(no arranca | batería cargada) = 1%
        [0.01, 0.99]   # P(arranca | batería cargada) = 99%
    ],
    evidence=['Bateria'],
    evidence_card=[2]
)

# Probabilidad de que la Radio funcione DADO el Motor
# P(Radio_Funciona | Motor_Arranca)
cpd_radio = TabularCPD(
    variable='Radio_Funciona',
    variable_card=2,  # 0 = no funciona, 1 = funciona
    values=[
        [0.95, 0.05],  # P(no funciona | motor arranca) = 5%
        [0.05, 0.95]   # P(funciona | motor arranca) = 95%
    ],
    evidence=['Motor_Arranca'],
    evidence_card=[2]
)

# Agregar las probabilidades al modelo
modelo.add_cpds(cpd_bateria, cpd_motor, cpd_radio)

# Verificar que el modelo es válido
print("¿Modelo válido?", modelo.check_model())
print()

# 3. MOSTRAR LAS PROBABILIDADES DEFINIDAS
print("=== PROBABILIDADES CONDICIONALES ===")
print("Probabilidad de la Batería:")
print(cpd_bateria)
print("\nProbabilidad del Motor dado el estado de la Batería:")
print(cpd_motor)
print("\nProbabilidad de la Radio dado el estado del Motor:")
print(cpd_radio)
print()

# 4. INFERENCIA - HACER PREGUNTAS A LA RED
print("=== INFERENCIA EN LA RED ===")
inferencia = VariableElimination(modelo)

# a) Probabilidad a priori de que la radio funcione
resultado = inferencia.query(variables=['Radio_Funciona'])
print("1. Probabilidad de que la radio funcione (sin evidencia):")
print(resultado)
print()

# b) Si la radio NO funciona, ¿cuál es la probabilidad de que la batería esté descargada?
resultado = inferencia.query(
    variables=['Bateria'], 
    evidence={'Radio_Funciona': 0}  # Evidencia: radio no funciona
)
print("2. Si la radio NO funciona, probabilidad de la batería:")
print(resultado)
print()

# c) Si el motor NO arranca, ¿cuál es la probabilidad de que la radio funcione?
resultado = inferencia.query(
    variables=['Radio_Funciona'], 
    evidence={'Motor_Arranca': 0}  # Evidencia: motor no arranca
)
print("3. Si el motor NO arranca, probabilidad de que la radio funcione:")
print(resultado)
print()

# 5. PREDICCIÓN - ¿QUÉ PASA SI...?
print("=== PREDICCIÓN ===")
print("4. Si la batería está descargada, ¿probabilidad de que el motor arranque y la radio funcione?")

# Probabilidad conjunta P(Motor_Arranca, Radio_Funciona | Bateria=0)
resultado = inferencia.query(
    variables=['Motor_Arranca', 'Radio_Funciona'], 
    evidence={'Bateria': 0}  # Batería descargada
)
print(resultado)

print("\n=== FIN DEL PROGRAMA ===") 