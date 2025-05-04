import networkx as nx
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
import random
from geopy.distance import great_circle
from haversine import haversine
import openrouteservice

client = openrouteservice.Client(key='5b3ce3597851110001cf624857f9ba1e285e4965996cd6c08382472c')

# Creación del modelo en Pyomo
Model = ConcreteModel()

opcion = int(input("¿Desea ingresar los datos reales o utilizar los valores por defecto? \n1. Reales \n2. Por defecto\n"))

if opcion == 1:
    # --------------------------------------------------------------------------------------------------------------------
    # Parámetros del problema
    n = 4  # Número de nodos (incluyendo CD)
    V = n-1
    n = n + V
    Vehiculos = 3  # Número de vehículos
    Nombre_vehiculo = ["V1", "V2", "V3"]  # Lista con los nombres de los vehículos
    nodo_reabastecimiento = [n-V+i for i in range(1, V+1)]  # Nodos de reabastecimiento
    nodo_inicio = 1  # Nodo de inicio 

    # Conjunto de nodos
    N = range(1, n + 1)
    K = range(1, Vehiculos + 1)

    # Condición climática actual
    Condicion_Climatica = "Normal"  # Opciones: "Normal", "Lluvia", "Nieve", "Tormenta"

    # Parámetros de vehículos
    vehiculos_info = {
        "V1": {"tipo": "Camioneta", "capacidad": 200, "rango": 300},
        "V2": {"tipo": "Dron", "capacidad": 20, "rango": 50},
        "V3": {"tipo": "Dron", "capacidad": 15, "rango": 40}
    }

    # Factores de ajuste por tipo de vehículo
    ajustes_vehiculo = {
        "Camioneta": {"cost_factor": 1.2, "time_factor": 0.9},  # Costo ↑ 20%, Tiempo ↓ 10%
        "Dron": {"cost_factor": 0.85, "time_factor": 1.2}  # Costo ↓ 15%, Tiempo ↑ 20%
    }

    # Factores de ajuste por condición climática
    factores_clima = {
        "Normal": {"cost_factor": 1.0, "time_factor": 1.0},
        "Lluvia": {"cost_factor": 1.1, "time_factor": 1.3},
        "Nieve": {"cost_factor": 1.3, "time_factor": 1.7},
        "Tormenta": {"cost_factor": 1.5, "time_factor": 2.0}
    }

    # Factores climáticos actuales
    factor_clima_cost = factores_clima[Condicion_Climatica]["cost_factor"]
    factor_clima_time = factores_clima[Condicion_Climatica]["time_factor"]

    # Costos operativos
    Pf = 15000  # Precio del combustible (COP por litro)
    Ft = 5000  # Tarifa de flete (COP por km)
    Cm = 700  # Costo de mantenimiento (COP por km)
    Seguros = 300  # Costo de seguros por km (COP)
    Peajes = 2000  # Costo de peajes por km (COP)
    Salarios = 8000  # Costo de salarios de conductor por km (COP)

    # Demandas de los nodos
    demanda = [0, 30, 50, 40]  # CD tiene 0 demanda
    for i in range(1, V+1):
        demanda.append(0)

    coordenadas_reales = True
    # Coordenadas de los nodos
    if not coordenadas_reales:
        coordenadas = {
        1: (4.6486, -74.0703),  # Parque de la 93
        2: (4.6533, -74.0670),  # Zona T
        3: (4.6457, -74.0654),  # Parque El Virrey
        4: (4.6425, -74.0698)   # Calle 85 con Carrera 15
        }
        # Coordenadas de los nodos de reabastecimiento
        for i in range(1, V+1):
            coordenadas[n-V+i] = coordenadas[1]  # CDP (Centro de distribución) como nodo de reabastecimiento      
    else:
        coordenadas = {
            1: (11.5449, -72.9071),  # Bogotá (Centro de distribución)
            2: (11.6000, -72.8500),  # Caserío El Sol
            3: (11.5200, -72.9200),  # Centro de Salud La Esperanza
            4: (11.5800, -72.8800)   # Ranchería Los Pinos
        }
        # Coordenadas de los nodos de reabastecimiento
        for i in range(1, V+1):
            coordenadas[n-V+i] = coordenadas[1]  # CDP (Centro de distribución) como nodo de reabastecimiento     

    # Cálculo de distancias
    def calcular_distancia(nodo1, nodo2, tipo):
        if tipo == "Dron":
            return haversine(coordenadas[nodo1], coordenadas[nodo2])
        else:
            if not coordenadas_reales:           
                routes = client.directions(
                    coordinates=[coordenadas[nodo1][::-1], coordenadas[nodo2][::-1]],
                    profile='driving-car',  # Tipo de perfil de transporte (coche)
                    format='geojson'
                )
                return routes['features'][0]['properties']['segments'][0]['distance']/1000
            else:
                return great_circle(coordenadas[nodo1], coordenadas[nodo2]).kilometers

    # Inicialización de matrices para cada vehículo
    cost = np.zeros((n, n, Vehiculos))
    tiempos_viaje = np.zeros((n, n, Vehiculos))
    matriz_distancias = np.zeros((n, n, Vehiculos))

    ventanas_tiempo = {
        1: (0, 960),  # 6:00 AM - 10:00 PM
        2: (180, 420),  # 9:00 AM - 1:00 PM
        3: (240, 480),  # 10:00 AM - 2:00 PM
        4: (360, 720),   # 12:00 PM - 6:00 PM
    }
    for i in range(1, V+1):
        ventanas_tiempo[n-V+i] = ventanas_tiempo[1]

    # Ajuste de matrices según vehículo y clima
    for k in range(Vehiculos):
        vehiculo = Nombre_vehiculo[k]
        tipo_vehiculo = vehiculos_info[vehiculo]["tipo"]
        factor_cost_veh = ajustes_vehiculo[tipo_vehiculo]["cost_factor"]
        factor_time_veh = ajustes_vehiculo[tipo_vehiculo]["time_factor"]

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i != j:
                    distancia = round(calcular_distancia(i, j, tipo_vehiculo), 2)
                    matriz_distancias[i - 1, j - 1, k] = distancia  # Se usa la distancia real para cada tipo de vehículo
                    cost[i - 1, j - 1, k] = distancia * (Pf + Ft + Cm + Seguros + Peajes + Salarios) * factor_cost_veh * factor_clima_cost
                    tiempos_viaje[i - 1, j - 1, k] = round(((distancia / 60 * 60) * factor_time_veh * factor_clima_time), 2)

    # Convertir a listas si es necesario
    cost = cost.tolist()
    tiempos_viaje = tiempos_viaje.tolist()
    matriz_distancias = matriz_distancias.tolist()

    # Capacidad útil por vehículo
    Capacidad_util = {k + 1: vehiculos_info[Nombre_vehiculo[k]]["capacidad"] for k in range(Vehiculos)}

    # Distancia útil por vehículo
    Distancia_util = {k + 1: vehiculos_info[Nombre_vehiculo[k]]["rango"] for k in range(Vehiculos)}
    # --------------------------------------------------------------------------------------------------------------------
elif opcion == 2:
    # --------------------------------------------------------------------------------------------------------------------
    # Parámetros generales

    # Parámetros del problema
    n = 4  # Número de nodos
    V = n-1
    n = n + V
    Distancia_util = [100, 100, 50]  # Distancia máxima permitida
    Capacidad_util = [50, 50, 50]  # Capacidad máxima del vehículo
    nodo_inicio = 1  # Nodo de inicio 
    nodo_reabastecimiento = [n-V+i for i in range(1, V+1)] 
    Vehiculos = 1  # Número de vehículos
    Nombre_vehiculo = ["V1", "V2", "V3"]  # Lista con los nombres de los vehículos
    Nombre_vehiculo = ["V4"]

    # Demandas de los nodos
    demanda = [0, 50, 10, 10] 
    for i in range(1, V+1):
        demanda.append(0)

    # Condición climática actual
    Condicion_Climatica = "Normal"  # Opciones: "Normal", "Lluvia", "Nieve", "Tormenta"

    # Conjunto de nodos
    N = range(1, n + 1)
    K = range(1, Vehiculos + 1)

    Distancia_util = {k: Distancia_util[k-1] for k in K}
    Capacidad_util = {k: Capacidad_util[k-1] for k in K}

    # Matrices base de costos y tiempos
    cost_base = np.array([
        [0, 10, 10, 10],
        [10, 0, 10, 10],
        [10, 10, 0, 10],
        [10, 10, 10, 0],
    ])

    nuevo_array = []
    primera = True
    for fila in cost_base:
        nueva_fila = []
        for valor in fila:
            nueva_fila.append(valor)
        if primera:
            primera = False
            for i in range(V):
                nueva_fila.append(0)
        else:
            for i in range(V):
                nueva_fila.append(nueva_fila[0])
        nuevo_array.append(nueva_fila)

    for i in range(V):
        nuevo_array.append(nuevo_array[0])

    cost_base = np.array(nuevo_array)

    tiempos_base = np.array([
        [0, 10, 10, 10],
        [10, 0, 10, 10],
        [10, 10, 0, 10],
        [10, 10, 10, 0],
    ])

    nuevo_array = []
    primera = True
    for fila in tiempos_base:
        nueva_fila = []
        for valor in fila:
            nueva_fila.append(valor)
        if primera:
            primera = False
            for i in range(V):
                nueva_fila.append(0)
        else:
            for i in range(V):
                nueva_fila.append(nueva_fila[0])
        nuevo_array.append(nueva_fila)

    for i in range(V):
        nuevo_array.append(nuevo_array[0])

    tiempos_base = np.array(nuevo_array)

    # Diccionario de ajustes por tipo de vehículo
    ajustes_vehiculo = {
        "V1": {"cost_factor": 1.2, "time_factor": 0.9},  # Costo ↑ 20%, Tiempo ↓ 10%
        "V2": {"cost_factor": 0.85, "time_factor": 1.2},  # Costo ↓ 15%, Tiempo ↑ 20%
        "V3": {"cost_factor": 0.75, "time_factor": 1.5},  # Costo ↓ 25%, Tiempo ↑ 50%
        "V4": {"cost_factor": 1.0, "time_factor": 1.0}   # Sin cambios
    }

    # Diccionario de ajustes por condición climática
    factores_clima = {
        "Normal":   {"cost_factor": 1.0,  "time_factor": 1.0},  # Sin cambios
        "Lluvia":   {"cost_factor": 1.1,  "time_factor": 1.3},  # Costo ↑ 10%, Tiempo ↑ 30%
        "Nieve":    {"cost_factor": 1.3,  "time_factor": 1.7},  # Costo ↑ 30%, Tiempo ↑ 70%
        "Tormenta": {"cost_factor": 1.5,  "time_factor": 2.0}   # Costo ↑ 50%, Tiempo ↑ 100%
    }

    # Obtener los factores climáticos actuales
    factor_clima_cost = factores_clima[Condicion_Climatica]["cost_factor"]
    factor_clima_time = factores_clima[Condicion_Climatica]["time_factor"]

    # Inicializar matrices 4x4xVehiculos
    cost = np.zeros((n, n, Vehiculos))
    tiempos_viaje = np.zeros((n, n, Vehiculos))

    # Ajustar valores según el tipo de vehículo y la condición climática
    for k in range(Vehiculos):
        vehiculo = Nombre_vehiculo[k]  # Obtener el nombre del vehículo
        factor_cost_veh = ajustes_vehiculo.get(vehiculo, {"cost_factor": 1.0})["cost_factor"]
        factor_time_veh = ajustes_vehiculo.get(vehiculo, {"time_factor": 1.0})["time_factor"]

        # Aplicar ajustes progresivos (primero por vehículo, luego por clima)
        cost[:, :, k] = cost_base * factor_cost_veh * factor_clima_cost
        tiempos_viaje[:, :, k] = tiempos_base * factor_time_veh * factor_clima_time

    # Convertir a listas si es necesario
    cost = cost.tolist()
    tiempos_viaje = tiempos_viaje.tolist()

    # Ventanas de tiempo para cada nodo
    ventanas_tiempo = {1: (0, 1000), 4: (5, 15), 3: (15, 25), 2: (25, 35) } 
    for i in range(1, V+1):
        ventanas_tiempo[n-V+i] = ventanas_tiempo[1]


    # Distancias entre nodos
    distancias = {
        (1, 2): 10, (1, 3): 15, (1, 4): 20,
        (2, 1): 10, (2, 3): 5,  (2, 4): 15,
        (3, 1): 15, (3, 2): 5,  (3, 4): 10,
        (4, 1): 20, (4, 2): 15, (4, 3): 10,
    }

    for i in range(1, V+1):
        nodo = n-V+i
        distancias[(1, nodo)] = 0
        distancias[(nodo, 1)] = 0
    
    for i in range(1, V+1):
        for j in range(1, n-V+1):
            if j != nodo_inicio:
                nodo = n-V+i
                distancias[(nodo, j)] = distancias[(1, j)]
                distancias[(j, nodo)] = distancias[(j, 1)]

    # Crear la matriz de distancias de tamaño (n, n, Vehiculos)
    matriz_distancias = np.zeros((n, n, Vehiculos))
    for (i, j), distancia in distancias.items():
        for k in K: 
            matriz_distancias[i-1, j-1, k-1] = distancia 
    # ------------------------------------------------------------------------------------------------------------------------

# VARIABLES DEL MODELO
Model.x = Var(N, N, K, domain=Binary)  # Variable binaria de decisión (ruta)
Model.carga = Var(N, K, domain=NonNegativeIntegers)  # Carga en cada nodo
Model.distancia = Var(N, K, domain=NonNegativeIntegers)  # Distancia recorrida
Model.t = Var(N, K, domain=NonNegativeIntegers)  # Tiempo de llegada a cada nodo
Model.u = Var(N, K, domain=NonNegativeIntegers, bounds=(1, n-1))  # Orden de visita de los nodos
Model.particion = Var(N, K, domain=Reals, bounds=(0, 1))  # Parte de demanda que lleva cada vehículo
Model.z = Var(N, K, within=Binary) # Variable auxiliar para la partición de la carga

# FUNCIÓN OBJETIVO: Minimizar la distancia recorrida
Model.obj = Objective(expr=sum(Model.x[i, j, k] * cost[i - 1][j - 1][k - 1] * tiempos_viaje[i-1][j-1][k-1] for i in N for j in N for k in K), sense=minimize)


# RESTRICCIONES

# Restricciones iniciales
# --------------------------------------------------------------------------------------------------------------------
for k in K:
    Model.carga[nodo_inicio, k].fix(0)  # Carga inicial
    Model.u[nodo_inicio, k].fix(1)  # Numerado de nodos
    Model.t[nodo_inicio, k].fix(ventanas_tiempo[nodo_inicio][0]) # Tiempo de inicio

for k in K:
    for i in nodo_reabastecimiento:
        Model.carga[i, k].fix(0)  
# --------------------------------------------------------------------------------------------------------------------


# Restricciones de tiempo de viaje
# --------------------------------------------------------------------------------------------------------------------

# Máximo tiempo de viaje permitido
M = max(sum(row) for matrix in tiempos_viaje for row in matrix)
def restriccion_tiempo_viaje(Model, i, j, k):
    if i != j and j != nodo_inicio and j not in nodo_reabastecimiento:
        return Model.t[j, k] >= Model.t[i, k] + tiempos_viaje[i-1][j-1][k-1]*Model.x[i, j, k] - (1 - Model.x[i, j, k]) * M
    else:
        return Constraint.Skip
    

Model.tiempo_viaje = Constraint(N, N, K, rule=restriccion_tiempo_viaje)

# Restricción de tiempo máximo permitido
def restriccion_tiempo_max(Model, j, k):
    if j != nodo_inicio and j not in nodo_reabastecimiento:
        return Model.t[j, k] <= ventanas_tiempo[j][1]
    else:
        return Constraint.Skip

Model.tiempo_max = Constraint(N, K, rule=restriccion_tiempo_max)

# Restricción de tiempo mínimo permitido
def restriccion_tiempo_min(Model, j, k):
    if j != nodo_inicio and j not in nodo_reabastecimiento:
        return Model.t[j, k] >= sum(ventanas_tiempo[j][0]*Model.x[i, j, k] for i in N if i != j)
    else:
        return Constraint.Skip

Model.tiempo_min = Constraint(N, K, rule=restriccion_tiempo_min)
# --------------------------------------------------------------------------------------------------------------------


# Restricciones de capacidad 
# --------------------------------------------------------------------------------------------------------------------
# Capacidad de carga acumulada
def restriccion_carga(Model, i, j, k):
    if i != j and j != nodo_inicio and j not in nodo_reabastecimiento:
        return Model.carga[j, k] >= Model.carga[i, k] + demanda[j-1] - Capacidad_util[k]*(1 - Model.x[i, j, k])
    else:
        return Constraint.Skip
Model.capacidad_acumulada = Constraint(N, N, K, rule=restriccion_carga)


# Restricción para inicializar la variable auxiliar z si un trabajador k visita el nodo j
""" Asegurar que z[j, k] = 1 si el nodo j es visitado por el vehículo k y z[j, k] = 0 en caso contrario. """
def def_z(Model, j, k):
    if j != nodo_inicio and j not in nodo_reabastecimiento:
        return Model.z[j, k] == sum(Model.x[i, j, k] for i in N if i != j)
    else:
        return Constraint.Skip

Model.relacion_z = Constraint(N, K, rule=def_z)

# Restricción para inicializar la partición con la varuiable auxiliar z
def restriccion_parte_carga(Model, j, k):
    if j != nodo_inicio and j not in nodo_reabastecimiento:
        return Model.z[j, k] >= Model.particion[j, k]
    else:
        return Constraint.Skip

Model.particion_acumulada = Constraint(N, K, rule=restriccion_parte_carga)

# Restricción suma de particiones es igual a 1
def restriccion_unica_particion(Model, j):
    if j != nodo_inicio and j not in nodo_reabastecimiento:
        return sum(Model.particion[j, k] for k in K) == 1
    else:
        return Constraint.Skip

Model.unica_particion = Constraint(N, rule=restriccion_unica_particion)

# Carga máxima del vehículo
def carga_limite(Model, i, k):
    return Model.carga[i, k] <= Capacidad_util[k]

Model.carga_limite_veh = Constraint(N,K, rule=carga_limite)
# --------------------------------------------------------------------------------------------------------------------


# Restricciones de distancia
# --------------------------------------------------------------------------------------------------------------------

# Restricción de distancia acumulada
def restriccion_distancia(Model, i, j, k):
    if i != j and j != nodo_inicio:
        return Model.distancia[j, k] >= Model.distancia[i, k] + matriz_distancias[i - 1][j - 1][k - 1]*Model.x[i, j, k] - Distancia_util[k] * (1 - Model.x[i, j, k])
    else:
        return Constraint.Skip

Model.distancia_acumulada = Constraint(N, N, K, rule=restriccion_distancia)

# Límite de distancia máxima
def distancia_limite(Model, i, k):
    return Model.distancia[i, k] <= Distancia_util[k]

Model.distancia_limite_veh = Constraint(N, K, rule=distancia_limite)
# --------------------------------------------------------------------------------------------------------------------


# Restricciones de flujo
# --------------------------------------------------------------------------------------------------------------------

# Reglas de entrada y salida del nodo inicio
def source_rule(Model, i, k):
    if i == nodo_inicio: 
        return sum(Model.x[i, j, k] for j in N if i != j) == 1
    else:
        return Constraint.Skip

Model.source = Constraint(N, K, rule=source_rule)

# Regla de llegada al nodo inicio
def destination_rule(Model, j, k):
    if j == nodo_inicio:
        return sum(Model.x[i, j, k] for i in N if i != j) == 1
    else:
        return Constraint.Skip

Model.destination = Constraint(N, K, rule=destination_rule)

# Restricción de intermedios (flujo de nodos)
def intermediate_rule(Model, i, k):
    if i != nodo_inicio:
        return sum(Model.x[i, j, k] for j in N) - sum(Model.x[j, i, k] for j in N) == 0
    else:
        return Constraint.Skip

Model.intermediate = Constraint(N, K, rule=intermediate_rule)

# Se visita cada nodo exactamente una vez por vehículo
def destinos_alcanzables(Model, j):
    if j not in [nodo_inicio, nodo_reabastecimiento]:
        return sum(Model.x[i, j, k] for i in N for k in K if i != j) == 1
    else:
        return Constraint.Skip

Model.destinos = Constraint(N, rule=destinos_alcanzables)

# Evitar ciclos
def anti_ciclos(Model, i, j, k):
    if i != nodo_inicio and j != nodo_inicio:
        return Model.u[i, k] - Model.u[j, k] + n * Model.x[i, j, k] <= n - 1
    else:
        return Constraint.Skip
Model.sin_ciclos = Constraint(N, N, K, rule=anti_ciclos)


# --------------------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------------------------------
# Solución del Multi-TSP
solver = SolverFactory('glpk')
results = solver.solve(Model, tee=True)

Model.display()

if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:
    print("\n✅ Solución óptima encontrada:")

    ruta_tsp = {}
    costo_vehiculo = {k: 0 for k in K}
    distancia_vehiculo = {k: 0 for k in K}
    tiempo_vehiculo = {k: 0 for k in K}
    demanda_vehiculo = {k: 0 for k in K}

    # Obtener rutas, costos y distancias
    for i in N:
        for j in N:
            for k in K:
                if Model.x[i, j, k].value and Model.x[i, j, k].value > 0.5:
                    ruta_tsp[(i, k)] = j
                    costo_vehiculo[k] += cost[i-1][j-1][k-1]  # Ajuste en el costo
                    distancia_vehiculo[k] += matriz_distancias[i-1][j-1][k-1]  # Usar matriz base para la distancia

    # Calcular tiempos de llegada y tiempos totales por vehículo
    for k in K:
        tiempos_llegada = [
            Model.t[j, k].value for j in N 
            if Model.t[j, k].value is not None and (j, k) in ruta_tsp
        ]
        
        tiempo_vehiculo[k] = sum(tiempos_llegada) if tiempos_llegada else 0

        # Encontrar el último nodo visitado por el vehículo k
        if tiempos_llegada:
            valores_validos = [j for j in N if (j, k) in ruta_tsp and Model.t[j, k].value is not None]
            if valores_validos:
                ultimo_nodo = max(valores_validos, key=lambda x: Model.t[x, k].value)
            else:
                ultimo_nodo = None  # o algún valor por defecto que tenga sentido
        else:
            ultimo_nodo = nodo_inicio

        # Agregar el tiempo de regreso al nodo inicial
        tiempo_regreso = tiempos_viaje[ultimo_nodo-1][nodo_inicio-1][k-1]
        tiempo_vehiculo[k] += tiempo_regreso

        # Calcular demanda total abastecida
        demanda_vehiculo[k] = sum(
            demanda[j-1]*Model.particion[j, k].value for i in N for j in N 
            if Model.x[i, j, k].value and Model.x[i, j, k].value > 0.5 and Model.particion[j, k].value
        )

    # Definir nombres de nodos según la opción seleccionada
    if opcion == 1:
        nombres_nodos = {
            1: ("Depósito Principal\n(Riohacha)" if coordenadas_reales else "Parque de la 93 (Origen)"),
            2: ("Caserío El Sol" if coordenadas_reales else "Zona T"), 
            3: ("Centro de Salud\nLa Esperanza" if coordenadas_reales else "Parque El Virrey"),
            4: ("Ranchería Los Pinos" if coordenadas_reales else "Calle 85 con Carrera 15")
        }
    else:
        nombres_nodos = {
            1: "CDP",
            2: "C2",
            3: "C3",
            4: "C4"
        }

        for i in range(1, V+1):
            nombres_nodos[n-V+i] = "CD"

    # Imprimir rutas
    print("\n  📌 Rutas del Multi-TSP:")
    for k in K:
        print(f"\n  Vehículo {k} ({Nombre_vehiculo[k-1]}):", end=" ")
        
        nodo_actual = nodo_inicio  
        recorrido = [nodo_actual]
        visitados = set()
        max_iter = len(N) * 2
        Valores = True
        paso_por_clientes = 0

        for _ in range(max_iter):
            if (nodo_actual, k) in ruta_tsp:
                nodo_siguiente = ruta_tsp[(nodo_actual, k)]
                
                if nodo_siguiente in visitados and nodo_siguiente != nodo_inicio:
                    print(" [⚠ ERROR] Se detectó un ciclo.")
                    break
                
                recorrido.append(nodo_siguiente)
                if nodo_siguiente not in nodo_reabastecimiento:
                    paso_por_clientes += 1
                visitados.add(nodo_siguiente)
                nodo_actual = nodo_siguiente

                if nodo_actual == nodo_inicio and paso_por_clientes > 0:
                    break  
            else:
                print(" ⚠  Vehiculo no Utilizado.")
                Valores = False
                break

        # Filtrar repeticiones de CD antes de regresar a nodo_inicio
        recorrido_filtrado = []
        for i in range(len(recorrido)):
            actual = recorrido[i]
            siguiente = recorrido[i + 1] if i + 1 < len(recorrido) else None
            # Eliminar CD duplicados antes de CD o nodo_inicio
            if actual in nodo_reabastecimiento and (siguiente in nodo_reabastecimiento or siguiente == nodo_inicio):
                continue
            recorrido_filtrado.append(actual)

        for i in range(len(recorrido_filtrado)):
            nodo = recorrido_filtrado[i]
            nombre_nodo = nombres_nodos[1].replace("\n", " ") if nodo in nodo_reabastecimiento else nombres_nodos[nodo].replace("\n", " ")

            if i == len(recorrido_filtrado) - 1:
                recorrido_filtrado[i] = "    🏠  " + nombres_nodos[1].replace("\n", " ") + "\n"
            elif i == 0:
                recorrido_filtrado[i] = "      Ubicaciones recorridas:\n\n      🏠  " + nombres_nodos[1].replace("\n", " ")
            else:
                if nodo in nodo_reabastecimiento:
                    recorrido_filtrado[i] = "      ➡️  CDP (Reabastecimiento) ⬇️"
                else:
                    recorrido_filtrado[i] = "      ➡️  " + nombre_nodo + " ⬇️"

        if Valores:
            print("\n")
            print("\n  ".join(map(str, recorrido_filtrado)))
            print(f"      📌 Valores de la Ruta:\n")
            print(f"        💰 Costo Total: {costo_vehiculo[k]}")
            print(f"        📏 Distancia Total: {distancia_vehiculo[k]}")
            print(f"        ⏳ Tiempo Total: {tiempo_vehiculo[k]}")
            print(f"        📦 Demanda Abastecida: {demanda_vehiculo[k]}")
        else:
            Ubicacion = nombres_nodos[nodo_inicio].replace("\n", " ")
            print(f"        📌 Vehículo {Nombre_vehiculo[k-1]} en {Ubicacion}")

    # Cálculo del costo total del sistema
    costo_total = sum(costo_vehiculo.values())
    distancia_total = sum(distancia_vehiculo.values())
    tiempo_total = sum(tiempo_vehiculo.values())
    demanda_total = sum(demanda_vehiculo.values())
    print(
        f"\n 💲 Costo total del sistema: {costo_total: .2f} COP "
        f"\n 📏 Distancia total recorrida: {distancia_total: .2f} Km" 
        f"\n ⏳ Tiempo total de recorrido: {tiempo_total} minutos y Tiempo real: {max(tiempo_vehiculo.values())} minutos" 
        f"\n 📦 Demanda total abastecida: {demanda_total} Unidades")

    # --------------------------------------------------------------------------------------------------------------------

    nodos_reabastecimiento_d = [nodo_inicio] + nodo_reabastecimiento

    # Crear el grafo dirigido
    G = nx.DiGraph()
    for i in N:
        if i not in nodo_reabastecimiento:
            G.add_node(i)

    random.seed(42)
    colores_vehiculos = {k: (random.random(), random.random(), random.random()) for k in K}

    # Lista de aristas con curvatura y colores por vehículo
    aristas_tsp = []
    vehiculo_rutas = {k: [] for k in K}

    for i in N:
        for j in N:
            for k in K:
                if Model.x[i, j, k].value and Model.x[i, j, k].value > 0.5:
                    origen = 1 if i in nodo_reabastecimiento else i
                    destino = 1 if j in nodo_reabastecimiento else j
                    if origen != destino:
                        aristas_tsp.append((origen, destino, k))
                        vehiculo_rutas[k].append((origen, destino))
                        G.add_edge(origen, destino)

    # Posicionamiento de nodos con distancia entre ellos
    pos = nx.spring_layout(G, seed=42, k=2.0, scale=2)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.canvas.manager.set_window_title("Ruteo Vehicular")

    # Dibujar nodos con tamaño reducido a 450
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=450, ax=ax)

    # Dibujar arcos con curvatura y colores por vehículo
    for (i, j, k) in aristas_tsp:
        color = colores_vehiculos[k]
        nx.draw_networkx_edges(
            G, pos, edgelist=[(i, j)], edge_color=[color], width=2, ax=ax,
            arrowstyle='-|>', arrowsize=12, connectionstyle=f'arc3,rad={(k - len(K)/2) * 0.2}'
        )

    # Dibujar etiquetas de los nodos centradas
    for node, (x, y) in pos.items():
        if node not in nodo_reabastecimiento:
            nombre = nombres_nodos[node].title()
            plt.text(
                x, y, nombre, fontsize=10, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
            )

    plt.title(f"Ruteo Vehicular para {Vehiculos} Vehículos", fontsize=14, fontweight='bold')

    # Agregar índice de colores con nombres de vehículos a la derecha
    x_legend = 1.05
    y_start = 0.8

    for idx, k in enumerate(K):
        color = colores_vehiculos[k]
        plt.text(
            x_legend, y_start - (idx * 0.08), f"{Nombre_vehiculo[k - 1]}",
            fontsize=10, bbox=dict(facecolor=color, alpha=0.6, edgecolor='black'), transform=ax.transAxes
        )

    # Ajustar márgenes
    plt.subplots_adjust(left=0.1, right=0.8, bottom=0.2)

    # --------------------------------------------------------------------------------------------------------------------

    # Grafo de rutas
    # --------------------------------------------------------------------------------------------------------------------
   # Crear colores únicos para cada vehículo
    random.seed(42)
    colores_vehiculos = {k: (random.random(), random.random(), random.random()) for k in K}

    # Generar un gráfico separado para cada vehículo
    for k_resaltado in K:
        # Crear el grafo dirigido
        G = nx.DiGraph()
        for i in N:
            if i not in nodo_reabastecimiento:
                G.add_node(i)

        # Construir aristas solo para el vehículo resaltado con etiquetas
        arcos_resaltados = []
        edge_labels = {}

        for i in N:
            for j in N:
                if Model.x[i, j, k_resaltado].value and Model.x[i, j, k_resaltado].value > 0.5 :
                    origen = 1 if i in nodo_reabastecimiento else i
                    destino = 1 if j in nodo_reabastecimiento else j
                    if origen != destino:
                        arcos_resaltados.append((origen, destino))

                        # Obtener valores del arco (usamos i y j originales para las métricas)
                        costo_arco = cost[i - 1][j - 1][k_resaltado - 1]  
                        distancia_arco = matriz_distancias[i - 1][j - 1][k_resaltado - 1]  
                        tiempo_viaje = tiempos_viaje[i - 1][j - 1][k_resaltado - 1]  
                        demanda_val = demanda[j - 1] * Model.particion[j, k_resaltado].value if Model.particion[j, k_resaltado].value else 0

                        edge_labels[(origen, destino)] = f"Cost: {costo_arco:.0f} / Dist: {distancia_arco:.2f} / Temp: {tiempo_viaje:.2f} / Dem: {demanda_val:.0f}"


        # Si el vehículo no tiene arcos, se omite el gráfico
        if not arcos_resaltados:
            continue

        # Posiciones de nodos optimizadas
        pos = nx.kamada_kawai_layout(G)

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.canvas.manager.set_window_title(f"Ruteo Vehicular - Vehículo {k_resaltado}")

        # Dibujar nodos más pequeños
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=450, ax=ax)

        # Dibujar solo los arcos del vehículo resaltado con flechas más delgadas
        nx.draw_networkx_edges(
            G, pos, edgelist=arcos_resaltados, edge_color=[colores_vehiculos[k_resaltado]], width=1.5, ax=ax,
            arrowstyle='-|>', arrowsize=15
        )

        # Dibujar etiquetas de nodos centradas
        labels = {node: nombres_nodos[node] for node in G.nodes if node not in nodo_reabastecimiento}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold', verticalalignment='center')

        # Dibujar etiquetas en los arcos con costo, distancia y tiempo
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, bbox=dict(facecolor='white', edgecolor='none', alpha=1))

        plt.title(f"Ruteo Vehicular - Vehículo {k_resaltado}", fontsize=14, fontweight='bold')

        # Mostrar sin bloquear ejecución
        plt.show(block=False)

    # Mantener abiertas todas las gráficas
    plt.show()
    # --------------------------------------------------------------------------------------------------------------------

else:
    print("\n❌ No se encontró una solución óptima.")
    print(f"Estado del solver: {results.solver.status}")
    print(f"Condición de terminación: {results.solver.termination_condition}")
