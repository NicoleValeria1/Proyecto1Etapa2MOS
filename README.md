# Proyecto1Etapa2MOS

Este proyecto implementa un modelo de optimización logística para la empresa **LogistiCo**, que distribuye insumos médicos a comunidades remotas de La Guajira. Se utiliza un enfoque de programación entera mixta (MIP) con el paquete Pyomo y se soportan distintos escenarios logísticos mediante el uso de archivos CSV externos.

## 📂 Archivos requeridos

Ubica los siguientes archivos en el mismo directorio que `Proyecto MOS.py`:

- `Proyecto MOS.py`: Script principal que define y resuelve el modelo.
- `clients.csv`: Lista de comunidades/clientes. Debe contener:
  - `Latitude`, `Longitude`: Coordenadas geográficas.
  - `Demand`: Demanda en unidades.
  - `TimeWindow` (solo para Casos 2 y 3): Ej. `"08:00-12:00"`.

- `depots.csv`: Información del centro de distribución (generalmente uno solo).
  - `Latitude`, `Longitude`

- `vehicles.csv`: Información de la flota disponible. Debe incluir:
  - `VehicleID`: Identificador único del vehículo.
  - `Capacity`: Capacidad de carga (unidades).
  - `Range`: Autonomía en km.
  - `Speed`: Velocidad (dejar vacío si es camioneta para diferenciarla de drones).
