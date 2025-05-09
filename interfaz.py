from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os
import json
import threading

app = Flask(__name__)
app.secret_key = 'clave_secreta_muy_segura'

# Carpeta para guardar archivos subidos
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Datos temporales para almacenar la información
data = {
    'vehicles': [],
    'depots': [],
    'clients': []
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vehicles', methods=['GET', 'POST'])
def vehicles():
    if request.method == 'POST':
        if 'csv_file' in request.files:
            file = request.files['csv_file']
            if file.filename != '':
                try:
                    df = pd.read_csv(file)
                    data['vehicles'] = df.to_dict('records')
                    flash('Archivo CSV de vehículos cargado correctamente', 'success')
                    return redirect(url_for('vehicles'))
                except Exception as e:
                    flash(f'Error al procesar el archivo: {str(e)}', 'danger')
    return render_template('vehicles.html', vehicles=data['vehicles'])

@app.route('/depots', methods=['GET', 'POST'])
def depots():
    if request.method == 'POST':
        if 'csv_file' in request.files:
            file = request.files['csv_file']
            if file.filename != '':
                try:
                    df = pd.read_csv(file)
                    data['depots'] = df.to_dict('records')
                    flash('Archivo CSV de depósitos cargado correctamente', 'success')
                    return redirect(url_for('depots'))
                except Exception as e:
                    flash(f'Error al procesar el archivo: {str(e)}', 'danger')
    return render_template('depots.html', depots=data['depots'])

@app.route('/clients', methods=['GET', 'POST'])
def clients():
    if request.method == 'POST':
        if 'csv_file' in request.files:
            file = request.files['csv_file']
            if file.filename != '':
                try:
                    df = pd.read_csv(file)
                    data['clients'] = df.to_dict('records')
                    flash('Archivo CSV de clientes cargado correctamente', 'success')
                    return redirect(url_for('clients'))
                except Exception as e:
                    flash(f'Error al procesar el archivo: {str(e)}', 'danger')
    return render_template('clients.html', clients=data['clients'])

# filepath: c:\Users\esteb\Downloads\Proyecto_B_Caso2\interfaz.py
@app.route('/export', methods=['GET', 'POST'])
def export():
    if request.method == 'POST':
        # Guardar los datos en archivos CSV
        try:
            if data['vehicles']:
                pd.DataFrame(data['vehicles']).to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'vehicles.csv'), index=False)
            if data['depots']:
                pd.DataFrame(data['depots']).to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'depots.csv'), index=False)
            if data['clients']:
                pd.DataFrame(data['clients']).to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'clients.csv'), index=False)
            flash('Datos exportados correctamente', 'success')
        except Exception as e:
            flash(f'Error al exportar datos: {str(e)}', 'danger')

        # Intentar apagar el servidor Flask
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            print("Advertencia: No se puede apagar el servidor Flask desde esta solicitud.")
        else:
            func()
        return "Datos exportados y servidor detenido. Puede cerrar esta ventana."

    # Mostrar estadísticas básicas
    stats = {
        'vehicles_count': len(data['vehicles']),
        'depots_count': len(data['depots']),
        'clients_count': len(data['clients']),
    }
    return render_template('export.html', stats=stats)

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)