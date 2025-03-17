#!/bin/bash
set -e

# Script para inicializar PostgreSQL - guarda esto en configs/init-postgresql.sh
# y asegúrate de que sea ejecutable con: chmod +x configs/init-postgresql.sh

# Función para crear base de datos si no existe
create_db_if_not_exists() {
    DB_NAME=$1
    
    # Verificar si la base de datos ya existe
    if psql -U postgres -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
        echo "✅ Base de datos '$DB_NAME' ya existe"
    else
        echo "🔄 Creando base de datos '$DB_NAME'..."
        psql -U postgres -c "CREATE DATABASE $DB_NAME;"
        psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO geo_user;"
        echo "✅ Base de datos '$DB_NAME' creada correctamente"
    fi
}

# Crear base de datos Airflow (además de geo_db que ya existe por defecto)
create_db_if_not_exists "airflow"

echo "✅ Inicialización de PostgreSQL completada"