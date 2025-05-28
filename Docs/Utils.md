# Guía de Utilidades y Herramientas

Esta guía contiene comandos y herramientas útiles para ejecutar y monitorear el entrenamiento del modelo U-Net de forma eficiente.

## 🖥️ Ejecución con Screen (Recomendado)

### ¿Qué es Screen?
Screen es una utilidad que permite ejecutar procesos en segundo plano, manteniéndolos activos incluso después de cerrar la terminal o desconectarse de SSH.

### Comandos Básicos de Screen

#### 1. Iniciar una nueva sesión de screen
```bash
# Crear sesión con nombre descriptivo
screen -S unet_training

# O crear sesión simple
screen
```

#### 2. Ejecutar el entrenamiento
```bash
# Una vez dentro de screen, ejecutar:
cd ~/computer-vision/app
python main.py

# O ejecutar directamente al crear la sesión:
screen -S unet_training -dm bash -c 'cd ~/computer-vision/app && python main.py'
```

#### 3. Despegarse de la sesión (Detach)
```bash
# Presionar las teclas (mientras el entrenamiento corre):
Ctrl + A, luego D
```

#### 4. Listar sesiones activas
```bash
screen -ls
```

#### 5. Reconectarse a una sesión (Attach)
```bash
# Reconectarse por nombre
screen -r unet_training

# O por ID si hay una sola sesión
screen -r
```

#### 6. Terminar una sesión
```bash
# Dentro de la sesión, presionar:
Ctrl + A, luego K
# O simplemente:
exit
```

### Ejemplo Completo de Uso

```bash
# 1. Crear sesión y ejecutar entrenamiento
screen -S unet_training

# 2. Navegar al directorio
cd ~/computer-vision/app

# 3. Verificar que todo esté listo
ls -la  # Verificar archivos
ls persons/project/  # Ver datasets

# 4. Ejecutar entrenamiento
python main.py

# 5. Despegarse (Ctrl+A, D) para dejar corriendo
# La terminal se puede cerrar y el proceso sigue ejecutándose

# 6. Más tarde, reconectarse para ver progreso
screen -r unet_training
```

## 📊 Monitoreo y Logs

### Ver logs en tiempo real
```bash
# En otra terminal (o después de detach):
tail -f logs/training_*.log

# Ver las últimas 50 líneas
tail -n 50 logs/training_*.log

# Buscar métricas específicas
grep "IoU" logs/training_*.log | tail -10
grep "Loss" logs/training_*.log | tail -10
```

### Monitorear progreso
```bash
# Ver checkpoints guardados
ls -la checkpoints/

# Ver gráficas generadas
ls -la plots/

# Monitorear uso de GPU (si disponible)
nvidia-smi

# Monitorear cada 2 segundos
watch -n 2 nvidia-smi
```

## 🚀 Scripts de Automatización

### Script de inicio automático (Mejorado)
Crear archivo `start_training.sh`:

```bash
#!/bin/bash

# Configuración
PROJECT_DIR="$HOME/computer-vision/app"
SESSION_NAME="unet_training"
EXECUTION_LOG="execution.log"
ERROR_LOG="errors.log"
STARTUP_LOG="training_startup.log"

echo "=== Iniciando entrenamiento U-Net ===" | tee $STARTUP_LOG
echo "Fecha: $(date)" | tee -a $STARTUP_LOG
echo "Directorio: $PROJECT_DIR" | tee -a $STARTUP_LOG

# Verificar que screen esté instalado
if ! command -v screen &> /dev/null; then
    echo "ERROR: Screen no está instalado" | tee -a $STARTUP_LOG
    echo "Instalar con: sudo apt-get install screen" | tee -a $STARTUP_LOG
    exit 1
fi

# Verificar directorio del proyecto
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Directorio del proyecto no encontrado: $PROJECT_DIR" | tee -a $STARTUP_LOG
    exit 1
fi

# Cambiar al directorio del proyecto
cd "$PROJECT_DIR"

# Verificar archivos necesarios
if [ ! -f "main.py" ]; then
    echo "ERROR: main.py no encontrado en $PROJECT_DIR" | tee -a $STARTUP_LOG
    exit 1
fi

if [ ! -d "persons/project" ]; then
    echo "WARNING: Dataset no encontrado en persons/project" | tee -a $STARTUP_LOG
fi

# Limpiar logs anteriores si existen
[ -f "$EXECUTION_LOG" ] && mv "$EXECUTION_LOG" "${EXECUTION_LOG}.backup"
[ -f "$ERROR_LOG" ] && mv "$ERROR_LOG" "${ERROR_LOG}.backup"

# Matar sesión existente si existe
screen -S $SESSION_NAME -X quit 2>/dev/null
sleep 2

# Crear nueva sesión con logging completo
echo "Creando sesión de screen: $SESSION_NAME" | tee -a $STARTUP_LOG
screen -S $SESSION_NAME -dm bash -c "
    echo '=== INICIO DEL ENTRENAMIENTO ===' > $EXECUTION_LOG;
    echo 'Fecha: $(date)' >> $EXECUTION_LOG;
    echo 'Directorio: $(pwd)' >> $EXECUTION_LOG;
    echo 'Usuario: $(whoami)' >> $EXECUTION_LOG;
    echo 'GPU Info:' >> $EXECUTION_LOG;
    nvidia-smi >> $EXECUTION_LOG 2>&1 || echo 'GPU no disponible' >> $EXECUTION_LOG;
    echo '================================' >> $EXECUTION_LOG;
    echo '';
    echo 'Iniciando entrenamiento...';
    python main.py 2>&1 | tee -a $EXECUTION_LOG;
    EXIT_CODE=\$?;
    echo '' >> $EXECUTION_LOG;
    echo '=== FINALIZACIÓN ===' >> $EXECUTION_LOG;
    echo \"Código de salida: \$EXIT_CODE\" >> $EXECUTION_LOG;
    echo \"Fecha de finalización: \$(date)\" >> $EXECUTION_LOG;
    if [ \$EXIT_CODE -eq 0 ]; then
        echo '✅ Entrenamiento completado exitosamente' | tee -a $EXECUTION_LOG;
    else
        echo '❌ Entrenamiento terminó con errores' | tee -a $EXECUTION_LOG;
    fi;
    echo '';
    echo 'Presiona Enter para cerrar la sesión...';
    read
"

# Verificar que la sesión se creó correctamente
sleep 3
if screen -list | grep -q "$SESSION_NAME"; then
    echo "✅ Sesión iniciada exitosamente!" | tee -a $STARTUP_LOG
    echo "" | tee -a $STARTUP_LOG
    echo "=== COMANDOS ÚTILES ===" | tee -a $STARTUP_LOG
    echo "Conectar a sesión:       screen -r $SESSION_NAME" | tee -a $STARTUP_LOG
    echo "Ver ejecución:           tail -f $EXECUTION_LOG" | tee -a $STARTUP_LOG
    echo "Ver logs del modelo:     tail -f logs/training_*.log" | tee -a $STARTUP_LOG
    echo "Buscar errores:          grep -i error $EXECUTION_LOG" | tee -a $STARTUP_LOG
    echo "Terminar sesión:         screen -S $SESSION_NAME -X quit" | tee -a $STARTUP_LOG
    echo "" | tee -a $STARTUP_LOG
    
    # Mostrar status inicial
    echo "=== STATUS INICIAL ===" | tee -a $STARTUP_LOG
    screen -ls | grep "$SESSION_NAME" | tee -a $STARTUP_LOG
else
    echo "❌ Error al crear la sesión" | tee -a $STARTUP_LOG
    exit 1
fi
```

Hacer ejecutable y usar:
```bash
chmod +x start_training.sh
./start_training.sh
```

### Script de monitoreo (Mejorado)
Crear archivo `monitor_training.sh`:

```bash
#!/bin/bash

SESSION_NAME="unet_training"
LOG_DIR="logs"
EXECUTION_LOG="execution.log"

echo "=== Monitor de Entrenamiento U-Net ==="
echo "Fecha: $(date)"
echo

# Verificar si la sesión existe
if screen -list | grep -q "$SESSION_NAME"; then
    echo "✅ Sesión '$SESSION_NAME' está activa"
    PROCESS_STATUS="🟢 EJECUTÁNDOSE"
else
    echo "❌ Sesión '$SESSION_NAME' no encontrada"
    PROCESS_STATUS="🔴 DETENIDO"
    echo "Sesiones disponibles:"
    screen -ls
fi

# Verificar si hay errores críticos
echo
echo "=== Estado del Proceso: $PROCESS_STATUS ==="
if [ -f "$EXECUTION_LOG" ]; then
    # Buscar errores recientes
    ERROR_COUNT=$(grep -ci "error\|exception\|failed\|traceback" "$EXECUTION_LOG" 2>/dev/null || echo "0")
    WARNING_COUNT=$(grep -ci "warning" "$EXECUTION_LOG" 2>/dev/null || echo "0")
    
    echo "Errores encontrados: $ERROR_COUNT"
    echo "Warnings encontrados: $WARNING_COUNT"
    
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo
        echo "🚨 ERRORES RECIENTES:"
        grep -i "error\|exception\|failed" "$EXECUTION_LOG" | tail -3
    fi
    
    # Mostrar progreso si está disponible
    echo
    echo "=== Progreso del Entrenamiento ==="
    if grep -q "Época" "$EXECUTION_LOG" 2>/dev/null; then
        echo "Última época procesada:"
        grep "Época" "$EXECUTION_LOG" | tail -1
    fi
    
    # Mostrar métricas recientes
    if grep -q "IoU\|Loss" "$EXECUTION_LOG" 2>/dev/null; then
        echo "Métricas recientes:"
        grep -E "IoU|Loss.*:" "$EXECUTION_LOG" | tail -3
    fi
else
    echo "❌ Archivo de ejecución no encontrado: $EXECUTION_LOG"
fi

# Mostrar logs internos del modelo
echo
echo "=== Logs Internos del Modelo ==="
if [ -d "$LOG_DIR" ]; then
    LATEST_LOG=$(ls -t $LOG_DIR/training_*.log 2>/dev/null | head -n1)
    if [ -n "$LATEST_LOG" ]; then
        echo "📄 Archivo de log: $LATEST_LOG"
        echo "Últimas 5 líneas:"
        tail -n 5 "$LATEST_LOG"
    else
        echo "No se encontraron logs internos de entrenamiento"
    fi
else
    echo "Directorio de logs no encontrado"
fi

# Mostrar checkpoints
echo
echo "=== Checkpoints y Resultados ==="
if [ -d "checkpoints" ]; then
    CHECKPOINT_COUNT=$(ls checkpoints/*.pth 2>/dev/null | wc -l)
    echo "📊 Total checkpoints: $CHECKPOINT_COUNT"
    
    if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
        echo "Checkpoints más recientes:"
        ls -lt checkpoints/*.pth | head -3 | awk '{print "  " $9 " (" $6 " " $7 " " $8 ")"}'
        
        # Mostrar tamaño del mejor modelo si existe
        if [ -f "checkpoints/best_model.pth" ]; then
            BEST_SIZE=$(ls -lh checkpoints/best_model.pth | awk '{print $5}')
            echo "  🏆 Mejor modelo: $BEST_SIZE"
        fi
    fi
else
    echo "❌ Directorio de checkpoints no encontrado"
fi

# Mostrar gráficas generadas
if [ -d "plots" ]; then
    PLOT_COUNT=$(ls plots/*.png 2>/dev/null | wc -l)
    if [ "$PLOT_COUNT" -gt 0 ]; then
        echo "📈 Gráficas generadas: $PLOT_COUNT"
    fi
fi

# Mostrar uso de recursos
echo
echo "=== Uso de Recursos ==="
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
echo "🖥️  CPU: ${CPU_USAGE}% usado"

MEM_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
echo "💾 RAM: ${MEM_USAGE}% usado"

DISK_USAGE=$(df . | tail -1 | awk '{print $5}')
echo "💿 Disco: $DISK_USAGE usado"

# GPU si está disponible
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    awk -F, '{printf "  Uso: %s%%, Memoria: %s/%s MB, Temp: %s°C\n", $1, $2, $3, $4}'
else
    echo "🎮 GPU: No disponible o no detectada"
fi

# Verificar espacio libre
FREE_SPACE=$(df . | tail -1 | awk '{print $4}')
FREE_SPACE_GB=$((FREE_SPACE / 1024 / 1024))
if [ "$FREE_SPACE_GB" -lt 5 ]; then
    echo "⚠️  ADVERTENCIA: Poco espacio libre ($FREE_SPACE_GB GB)"
fi

echo
echo "=== Comandos Útiles ==="
echo "🔌 Conectar a sesión:      screen -r $SESSION_NAME"
echo "📋 Ver ejecución:          tail -f $EXECUTION_LOG"
echo "📊 Ver logs del modelo:    tail -f $LOG_DIR/training_*.log"
echo "🔍 Buscar errores:         grep -i error $EXECUTION_LOG"
echo "⏹️  Detener entrenamiento:  screen -S $SESSION_NAME -X quit"
echo "🔄 Actualizar monitor:     ./monitor_training.sh"

# Función de auto-refresh (opcional)
echo
read -p "¿Deseas activar auto-refresh cada 30 segundos? (y/N): " AUTO_REFRESH
if [[ $AUTO_REFRESH =~ ^[Yy]$ ]]; then
    echo "Presiona Ctrl+C para detener el monitoreo automático"
    while true; do
        sleep 30
        clear
        echo "🔄 Actualizado: $(date)"
        echo
        $0  # Re-ejecutar este script
    done
fi
```

Usar el monitor:
```bash
chmod +x monitor_training.sh
./monitor_training.sh
```

## 🔧 Configuraciones Adicionales

### Variables de entorno útiles
```bash
# Configurar GPU específica (si hay múltiples)
export CUDA_VISIBLE_DEVICES=0

# Aumentar memoria compartida para DataLoader
export TORCH_NUM_THREADS=4

# Configurar menos verbose para algunas librerías
export TF_CPP_MIN_LOG_LEVEL=2
```

### Optimizaciones del sistema
```bash
# Verificar espacio en disco
df -h

# Limpiar cache si es necesario
sync && echo 3 > /proc/sys/vm/drop_caches  # Como root

# Verificar procesos que usan mucha CPU/memoria
top -o %CPU
htop  # Si está instalado
```

## 📱 Notificaciones

### Notificación por email (opcional)
Crear script `notify_completion.py`:

```python
import smtplib
from email.mime.text import MIMEText
import sys
import os

def send_notification(subject, message):
    # Configurar según tu proveedor de email
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    email = "tu_email@gmail.com"
    password = "tu_app_password"  # App password, no la contraseña normal
    
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = email
    msg['To'] = email
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(email, password)
        server.send_message(msg)
        server.quit()
        print("Notificación enviada exitosamente")
    except Exception as e:
        print(f"Error enviando notificación: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        message = sys.argv[1]
    else:
        message = "El entrenamiento ha finalizado"
    
    send_notification("Entrenamiento U-Net", message)
```

### Integrar notificación en el entrenamiento
Modificar `main.py` para agregar al final de la función `main()`:

```python
# Al final del entrenamiento exitoso
try:
    os.system("python notify_completion.py 'Entrenamiento completado exitosamente'")
except:
    pass
```

## 🛠️ Troubleshooting

### Problemas comunes con Screen

#### Screen no inicia
```bash
# Verificar instalación
which screen

# Instalar si no está disponible
sudo apt-get update
sudo apt-get install screen
```

#### Sesión "perdida"
```bash
# Listar todas las sesiones
screen -ls

# Si aparece como "Detached" o "Dead"
screen -r session_name

# Si está "Dead", limpiar:
screen -wipe
```

#### No se puede reconectar
```bash
# Forzar reconexión
screen -d -r session_name

# O terminar y crear nueva
screen -S session_name -X quit
screen -S session_name
```

### Problemas de entrenamiento

#### Out of Memory (GPU)
```bash
# Reducir batch size en main.py
# Cambiar en config:
'batch_size': 4,  # En lugar de 8
```

#### Entrenamiento muy lento
```bash
# Verificar que esté usando GPU
nvidia-smi

# Verificar en logs si aparece:
# "device: cuda" o "device: cpu"
```

#### Interrupción inesperada
```bash
# Los checkpoints se guardan automáticamente
# Verificar último checkpoint:
ls -la checkpoints/

# El entrenamiento puede continuar desde el último checkpoint
```

## 📋 Checklist Pre-Entrenamiento

Antes de iniciar el entrenamiento con screen:

- [ ] Dataset presente en `persons/project/ds*/`
- [ ] Espacio suficiente en disco (>20GB recomendado)
- [ ] GPU disponible y funcionando (`nvidia-smi`)
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Screen instalado (`screen --version`)
- [ ] Directorio de logs existe o se creará automáticamente
- [ ] Configuración revisada en `main.py`

## 🚀 Comando Final Recomendado

```bash
# Comando completo para iniciar entrenamiento desatendido
cd ~/computer-vision/app && \
screen -S unet_training -dm bash -c 'python main.py; echo "Finalizado. Presiona Enter."; read' && \
echo "Entrenamiento iniciado en background. Usa 'screen -r unet_training' para conectar."
```

---

**Tip**: Guarda este archivo como referencia rápida. El entrenamiento puede tomar varias horas, así que screen es esencial para mantener el proceso activo.