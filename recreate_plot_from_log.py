#!/usr/bin/env python3
"""
Script para recrear gráficos de entrenamiento desde archivos de log.

Uso:
    python recreate_plot_from_log.py logs/segmentation_20251022_101413.log
    python recreate_plot_from_log.py  # Usa el log más reciente

Autor: Claude Code
Proyecto: Computer Vision - PUCP
"""

import re
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def parse_training_log(log_path: str) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], List[int]]:
    """
    Parsea un archivo de log de entrenamiento y extrae las métricas.

    Args:
        log_path: Ruta al archivo de log

    Returns:
        Tupla de (train_history, val_history, best_model_epochs)
    """
    train_history = {
        'loss': [],
        'iou': [],
        'dice': [],
        'f1': [],
        'precision': [],
        'recall': []
    }

    val_history = {
        'loss': [],
        'iou': [],
        'dice': [],
        'f1': [],
        'precision': [],
        'recall': []
    }

    best_model_epochs = []

    # Patrones regex para extraer métricas
    train_pattern = re.compile(
        r'Epoch (\d+) - Train Loss: ([\d.]+), IoU: ([\d.]+), Dice: ([\d.]+), '
        r'F1: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+)'
    )

    val_pattern = re.compile(
        r'Epoch (\d+) - Val Loss: ([\d.]+), IoU: ([\d.]+), Dice: ([\d.]+), '
        r'F1: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+)'
    )

    best_model_pattern = re.compile(
        r'✅ Mejor modelo guardado.*IoU: ([\d.]+)'
    )

    print(f"📖 Leyendo log: {log_path}")

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_epoch = 0

    for i, line in enumerate(lines):
        # Extraer métricas de entrenamiento
        train_match = train_pattern.search(line)
        if train_match:
            epoch = int(train_match.group(1))
            current_epoch = epoch
            train_history['loss'].append(float(train_match.group(2)))
            train_history['iou'].append(float(train_match.group(3)))
            train_history['dice'].append(float(train_match.group(4)))
            train_history['f1'].append(float(train_match.group(5)))
            train_history['precision'].append(float(train_match.group(6)))
            train_history['recall'].append(float(train_match.group(7)))
            continue

        # Extraer métricas de validación
        val_match = val_pattern.search(line)
        if val_match:
            epoch = int(val_match.group(1))
            val_history['loss'].append(float(val_match.group(2)))
            val_history['iou'].append(float(val_match.group(3)))
            val_history['dice'].append(float(val_match.group(4)))
            val_history['f1'].append(float(val_match.group(5)))
            val_history['precision'].append(float(val_match.group(6)))
            val_history['recall'].append(float(val_match.group(7)))
            continue

        # Detectar cuando se guarda el mejor modelo
        best_match = best_model_pattern.search(line)
        if best_match and current_epoch > 0:
            if current_epoch not in best_model_epochs:
                best_model_epochs.append(current_epoch)

    # Validar que tenemos datos
    if len(train_history['loss']) == 0:
        raise ValueError("No se encontraron datos de entrenamiento en el log")

    print(f"✅ Parseado completo:")
    print(f"   • Epochs procesados: {len(train_history['loss'])}")
    print(f"   • Mejores modelos guardados: {len(best_model_epochs)} veces")
    print(f"   • Epochs con mejor modelo: {best_model_epochs}")

    return train_history, val_history, best_model_epochs


def plot_training_history(
    train_history: Dict[str, List[float]],
    val_history: Dict[str, List[float]],
    best_epochs: List[int],
    output_path: str = 'plots/segmentation_training_history.png'
):
    """
    Genera gráficos de entrenamiento (formato estándar del trainer).

    Args:
        train_history: Diccionario con métricas de entrenamiento
        val_history: Diccionario con métricas de validación
        best_epochs: Lista de epochs donde se guardó el mejor modelo
        output_path: Ruta donde guardar el gráfico
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Crear figura con 3 subplots (Loss, IoU, Dice)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    epochs = list(range(1, len(train_history['loss']) + 1))

    # Subplot 1: Loss
    axes[0].plot(epochs, train_history['loss'], label='Train Loss', linewidth=2, color='#3498db')
    axes[0].plot(epochs, val_history['loss'], label='Val Loss', linewidth=2, color='#e74c3c')
    axes[0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Marcar mejores epochs
    for epoch in best_epochs:
        if epoch <= len(epochs):
            axes[0].axvline(x=epoch, color='green', linestyle='--', alpha=0.3, linewidth=1)

    # Subplot 2: IoU
    axes[1].plot(epochs, train_history['iou'], label='Train IoU', linewidth=2, color='#3498db')
    axes[1].plot(epochs, val_history['iou'], label='Val IoU', linewidth=2, color='#e74c3c')
    axes[1].set_title('Training and Validation IoU', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('IoU', fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Marcar mejores epochs
    for epoch in best_epochs:
        if epoch <= len(epochs):
            axes[1].axvline(x=epoch, color='green', linestyle='--', alpha=0.3, linewidth=1)

    # Agregar anotación del mejor IoU
    best_val_iou = max(val_history['iou'])
    best_val_epoch = val_history['iou'].index(best_val_iou) + 1
    axes[1].scatter([best_val_epoch], [best_val_iou], color='green', s=100, zorder=5, marker='*')
    axes[1].annotate(f'Best: {best_val_iou:.4f}',
                     xy=(best_val_epoch, best_val_iou),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=9, color='green', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.7))

    # Subplot 3: Dice
    axes[2].plot(epochs, train_history['dice'], label='Train Dice', linewidth=2, color='#3498db')
    axes[2].plot(epochs, val_history['dice'], label='Val Dice', linewidth=2, color='#e74c3c')
    axes[2].set_title('Training and Validation Dice', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=10)
    axes[2].set_ylabel('Dice', fontsize=10)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    # Marcar mejores epochs
    for epoch in best_epochs:
        if epoch <= len(epochs):
            axes[2].axvline(x=epoch, color='green', linestyle='--', alpha=0.3, linewidth=1)

    # Agregar anotación del mejor Dice
    best_val_dice = max(val_history['dice'])
    best_val_dice_epoch = val_history['dice'].index(best_val_dice) + 1
    axes[2].scatter([best_val_dice_epoch], [best_val_dice], color='green', s=100, zorder=5, marker='*')
    axes[2].annotate(f'Best: {best_val_dice:.4f}',
                     xy=(best_val_dice_epoch, best_val_dice),
                     xytext=(10, 10), textcoords='offset points',
                     fontsize=9, color='green', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Gráfico guardado: {output_path}")


def plot_extended_metrics(
    train_history: Dict[str, List[float]],
    val_history: Dict[str, List[float]],
    best_epochs: List[int],
    output_path: str = 'plots/segmentation_extended_metrics.png'
):
    """
    Genera gráficos adicionales con F1, Precision, Recall.

    Args:
        train_history: Diccionario con métricas de entrenamiento
        val_history: Diccionario con métricas de validación
        best_epochs: Lista de epochs donde se guardó el mejor modelo
        output_path: Ruta donde guardar el gráfico
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Crear figura con 3 subplots (F1, Precision, Recall)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    epochs = list(range(1, len(train_history['loss']) + 1))

    # Subplot 1: F1 Score
    axes[0].plot(epochs, train_history['f1'], label='Train F1', linewidth=2, color='#3498db')
    axes[0].plot(epochs, val_history['f1'], label='Val F1', linewidth=2, color='#e74c3c')
    axes[0].set_title('Training and Validation F1 Score', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('F1 Score', fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    for epoch in best_epochs:
        if epoch <= len(epochs):
            axes[0].axvline(x=epoch, color='green', linestyle='--', alpha=0.3, linewidth=1)

    # Subplot 2: Precision
    axes[1].plot(epochs, train_history['precision'], label='Train Precision', linewidth=2, color='#3498db')
    axes[1].plot(epochs, val_history['precision'], label='Val Precision', linewidth=2, color='#e74c3c')
    axes[1].set_title('Training and Validation Precision', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('Precision', fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    for epoch in best_epochs:
        if epoch <= len(epochs):
            axes[1].axvline(x=epoch, color='green', linestyle='--', alpha=0.3, linewidth=1)

    # Subplot 3: Recall
    axes[2].plot(epochs, train_history['recall'], label='Train Recall', linewidth=2, color='#3498db')
    axes[2].plot(epochs, val_history['recall'], label='Val Recall', linewidth=2, color='#e74c3c')
    axes[2].set_title('Training and Validation Recall', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=10)
    axes[2].set_ylabel('Recall', fontsize=10)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    for epoch in best_epochs:
        if epoch <= len(epochs):
            axes[2].axvline(x=epoch, color='green', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Gráfico extendido guardado: {output_path}")


def print_summary(train_history: Dict[str, List[float]], val_history: Dict[str, List[float]]):
    """Imprime un resumen de las métricas finales."""
    print("\n" + "="*60)
    print("📊 RESUMEN DE MÉTRICAS FINALES")
    print("="*60)

    final_epoch = len(train_history['loss'])

    print(f"\n🏋️  TRAIN (Epoch {final_epoch}):")
    print(f"   • Loss:      {train_history['loss'][-1]:.4f}")
    print(f"   • IoU:       {train_history['iou'][-1]:.4f}")
    print(f"   • Dice:      {train_history['dice'][-1]:.4f}")
    print(f"   • F1:        {train_history['f1'][-1]:.4f}")
    print(f"   • Precision: {train_history['precision'][-1]:.4f}")
    print(f"   • Recall:    {train_history['recall'][-1]:.4f}")

    print(f"\n✅ VALIDATION (Epoch {final_epoch}):")
    print(f"   • Loss:      {val_history['loss'][-1]:.4f}")
    print(f"   • IoU:       {val_history['iou'][-1]:.4f}")
    print(f"   • Dice:      {val_history['dice'][-1]:.4f}")
    print(f"   • F1:        {val_history['f1'][-1]:.4f}")
    print(f"   • Precision: {val_history['precision'][-1]:.4f}")
    print(f"   • Recall:    {val_history['recall'][-1]:.4f}")

    print(f"\n🏆 MEJORES MÉTRICAS DE VALIDACIÓN:")
    print(f"   • Mejor Loss:      {min(val_history['loss']):.4f} (Epoch {val_history['loss'].index(min(val_history['loss'])) + 1})")
    print(f"   • Mejor IoU:       {max(val_history['iou']):.4f} (Epoch {val_history['iou'].index(max(val_history['iou'])) + 1})")
    print(f"   • Mejor Dice:      {max(val_history['dice']):.4f} (Epoch {val_history['dice'].index(max(val_history['dice'])) + 1})")
    print(f"   • Mejor F1:        {max(val_history['f1']):.4f} (Epoch {val_history['f1'].index(max(val_history['f1'])) + 1})")
    print(f"   • Mejor Precision: {max(val_history['precision']):.4f} (Epoch {val_history['precision'].index(max(val_history['precision'])) + 1})")
    print(f"   • Mejor Recall:    {max(val_history['recall']):.4f} (Epoch {val_history['recall'].index(max(val_history['recall'])) + 1})")

    print("\n" + "="*60)


def find_latest_log(logs_dir: str = 'logs') -> str:
    """Encuentra el log de segmentación más reciente."""
    log_files = list(Path(logs_dir).glob('segmentation_*.log'))

    if not log_files:
        raise FileNotFoundError(f"No se encontraron logs de segmentación en {logs_dir}/")

    # Ordenar por fecha de modificación (más reciente primero)
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    return str(latest_log)


def main():
    """Función principal."""
    print("="*60)
    print("🎨 RECREAR GRÁFICOS DE ENTRENAMIENTO DESDE LOG")
    print("="*60)

    # Determinar qué log usar
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        try:
            log_path = find_latest_log()
            print(f"\n💡 Usando log más reciente: {log_path}")
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            print("\nUso: python recreate_plot_from_log.py [ruta_al_log]")
            sys.exit(1)

    # Verificar que el archivo existe
    if not os.path.exists(log_path):
        print(f"\n❌ Error: No se encontró el archivo {log_path}")
        sys.exit(1)

    try:
        # Parsear log
        train_history, val_history, best_epochs = parse_training_log(log_path)

        # Generar gráfico principal (Loss, IoU, Dice)
        print("\n🎨 Generando gráfico principal...")
        plot_training_history(train_history, val_history, best_epochs)

        # Generar gráfico extendido (F1, Precision, Recall)
        print("\n🎨 Generando gráfico extendido...")
        plot_extended_metrics(train_history, val_history, best_epochs)

        # Mostrar resumen
        print_summary(train_history, val_history)

        print("\n✅ ¡Gráficos recreados exitosamente!")
        print("\n📁 Archivos generados:")
        print("   • plots/segmentation_training_history.png")
        print("   • plots/segmentation_extended_metrics.png")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
