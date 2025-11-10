#!/usr/bin/env python3
"""
modelo_regresion_ventas.py
Modelo de regresión para predecir monto total de ventas
Materia: Extracción de conocimientos de bases de datos
Equipo: Josué Alejandro Arenas Hernández, José Ángel Palomo Reyna, Karol Lizbeth Martínez Hernández
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def cargar_y_preparar_datos():
    """Carga y prepara los datos para el modelo"""
    print("=== CARGANDO Y PREPARANDO DATOS ===")
    
    # Cargar dataset limpio
    df = pd.read_csv('ventas_tienda_limpio.csv')
    print(f"Dataset original: {df.shape[0]} registros, {df.shape[1]} columnas")
    
    # Crear características temporales
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['mes'] = df['sale_date'].dt.month
    df['dia_semana'] = df['sale_date'].dt.dayofweek
    df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)
    
    # Codificar variables categóricas
    le_city = LabelEncoder()
    le_category = LabelEncoder()
    
    df['customer_city_encoded'] = le_city.fit_transform(df['customer_city'])
    df['category_encoded'] = le_category.fit_transform(df['category'])
    
    print("✓ Características temporales creadas")
    print("✓ Variables categóricas codificadas")
    
    return df, le_city, le_category

def exploracion_datos(df):
    """Análisis exploratorio de los datos"""
    print("\n=== ANÁLISIS EXPLORATORIO ===")
    
    # Estadísticas básicas
    print("Estadísticas de variable objetivo (total_amount):")
    print(df['total_amount'].describe())
    
    # Correlaciones
    numeric_cols = ['unit_price', 'quantity', 'total_amount', 'mes', 'dia_semana']
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlación - Variables Numéricas')
    plt.tight_layout()
    plt.savefig('correlacion_variables.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Matriz de correlación guardada")
    
    # Distribución de la variable objetivo
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['total_amount'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Monto Total')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Monto Total de Ventas')
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['unit_price'], df['total_amount'], alpha=0.6)
    plt.xlabel('Precio Unitario')
    plt.ylabel('Monto Total')
    plt.title('Relación Precio Unitario vs Monto Total')
    
    plt.tight_layout()
    plt.savefig('distribucion_ventas.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return corr_matrix

def preparar_variables_modelo(df):
    """Prepara las variables para el modelo de machine learning"""
    print("\n=== PREPARACIÓN DE VARIABLES PARA MODELO ===")
    
    # Selección de características
    features = [
        'product_id', 'unit_price', 'quantity', 'store_id',
        'customer_city_encoded', 'category_encoded', 
        'mes', 'dia_semana', 'es_fin_semana'
    ]
    
    X = df[features]
    y = df['total_amount']
    
    print(f"Variables predictoras: {len(features)}")
    print(f"Variable objetivo: total_amount")
    print(f"Forma de X: {X.shape}, Forma de y: {y.shape}")
    
    # División train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=True
    )
    
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} registros")
    print(f"Conjunto de prueba: {X_test.shape[0]} registros")
    
    return X_train, X_test, y_train, y_test, features

def entrenar_modelos(X_train, X_test, y_train, y_test, features):
    """Entrena y evalúa múltiples modelos de regresión"""
    print("\n=== ENTRENAMIENTO Y EVALUACIÓN DE MODELOS ===")
    
    # Inicializar modelos
    models = {
        'Regresión Lineal': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100)
    }
    
    resultados = {}
    
    for name, model in models.items():
        print(f"\n--- Entrenando {name} ---")
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Predecir
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Validación cruzada
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        resultados[name] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"R² Validación Cruzada: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    return resultados

def optimizar_mejor_modelo(X_train, y_train):
    """Optimiza el mejor modelo usando GridSearch"""
    print("\n=== OPTIMIZACIÓN DEL MEJOR MODELO ===")
    
    # Parámetros para Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='r2', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("✓ Mejores parámetros encontrados:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"✓ Mejor score (R²): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def visualizar_resultados(resultados, best_model, X_test, y_test, features):
    """Genera visualizaciones de los resultados"""
    print("\n=== GENERANDO VISUALIZACIONES ===")
    
    # 1. Comparación de modelos
    modelos = list(resultados.keys())
    r2_scores = [resultados[model]['r2'] for model in modelos]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(modelos, r2_scores, color=['lightblue', 'lightcoral'])
    plt.ylabel('R² Score')
    plt.title('Comparación de Rendimiento de Modelos (R²)')
    plt.ylim(0, 1)
    
    # Añadir valores en las barras
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('comparacion_modelos.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Predicciones vs Valores reales (mejor modelo)
    y_pred_best = best_model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_best, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs Valores Reales - Random Forest Optimizado')
    
    # Añadir métricas al gráfico
    r2_best = r2_score(y_test, y_pred_best)
    plt.text(0.05, 0.95, f'R² = {r2_best:.4f}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('predicciones_vs_reales.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Importancia de características
    if hasattr(best_model, 'feature_importances_'):
        importancia = best_model.feature_importances_
        indices = np.argsort(importancia)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.title('Importancia de Características - Random Forest')
        plt.barh(range(len(indices)), importancia[indices], color='lightseagreen')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Importancia Relativa')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('importancia_caracteristicas.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✓ Gráficas de resultados guardadas")

def generar_reporte_final(resultados, best_model):
    """Genera un reporte final con los resultados"""
    print("\n=== REPORTE FINAL ===")
    
    # Tabla comparativa de modelos
    print("\n" + "="*60)
    print("COMPARACIÓN DE MODELOS - MÉTRICAS DE EVALUACIÓN")
    print("="*60)
    print(f"{'Modelo':<20} {'MAE':<10} {'RMSE':<12} {'R²':<10} {'R² CV':<12}")
    print("-"*60)
    
    for name, metrics in resultados.items():
        print(f"{name:<20} {metrics['mae']:<10.2f} {metrics['rmse']:<12.2f} "
              f"{metrics['r2']:<10.4f} {metrics['cv_mean']:<12.4f}")
    
    # Identificar mejor modelo
    mejor_modelo = max(resultados.keys(), key=lambda x: resultados[x]['r2'])
    print(f"\n✓ MEJOR MODELO: {mejor_modelo}")
    print(f"✓ R² del mejor modelo: {resultados[mejor_modelo]['r2']:.4f}")
    
    # Interpretación de resultados
    print("\n" + "="*60)
    print("INTERPRETACIÓN DE RESULTADOS")
    print("="*60)
    
    r2_mejor = resultados[mejor_modelo]['r2']
    if r2_mejor >= 0.8:
        interpretacion = "Excelente poder predictivo"
    elif r2_mejor >= 0.6:
        interpretacion = "Buen poder predictivo"
    elif r2_mejor >= 0.4:
        interpretacion = "Poder predictivo moderado"
    else:
        interpretacion = "Poder predictivo limitado"
    
    print(f"• Poder predictivo del modelo: {interpretacion}")
    print(f"• El modelo explica aproximadamente {r2_mejor*100:.1f}% de la variabilidad en el monto total de ventas")
    print("• Las características más importantes influyen significativamente en las predicciones")
    
    return mejor_modelo

def main():
    """Función principal"""
    print("INICIANDO ANÁLISIS SUPERVISADO - PREDICCIÓN DE VENTAS")
    print("="*60)
    
    try:
        # 1. Cargar y preparar datos
        df, le_city, le_category = cargar_y_preparar_datos()
        
        # 2. Análisis exploratorio
        corr_matrix = exploracion_datos(df)
        
        # 3. Preparar variables para el modelo
        X_train, X_test, y_train, y_test, features = preparar_variables_modelo(df)
        
        # 4. Entrenar y evaluar modelos
        resultados = entrenar_modelos(X_train, X_test, y_train, y_test, features)
        
        # 5. Optimizar el mejor modelo
        best_model = optimizar_mejor_modelo(X_train, y_train)
        
        # Evaluar modelo optimizado
        y_pred_best = best_model.predict(X_test)
        r2_best = r2_score(y_test, y_pred_best)
        resultados['Random Forest Optimizado'] = {
            'model': best_model,
            'mae': mean_absolute_error(y_test, y_pred_best),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_best)),
            'r2': r2_best,
            'cv_mean': r2_best,
            'cv_std': 0.0
        }
        
        # 6. Visualizar resultados
        visualizar_resultados(resultados, best_model, X_test, y_test, features)
        
        # 7. Generar reporte final
        mejor_modelo = generar_reporte_final(resultados, best_model)
        
        print("\n" + "="*60)
        print("ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("✓ Archivos generados:")
        print("  - correlacion_variables.png")
        print("  - distribucion_ventas.png")
        print("  - comparacion_modelos.png")
        print("  - predicciones_vs_reales.png")
        print("  - importancia_caracteristicas.png")
        print(f"✓ Mejor modelo: {mejor_modelo}")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {str(e)}")
        raise

if __name__ == "__main__":
    main()