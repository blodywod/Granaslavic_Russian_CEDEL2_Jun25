import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pingouin as pg
import seaborn as sns


def procesar_texto(texto):
    """Procesa un texto y devuelve tokens, palabras limpias y estadísticas"""
    tokens = word_tokenize(texto, language='spanish')
    palabras = [word.lower() for word in tokens
                if word not in string.punctuation
                and word.isalpha()]
    return {
        'tokens': tokens,
        'palabras': palabras,
        'num_tokens': len(tokens),
        'num_palabras': len(palabras)
    }


def analizar_distribucion_longitudes(longitudes, nombre_grupo):
    """Analiza y muestra estadísticas descriptivas de longitudes de textos"""
    if not longitudes:
        print("No hay longitudes para analizar")
        return
    print(f"\n{'═' * 50}")
    print(f"ESTADÍSTICOS DE LONGITUD - {nombre_grupo.upper()}")
    print(f"Muestra (n): {len(longitudes)} textos")
    print(f"Media: {pd.Series(longitudes).mean():.1f} palabras")
    print(f"Mediana: {pd.Series(longitudes).median():.1f} palabras")
    print(f"IQR: {pd.Series(longitudes).quantile(0.75) - pd.Series(longitudes).quantile(0.25):.1f}")
    print(f"Rango: {min(longitudes)} - {max(longitudes)} palabras")

    # Test de normalidad
    stat, p = stats.shapiro(longitudes)
    print(f"\nTest de normalidad (Shapiro-Wilk):")
    print(f"Estadístico = {stat:.3f}, p = {p:.4f}")
    print("Interpretación:", "Distribución normal" if p > 0.05 else "Distribución NO normal")


def analizar_textos(textos, nombre_grupo):
    """Analiza una lista de textos y muestra resultados organizados"""
    if not textos:
        print(f"\nNo hay textos para analizar en el grupo {nombre_grupo}")
        return
    stop_words = set(stopwords.words('spanish'))
    frecuencias_todas = Counter()  # Contador para TODAS las palabras
    frecuencias_sin_stop = Counter()  # Contador excluyendo stopwords
    longitudes = []

    print(f"\n{'═'*50}")
    print(f"ANÁLISIS DE TEXTOS - GRUPO {nombre_grupo.upper()}")
    print(f"Total textos: {len(textos)}")
    print("Procesando textos...", end=' ')

    for i, texto in enumerate(textos, 1):
        analisis = procesar_texto(str(texto))
        palabras = analisis['palabras']
        longitudes.append(analisis['num_palabras'])
        frecuencias_todas.update(palabras)
        frecuencias_sin_stop.update([p for p in palabras if p not in stop_words])
    print("Completado ✓")

        # Resumen estadístico conciso
    print(f"\n{'═' * 50}")
    print(f"Texto {i} (Fragmento: '{texto[:50]}...')")
    print(f"Tokens: {analisis['num_tokens']} | Palabras: {analisis['num_palabras']}")

    analizar_distribucion_longitudes(longitudes, nombre_grupo)

    return longitudes


def comparar_grupos(longitudes_ruso, longitudes_no_ruso):
    """Compara estadísticamente dos grupos de longitudes"""
    print(f"\n{'═' * 50}")
    print("COMPARACIÓN ESTADÍSTICA ENTRE GRUPOS")

    # Test de Mann-Whitney (no paramétrico) and Size effect r
    stat, p = stats.mannwhitneyu(longitudes_ruso, longitudes_no_ruso, alternative='two-sided')
    res = pg.mwu(longitudes_ruso, longitudes_no_ruso)
    # CLES manual
    n1 = len(longitudes_ruso)
    n2 = len(longitudes_no_ruso)
    cles = stat / (n1 * n2)

    print(f"\nTest de Mann-Whitney U:")
    print(f"U = {stat:.1f}, p = {p:.4f}")
    print("Interpretación:", "Diferencias significativas" if p < 0.05 else "No hay diferencias significativas")
    print(f"r={res['RBC'][0]:.3f}") # Tamaño del efecto directamente, Rank Biserial Correlation
    print(f"CLES = {cles:.3f}  (Common Language Effect Size)")

    # Prepare data in long format
    data = pd.DataFrame({
        'Text length': longitudes_ruso + longitudes_no_ruso,
        'Group': ['Russian L1 learners'] * len(longitudes_ruso) + ['non-Russian L1 learners'] * len(longitudes_no_ruso)
    })

    # Plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x='Group', y='Text length', data=data,
        showmeans=True,
        meanprops=dict(marker='o', markerfacecolor='black', markeredgecolor='black'),
        boxprops=dict(facecolor='white', edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='black')
    )

    # Add stripplot
    sns.stripplot(
        x='Group', y='Text length', data=data,
        color='gray', alpha=0.3, jitter=0.2, size=3
    )

    plt.title('Text length differences across L1 groups', fontsize=14)
    plt.ylabel('Word count')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    TARGET_FILE_PATH = 'C:\\Users\\ASUS\\Documents\\PYTHON_CODE\\Congreso_Granaslavic_June25\\sources\\texts.csv'  # os.path.join('..', 'sources', 'ruso_data.csv')

    with open(TARGET_FILE_PATH, 'r', encoding='utf8') as f: # Leer todas las líneas una sola vez
        lineas = f.read().split('\n')[1:-1]  # Ignora encabezado y línea vacía final

    # Filtrar textos rusos (cualquier tarea)
        textos_rusos = [
            row.split('\t')[-2]  # Columna de texto (penúltima)
            for row in lineas
            if row.split('\t')[11] == 'Russian'
        ]

    # Filtrar: NO rusos + tarea 14 (cualquier formato)
    df = pd.read_csv(TARGET_FILE_PATH, sep='\t', encoding='utf8')
    textos_no_rusos_chap14 = df[
        (df['L1'] != 'Russian') &
        (df['Task number'].astype(str).str.contains('14'))  # Captura '14' o '14-Chaplin'
        ]['Text'].dropna().tolist()  # Elimina valores nulos

    longitudes_ruso = analizar_textos(textos_rusos, "Ruso")
    longitudes_no_ruso = analizar_textos(textos_no_rusos_chap14, "No-Ruso-Chap14")

    if longitudes_ruso and longitudes_no_ruso:
        comparar_grupos(longitudes_ruso, longitudes_no_ruso)
