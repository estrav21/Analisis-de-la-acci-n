
## Carga de Librerías y datos
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Google_stock_data.csv")

##Revisión de los datos y limpieza de los mismos
print(df.head())

print(df.info())
print("Valores nulos por columna:", df.isnull().sum())

df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

print(df.head())

sns.set_theme(style="darkgrid")

##Procedo a graficar los datos para ver su comportamiento a lo largo del tiempo

plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['Close'], color='blue', linewidth=2)
plt.title('Precio de Cierre de la Acción de Google a lo largo del tiempo', fontsize=16)
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre (USD)')
plt.tight_layout()
plt.show()

##Como gráfica adicional se muestra el rango de precios de las acciones en circulación

df['Range'] = df['High'] - df['Low']

plt.figure(figsize=(14, 4))
plt.plot(df['Date'], df['Range'], color='orange')
plt.title('Rango Diario de Precios (High - Low)', fontsize=16)
plt.xlabel('Fecha')
plt.ylabel('Rango')
plt.tight_layout()
plt.show()

##Se muestran las correlaciones de las variables del precio de apertura, 
## precio de cierre, precio mas alto, precio mas bajo y el volumen de las acciones

plt.figure(figsize=(8, 6))
corr = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación entre Variables')
plt.tight_layout()
plt.show() 

##Grafica de la distribución del precio de cierre del mercado

plt.figure(figsize=(10, 4))
sns.histplot(df['Close'], bins=50, kde=True, color='green')
plt.title('Distribución del Precio de Cierre')
plt.xlabel('Precio de Cierre')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

##Preparación de los datos para el modelo

df['Target'] = df['Close'].shift(-1)

# Se crea una columna 'Target' con el cierre del siguiente día y
# Se elimina la última fila (porque su valor queda vacío)

df = df.dropna().reset_index(drop=True)

##Se usa como variables del día actual para predecir el cierre del día siguiente:

X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
y = df['Target']

##Se dividen los datos para poder entrenar a nuestro modelo

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.4f}")

plt.figure(figsize=(12, 5))
plt.plot(y_test.values, label='Valor real')
plt.plot(y_pred, label='Predicción', alpha=0.7)
plt.title('Comparación: Precio de Cierre Real vs. Predicho')
plt.xlabel('Días')
plt.ylabel('Precio (USD)')
plt.legend()
plt.tight_layout()
plt.show()