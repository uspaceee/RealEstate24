import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Генеруємо штучні дані
np.random.seed(0)  # Для відтворюваності
X = 2 * np.random.rand(100, 1)  # Незалежна змінна (площа в м²)
y = 4 + 3 * X + np.random.randn(100, 1)  # Залежна змінна (ціна)

# Створюємо DataFrame
data = pd.DataFrame(data=np.hstack((X, y)), columns=['Area (m²)', 'Price'])
print(data.head())  # Виводимо перші кілька рядків даних

# Розбиваємо дані на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабування даних
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Створюємо модель лінійної регресії
model = LinearRegression()
model.fit(X_train_scaled, y_train)  # Навчаємо модель

# Прогнозуємо ціну для тестового набору
y_pred = model.predict(X_test_scaled)

# Оцінюємо точність моделі
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'Mean Absolute Error: {mae:.2f}')
print(f'R² Score: {r2:.2f}')

# Виводимо коефіцієнти регресії
print(f'Коефіцієнт (slope): {model.coef_[0][0]:.2f}')
print(f'Перехоплення (intercept): {model.intercept_[0]:.2f}')

# Візуалізуємо результати
plt.figure(figsize=(12, 6))

# Графік тренувальних і тестових даних
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Тренувальні дані')  # Тренувальні значення
plt.scatter(X_test, y_test, color='orange', label='Тестові дані')  # Тестові значення
plt.plot(X_test, y_pred, color='green', label='Лінія регресії')  # Лінія регресії
plt.title('Тренувальні та тестові дані з лінією регресії')
plt.xlabel('Площа (м²)')
plt.ylabel('Ціна')
plt.legend()

# Графік прогнозів
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='blue', label='Реальні значення')  # Реальні значення
plt.scatter(X_test, y_pred, color='red', label='Прогнозовані значення')  # Прогнозовані значення
plt.plot(X_test, y_pred, color='green', label='Лінія регресії')  # Лінія регресії
plt.title('Прогнозування цін на нерухомість')
plt.xlabel('Площа (м²)')
plt.ylabel('Ціна')
plt.legend()

plt.tight_layout()
plt.show()
