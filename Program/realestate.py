# Імпортуємо необхідні бібліотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Генеруємо штучні дані
np.random.seed(0)  # Для відтворюваності
X = 2 * np.random.rand(100, 1)  # Незалежна змінна (площа в м²)
y = 4 + 3 * X + np.random.randn(100, 1)  # Залежна змінна (ціна)

# Створюємо DataFrame
data = pd.DataFrame(data=np.hstack((X, y)), columns=['Area (m²)', 'Price'])
print(data.head())  # Виводимо перші кілька рядків даних

# Розбиваємо дані на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створюємо модель лінійної регресії
model = LinearRegression()
model.fit(X_train, y_train)  # Навчаємо модель

# Прогнозуємо ціну для тестового набору
y_pred = model.predict(X_test)

# Оцінюємо точність моделі
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')

# Візуалізуємо результати
plt.scatter(X_test, y_test, color='blue', label='Дані тестування')  # Реальні значення
plt.scatter(X_test, y_pred, color='red', label='Прогнозовані значення')  # Прогнозовані значення
plt.plot(X_test, y_pred, color='green', label='Лінія регресії')  # Лінія регресії
plt.title('Лінійна регресія для прогнозування цін на нерухомість')
plt.xlabel('Площа (м²)')
plt.ylabel('Ціна')
plt.legend()
plt.show()

