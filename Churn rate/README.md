# Прогнозирование оттока клиентов оператора связи
### Цель
Создать надежную и точную модель, способную предсказывать вероятность оттока клиента на основе его персональных данных, информации о тарифах, услугах и договоре. Это позволит оператору своевременно принимать меры по удержанию клиентов, предлагая им промокоды, специальные условия и индивидуальные предложения.
### Инструменты
* pandas
* sklearn
* lightgbm
* matplotlib, seaborn
* shap
### Ход работы
* Изучил данные, определили типы и пропуски в признаках.
* Провел обработку данных, в том числе преобразовал даты, заполнил пропуски и создал целевой признак для предсказания оттока.
* Собрал и разделил данные для моделей
* Обучил три модели с подбором гиперпараметров: SGDClassifier, DecisionTreeClassifier и LGBM Classifier.
* Проанализировал важность признаков.
* Проверил качество модели на тестовой выборке.
### Результат
Модель LGBMClassifier с оптимизированными гиперпараметрами показала наилучшую производительность с ROC-AUC 0.917 на тестовой выборке.
