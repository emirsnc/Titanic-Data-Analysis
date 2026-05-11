import pandas as pd
import kagglehub
import os

# --- BÖLÜM 1: VERİYİ ÇEKME ---
path = kagglehub.dataset_download("yasserh/titanic-dataset")
csv_path = os.path.join(path, "Titanic-Dataset.csv")
df = pd.read_csv(csv_path)

# 1. Önce boş veri var mı diye bakıyoruz
print("Boş ücret sayısı:", df["Fare"].isnull().sum())

print("Boş yaş sayısı:", df["Age"].isnull().sum())

# 2. Eğer boş veri varsa (Titanic'te genelde 1 tane olur), 
# senin görseldeki x_mean mantığıyla dolduralım:
fare_mean = df["Fare"].mean()
df["Fare"] = df["Fare"].fillna(fare_mean)

age_mean = df["Age"].mean()
df["Age"] = df["Age"].fillna(age_mean)

print("İşlem sonrası boş veri sayısı:", df["Fare"].isnull().sum())
print("İşlem sonrası boş yaş sayısı:", df["Age"].isnull().sum())

# Hem hayatta kalma hem de cinsiyete göre ücret ortalamaları
print("\n--- Cinsiyet ve Hayatta Kalma Bazlı Ücret Analizi ---")
analiz = df.groupby('Pclass')['Survived'].mean()
print(analiz)


print("Cinsiyete göre")
analiz2 = df.groupby(['Pclass', 'Sex'])['Survived'].mean()
print(analiz2)