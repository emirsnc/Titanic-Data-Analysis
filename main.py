import pandas as pd
import kagglehub
import os
import seaborn as sns
import matplotlib.pyplot as plt

# --- BÖLÜM 1: VERİYİ ÇEKME ---
path = kagglehub.dataset_download("yasserh/titanic-dataset")
csv_path = os.path.join(path, "Titanic-Dataset.csv")
df = pd.read_csv(csv_path)

# 1. Önce boş veri var mı diye bakıyoruz
print("Boş ücret sayısı:", df["Fare"].isnull().sum())

print("Boş yaş sayısı:", df["Age"].isnull().sum())


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


sns.set_theme(style="darkgrid")


plt.figure(figsize=(10, 6)) 
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df)

# 3. Etiketleri ekle (Grafiğin ne anlattığı anlaşılsın)
plt.title('Sınıf ve Cinsiyete Göre Hayatta Kalma Oranları', fontsize=15)
plt.ylabel('Hayatta Kalma Oranı (%)')
plt.xlabel('Yolcu Sınıfı')


df_numeric = df.copy() 
df_numeric['Sex'] = df_numeric['Sex'].map({'female': 1, 'male': 0}) #cinsiyetleri sayıyla iliskilendirdik

korelasyon = df_numeric['Sex'].corr(df_numeric['Survived']) 

print(f"Kadın Olmak ve Hayatta Kalma Arasındaki Korelasyon: {korelasyon}")
#0.5 civarı gayet yüksek, yolcu kadınsa kurtulma oranı doğru orantı gosteriyor.