

# 🐾 MultiZoo - Animal Species Classification System / Hayvan Türü Sınıflandırma Sistemi

MultiZoo is a deep learning-based desktop application that classifies animal species from images using a Vision Transformer (ViT) model. It includes a simple GUI built with Tkinter and was trained on a 90-class dataset of animal images.

MultiZoo, Vision Transformer (ViT) mimarisiyle hayvan türlerini sınıflandıran masaüstü tabanlı bir derin öğrenme uygulamasıdır. Tkinter ile hazırlanmış arayüzü sayesinde kullanıcı, görsel seçerek anında tahmin sonucunu görebilir. Model, 90 farklı hayvan türünü içeren özel bir veri seti ile eğitilmiştir.

---

## 📌 Overview / Genel Bilgiler

- 🔬 **Model**: Vision Transformer (ViT Base Patch16-224)
- 🐾 **Classes / Sınıf Sayısı**: 90 animal species / 90 hayvan türü
- 🖼️ **Dataset**: 4,770 labeled images / 4.770 etiketli görsel
- 💻 **Interface / Arayüz**: Python Tkinter
- 📈 **Accuracy / Doğruluk**: 90.1%
- ⚙️ **Frameworks / Kullanılan Kütüphaneler**: PyTorch, timm, torchvision, PIL

---

## 🧠 Model Training / Model Eğitimi

| Parameter / Parametre    | Value / Değer              |
|--------------------------|----------------------------|
| Model                    | ViT (timm)                 |
| Input Size / Girdi Boyutu| 224 × 224                  |
| Optimizer / Optimizasyon| AdamW                      |
| Learning Rate / Öğrenme Oranı | 0.0001 (with scheduler) |
| Scheduler / Öğrenme Planlayıcı | ReduceLROnPlateau   |
| Dropout                  | 0.1                        |
| Epochs / Epok Sayısı     | 50 (Early stopping at ~10) |
| Batch Size               | 32                         |

Training Techniques / Eğitim Teknikleri:
- Data Augmentation: RandomFlip, Rotation, ColorJitter
- Normalization: RGB mean=0.5, std=0.5

---

## 🧪 Dataset Structure / Veri Seti Yapısı

The dataset is split as 80% training and 20% validation. Test images are held out completely.

Veri seti %80 eğitim ve %20 doğrulama olarak ayrılmıştır. Test verileri eğitim sürecine dahil edilmemiştir.


