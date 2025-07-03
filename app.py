import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox
from PIL import Image, ImageTk, UnidentifiedImageError
import torch
from torchvision import transforms
import timm
import torch.nn as nn
import os
import matplotlib.pyplot as plt


# 📁 Dosya kontrolü
if not os.path.exists("best_model.pth") or not os.path.exists("labels.txt"):
    raise FileNotFoundError("best_model.pth veya labels.txt eksik. Lütfen dosyaları kontrol edin.")

# 📥 Model yükleme
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=90)
model.head = nn.Linear(model.head.in_features, 90)
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()

# 📂 Etiketleri yükle
with open("labels.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 🔄 Görsel dönüştürme
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 🔍 Tahmin fonksiyonu
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)
        return classes[predicted_class.item()], confidence.item()
    except Exception as e:
        messagebox.showerror("Hata", f"Resim tahmin edilirken bir hata oluştu:\n{str(e)}")
        return None, None



# Görsel Özelliklerini Hesaplayan Fonksiyon
def extract_image_features(image: Image.Image):
    try:
        width, height = image.size
        # Renk ortalamalarını hesapla
        mean_r, mean_g, mean_b = image.resize((50, 50)).convert("RGB").split()
        r_avg = sum(mean_r.getdata()) / len(mean_r.getdata())
        g_avg = sum(mean_g.getdata()) / len(mean_g.getdata())
        b_avg = sum(mean_b.getdata()) / len(mean_b.getdata())

        # Histogram bilgisi (basit çıktı)
        histogram = image.histogram()
        total_pixels = sum(histogram)

        return f"📐 Boyut: {width}x{height} px\n🎨 Ortalama Renk (RGB):\n R: {r_avg:.1f}, G: {g_avg:.1f}, B: {b_avg:.1f}\n🧮 Piksel Sayısı: {total_pixels}"
    except Exception as e:
        return f"Hata: {str(e)}"

# select_image fonksiyonu içinde güncellenmiş hali
def select_image():
    file_path = filedialog.askopenfilename(
        title="Görsel Seç",
        filetypes=[("Görüntü Dosyaları", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

    try:
        image = Image.open(file_path)
        image.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(image)
        panel.config(image=img_tk)
        panel.image = img_tk

        # Tahmin
        label, conf = predict_image(file_path)
        if label:
            result_label.config(
                text=f"🐾 Tahmin: {label.upper()}\n🎯 Güven: %{conf*100:.2f}",
                fg="green"
            )

        # Görsel Özelliklerini Göster
        info = extract_image_features(image)
        info_label.config(text=info)

        # Histogramı Göster (popup)
        show_histogram(image)

    except UnidentifiedImageError:
        messagebox.showerror("Hatalı Dosya", "Görsel açılırken hata oluştu. Lütfen geçerli bir görsel seçin.")
    except Exception as e:
        messagebox.showerror("Hata", f"Bir hata oluştu:\n{str(e)}")

def show_histogram(image: Image.Image):
    try:
        # RGB kanallarına ayır
        r, g, b = image.convert("RGB").split()

        # Histogramları al
        r_hist = r.histogram()
        g_hist = g.histogram()
        b_hist = b.histogram()

        # Matplotlib ile çiz
        plt.figure("📊 Görsel Histogramı")
        plt.title("Renk Kanalları Histogramı")
        plt.xlabel("Piksel Değeri")
        plt.ylabel("Frekans")
        plt.plot(r_hist, color='red', label='Kırmızı')
        plt.plot(g_hist, color='green', label='Yeşil')
        plt.plot(b_hist, color='blue', label='Mavi')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Histogram Hatası", f"Histogram çizilirken hata oluştu:\n{str(e)}")
        
def select_folder():
    folder_path = filedialog.askdirectory(title="Bir Klasör Seç")
    if not folder_path:
        return

    results = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            full_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(full_path)
                label, conf = predict_image(full_path)

                if label is None:
                   raise ValueError("Model tahmin üretemedi.")

                features = extract_image_features(image)
                results.append({
                   "Dosya": filename,
                   "Tahmin": label,
                   "Güven": f"{conf*100:.2f}%",
                   "Özellikler": features.replace("\n", " | ")
                })

            except Exception as e:
                results.append({
                   "Dosya": filename,
                   "Tahmin": "HATA",
                   "Güven": "N/A",
                   "Özellikler": f"Hata: {str(e)}"
               })
            
    

    # Raporu kaydet
    with open("klasor_sonuclari.txt", "w", encoding="utf-8") as f:
        f.write("📄 MultiZoo Klasör Analiz Raporu\n\n")
        for r in results:
            f.write(f"📁 Dosya: {r['Dosya']}\n")
            f.write(f"🐾 Tahmin: {r['Tahmin']} | 🎯 Güven: {r['Güven']}\n")
            f.write(f"📊 Özellikler: {r['Özellikler']}\n")
            f.write("-" * 60 + "\n")

    messagebox.showinfo("İşlem Tamamlandı", "Klasördeki tüm görseller analiz edildi ve 'klasor_sonuclari.txt' dosyasına yazıldı.")


# 🎨 Arayüz başlat
root = tk.Tk()
root.title("🐾 MultiZoo Hayvan Tanıma Arayüzü")
root.geometry("400x500")
root.configure(bg="#f9f9f9")

# 📷 Görsel seçme
# Görsel Özelliklerini Gösteren Etiket
info_label = Label(root, text="", font=("Arial", 10), bg="#f9f9f9", fg="#333", justify="left")
info_label.pack(pady=5)

# Başlık
title_label = Label(root, text="MultiZoo Görsel Tanıma", font=("Helvetica", 16, "bold"), bg="#f9f9f9")
title_label.pack(pady=10)

# Açıklama
desc_label = Label(
    root,
    text="Bir hayvan görseli seçin ve modelin tahminini görün.",
    font=("Arial", 10),
    bg="#f9f9f9",
    fg="#555"
)
desc_label.pack(pady=5)

# Buton
btn = Button(root, text="📁 Görsel Seç", command=select_image, font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
btn.pack(pady=10)

btn_folder = Button(root, text="📂 Klasör Tara", command=select_folder, font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5)
btn_folder.pack(pady=5)


# Görsel Paneli
panel = Label(root, bg="#ddd", width=300, height=300)
panel.pack(pady=10)

# Sonuç Etiketi
result_label = Label(root, text="", font=("Arial", 12), bg="#f9f9f9", fg="black", justify="center")
result_label.pack(pady=10)

# Başlat
root.mainloop()
