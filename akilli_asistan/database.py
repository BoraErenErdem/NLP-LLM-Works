

import sqlite3 # sqlite veritabanı işlemleri
import os

DB_PATH = os.path.join("data", "assistant_db") # veritabanı dosyasının yolu

# veritabanı'nı başlatan fonksiyon
def initialize_db():
    os.makedirs("data", exist_ok=True) # eğer data klasörü yoksa oluştur

    # veritabanına bağlan ve dosya yoksa oluştur
    conn = sqlite3.connect(DB_PATH) # veritabanı bağlantısı
    cursor = conn.cursor() # sql sorgularını çalıştırmak için gerekli olan araç

    # eğer tablolar yoksa oluştur
    # notes tablosu
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,                    -- otomatik artan birincil anahtar
            content TEXT NOT NULL,                                   -- not icerigi
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP           -- varsayılan olarak şu anki zaman
        )
        """
    )

    # calender tablosu
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS calender (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT NOT NULL,                                     -- etkinlik aciklamasi
            event_date TEXT NOT NULL                                 -- etkinlik tarihi
        )
        """
    )

    conn.commit() # değişiklikleri kaydet
    conn.close() # bağlantıyı kapat


# veritabanına yeni not ekleme
def add_notes(content):
    conn = sqlite3.connect(DB_PATH) # veritabanına bağlan
    cursor = conn.cursor()
    cursor.execute("INSERT INTO notes (content) VALUES (?)", (content,)) # content'i notes tablosuna ekle
    conn.commit()
    conn.close()


# veritabanına yeni etkinlik ekleme
def add_events(event, event_date): # hem event hem de event tarihi eklenir
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO calender (event, event_date) VALUES (?, ?)", (event, event_date)) # event ve event_date'i calender tablosuna ekle
    conn.commit()
    conn.close()


# tüm notları veritabanından sıralı şekilde getiren fonksiyon
def get_notes():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT content, created_at FROM notes ORDER BY created_at DESC") # "notes" tablosundan content ve created_at bilgilerini zaman sırasına göre getir
    notes = cursor.fetchall() # sonuçları liste olarak al
    conn.close()
    return notes


# tüm etkinlikleri veritabanından sıralı şekilde getiren fonksiyon
def get_events():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT event, event_date FROM calender ORDER BY event_date DESC") # "calender" tablosundan event ve event_date bilgilerini zaman sırasına göre getir
    calender = cursor.fetchall()
    conn.close()
    return calender


if __name__ == "__main__":
    initialize_db()
    add_events("motoru bakıma götür", "15.06.2026")
    print(f'Calender: {get_events()}')