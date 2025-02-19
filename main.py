# Импортируем необходимые библиотеки
import numpy as np
import cvzone
import cv2
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from google import genai
from PIL import Image

# Настройка страницы Streamlit
st.set_page_config(layout='wide')
st.image('your_image.png')  # Отображаем изображение на странице

# Создаем два столбца для интерфейса
col1, col2 = st.columns([2, 1])
with col1:
    run = st.checkbox('Run', value=True)  # Чекбокс для запуска
    FRAME_WIND = st.image([])  # Пустое изображение для отображения видео

with col2:
    st.title("Answer")  # Заголовок для колонки ответов
    output_data = st.subheader('')  # Подзаголовок для вывода данных

# Инициализация клиента Google GenAI с API ключом
client = genai.Client(api_key="YOUR_API")

# Инициализация веб-камеры для захвата видео
cap = cv2.VideoCapture(0)  # '0' обычно указывает на встроенную камеру
cap.set(propId=3, value=1280)  # Установка ширины кадра
cap.set(propId=4, value=720)  # Установка высоты кадра

# Инициализация класса HandDetector для отслеживания рук
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def gethandInfo(img):
    # Поиск рук в текущем кадре
    hands, img = detector.findHands(img, draw=False, flipType=True)

    # Проверка, обнаружены ли руки
    if hands:
        hand1 = hands[0]  # Получаем первую обнаруженную руку
        lmList1 = hand1["lmList"]  # Список из 21 маркера для первой руки

        # Подсчет количества поднятых пальцев
        fingers = detector.fingersUp(hand1)
        print(fingers)  # Вывод количества поднятых пальцев
        return fingers, lmList1
    else:
        return None  # Если руки не обнаружены, возвращаем None

def draw(inform, perv_pose, canvas):
    fingers, lmlist = inform  # Извлекаем информацию о пальцах и маркерах
    current_pose = None  # Переменная для текущей позы

    # Проверяем, подняты ли указательный палец
    if fingers == [0, 1, 0, 0, 0]:
        current_pose = lmlist[8][0:2]  # Получаем координаты кончика указательного пальца
        if perv_pose is None: 
            perv_pose = current_pose  # Если предыдущая поза не задана, устанавливаем текущую
        cv2.line(canvas, current_pose, perv_pose, color=(0, 0, 255), thickness=10)  # Рисуем линию

    # Проверяем, подняты ли все пять пальцев
    if fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(canvas)  # Очищаем канва

    return current_pose, canvas  # Возвращаем текущую позу и канвас

def sendAI(canvas, fingers):
    # Если подняты все пальцы, отправляем изображение в AI
    if fingers == [1, 1, 1, 1, 1]:
        pil_image = Image.fromarray(canvas)  # Преобразуем канвас в изображение PIL
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=["Solve this", pil_image],
        )
        return response.text  # Возвращаем текст ответа от AI

# Инициализация переменных
perv_pose = None
canvas = None
image_combines = None
output_text = 'Привет ;)'  # Начальный текст для вывода

# Цикл для непрерывного захвата кадров с веб-камеры
while True:
    success, img = cap.read()  # Захват кадра
    img = cv2.flip(img, 1)  # Отражение изображения по горизонтали

    if canvas is None:
        canvas = np.zeros_like(img)  # Инициализация канваса, если он еще не создан

    inform = gethandInfo(img)  # Получаем информацию о руках
    if inform:
        fingers, lmlist = inform  # Извлекаем количество поднятых пальцев и маркеры
        print(fingers)  # Выводим количество поднятых пальцев в консоль
        perv_pose, canvas = draw(inform, perv_pose, canvas)  # Рисуем на канвасе

        # Отправляем канвас в AI и получаем ответ (можно раскомментировать для использования)
        # output_text = sendAI(canvas, fingers)

    # Объединяем изображение с канвасом
    image_combines = cv2.addWeighted(img, 0.7, canvas, 0.15, 0)  
    FRAME_WIND.image(image_combines, channels="BGR")  # Отображаем объединенное изображение в Streamlit

    output_data.text(output_text)  # Обновляем текст ответа в интерфейсе