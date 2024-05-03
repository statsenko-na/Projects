import settings
import streamlit as st
import requests
import PIL
import os
import sys
import pandas as pd
import numpy as np
import torch
import PIL
import faiss
import settings
import math
from torchvision import models, transforms
from torchvision.transforms import v2

# Add the path to the parent folder of the current script's directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model_path = settings.MODEL
index_path = settings.INDEX
df_path = settings.DF
TOKEN = st.secrets.TOKEN

# Настройка страницы Streamlit
st.set_page_config(
    page_title="Album Cover Genre Classification",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_resources(model_path, index_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    index = faiss.deserialize_index(np.load(index_path))
    return model, index


model, index = load_resources(model_path, index_path)


@st.cache_data
def load_dataframe(df_path):
    return pd.read_pickle(df_path)


df_emb_2 = load_dataframe(df_path)
df_emb_2 = df_emb_2.loc[~df_emb_2.iloc[:, :1280].duplicated()]


def load_image(image_file):
    img = PIL.Image.open(image_file).convert('RGB')
    return img


def get_embeddings(image):
    model_emb = torch.nn.Sequential(*(list(model.children())[:-1]))
    model_emb.eval()
    with torch.no_grad():
        emb = model_emb(image).squeeze()
    return emb.numpy()


def predict_genre(image):
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.Resize((240, 240)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])
    image_1 = transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(image_1)
        probs = torch.nn.functional.softmax(outputs, dim=-1)
        top_p, top_class = probs.topk(3, dim=1)
    emb = get_embeddings(image_1)
    return top_p.squeeze(0).numpy(), top_class.squeeze(0).numpy(), emb

# Функция для поиска похожих обложек


def find_similar_covers(embedding):
    # Поиск 4 ближайших соседей
    D, I = index.search(embedding, 4)

    filtered_results = [(d, i) for d, i in zip(D[0], I[0]) if d > 0.0001][:3]

    distances, indices = zip(
        *filtered_results) if filtered_results else ([], [])

    return np.array(distances), np.array(indices)


genre = ['anime', 'black metal', 'classical', 'country', 'disco', 'edm',
         'jazz', 'pop', 'rap', 'reggae']


def get_download_link(public_key, oauth_token):
    """Получить прямую ссылку на скачивание файла по его публичной ссылке."""
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    headers = {
        "Authorization": f"OAuth {oauth_token}"
    }
    params = {
        "public_key": public_key
    }
    response = requests.get(api_url, headers=headers, params=params)
    if response.status_code == 200:
        download_url = response.json().get('href')
        return download_url
    else:
        return "Error: Не удалось получить ссылку на скачивание."


def main():
    # CSS для уменьшения отступов и изменения размера шрифта
    st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        html {
            font-size: 15px;  /* Размер шрифта для заголовка */
        }
        [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
        [data-testid="stImageCaption"] {
            width: 100% !important;  /* Приоритетное применение стиля ширины */
            text-align: center !important;  /* Приоритетное применение стиля выравнивания текста */
        }
    
    </style>
    """, unsafe_allow_html=True)

    st.subheader('Классификация музыкальных жанров по обложке альбома',
                 divider='rainbow')

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 0

        image_file = st.file_uploader(
            "Загрузите обложку альбома для определения жанра и рекомендаций", type=["png", "jpg", "jpeg"], key=st.session_state["file_uploader_key"])

        if st.button("Clear uploaded files"):
            st.session_state["file_uploader_key"] += 1
            st.rerun()
    with col2:
        if image_file is not None:
            image = load_image(image_file)
            st.image(image, width=300,
                     caption='Загруженная обложка')

    with col3:
        if image_file is not None:
            probs, top_classes, embedding = predict_genre(image)
            st.write("Вероятности жанров:")
            for prob, cls in zip(probs, top_classes):
                prob_pct = round(prob * 100, 1)
                bar_length = 20  # Длина полоски прогресса
                # Длина заполненной части
                filled_length = math.ceil(prob_pct / (100 / bar_length))
                # Зеленая полоса прогресса с дивами
                bar_filled = ''.join(
                    [f"<div style='background-color: #4CAF50; width: 10px; height: 20px; display: inline-block;'></div>" for _ in range(filled_length)])
                # Палочки "-" с соответствующим стилем
                bar_empty = ''.join(
                    [f"<div style='background-color: transparent; color: black; width: 10px; height: 20px; display: inline-block;'>-</div>" for _ in range(bar_length - filled_length)])
                # Устанавливаем фиксированную длину названия жанра
                genre_name = genre[cls].upper().ljust(11)
                # Форматирование строки с HTML и CSS
                st.markdown(
                    f"<div style='font-size: 16px;'><span style='font-weight: bold; display: inline-block; width: 100px; text-align: left;'>{genre_name}</span>"
                    f"<span style='display: inline-block; width: 240px;'>{bar_filled}{bar_empty}</span>"
                    f"<span style='font-weight: bold; display: inline-block; width: 50px; text-align: right; margin-left: 5px;'>{prob_pct}%</span></div>",
                    unsafe_allow_html=True)
    st.subheader('Рекомендации альбомов похожих по обложке',
                     divider='rainbow')
    if image_file is not None:
        D, I = find_similar_covers(np.expand_dims(embedding, axis=0))
        similar_images = []
        for i in I:
            image_path = df_emb_2.iloc[i]['public_url']
            genre_label = df_emb_2.iloc[i]['genre']
            download_url = requests.get(get_download_link(image_path, TOKEN))
            similar_images.append((download_url, genre_label.upper()))

        col4, col5, col6 = st.columns(3)
        with col4:
            st.image(similar_images[0][0].content, width=200,
                     caption=f'Рекомендация 1 - {similar_images[0][1]}')
        with col5:
            st.image(similar_images[1][0].content, width=200,
                     caption=f'Рекомендация 2 - {similar_images[1][1]} ')
        with col6:
            st.image(similar_images[2][0].content, width=200,
                     caption=f'Рекомендация 3 - {similar_images[2][1]}')


if __name__ == "__main__":
    main()
