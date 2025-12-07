# -*- coding: utf-8 -*-
"""
FastAPI приложение для генерации блог-постов с интеграцией Currents API и OpenAI.

Это приложение позволяет:
- Получать актуальные новости по теме через Currents API
- Генерировать блог-посты с использованием новостей как контекста
- Создавать заголовки, мета-описания и контент постов
"""

import os
import logging
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
import openai
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения
app = FastAPI(
    title="Blog Post Generator API",
    description="API для генерации блог-постов с использованием актуальных новостей",
    version="1.0.0"
)

# Получение API ключей из переменных окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CURRENTS_API_KEY = os.getenv("CURRENTS_API_KEY")

# Проверка наличия обязательных API ключей при запуске
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY не найден в переменных окружения")
if not CURRENTS_API_KEY:
    logger.warning("CURRENTS_API_KEY не найден в переменных окружения")

# Настройка OpenAI клиента
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None
    logger.error("OpenAI клиент не инициализирован из-за отсутствия API ключа")

# Базовый URL для Currents API
CURRENTS_API_BASE_URL = "https://api.currentsapi.services/v1"


def escape_markdown_v2(text: str) -> str:
    """
    Экранирует специальные символы для Telegram MarkdownV2.
    
    Args:
        text: Текст для экранирования
        
    Returns:
        Текст с экранированными специальными символами
    """
    # Полный список спецсимволов Telegram MarkdownV2
    special_chars = r"_*[]()~`>#+-=|{}.!"
    for char in special_chars:
        text = text.replace(char, f"\\{char}")
    return text


# Pydantic модели для запросов и ответов
class PostRequest(BaseModel):
    """Модель запроса на генерацию поста"""
    topic: str = Field(..., description="Тема для генерации блог-поста", min_length=1)
    include_news: bool = Field(default=True, description="Использовать ли актуальные новости как контекст")
    max_news: int = Field(default=5, description="Максимальное количество новостей для использования", ge=1, le=20)
    language: str = Field(default="ru", description="Язык для поиска новостей (ru, en, etc.)")
    max_tokens: int = Field(default=2048, description="Максимальное количество токенов для генерации контента", ge=50, le=4096)


class PostResponse(BaseModel):
    """Модель ответа с сгенерированным постом"""
    title: str = Field(..., description="Заголовок поста")
    meta_description: str = Field(..., description="Мета-описание для SEO")
    post_content: str = Field(..., description="Содержимое поста")
    news_used: Optional[List[Dict]] = Field(default=None, description="Использованные новости в качестве контекста")


class HealthResponse(BaseModel):
    """Модель ответа для проверки работоспособности"""
    status: str = Field(..., description="Статус сервиса")
    openai_configured: bool = Field(..., description="Настроен ли OpenAI API")
    currents_configured: bool = Field(..., description="Настроен ли Currents API")


async def get_news_from_currents(topic: str, language: str = "ru", max_results: int = 5) -> List[Dict]:
    """
    Получает актуальные новости по теме через Currents API.
    
    Args:
        topic: Тема для поиска новостей
        language: Язык новостей (ru, en, etc.)
        max_results: Максимальное количество новостей
        
    Returns:
        Список словарей с информацией о новостях
        
    Raises:
        HTTPException: При ошибках запроса к Currents API
    """
    if not CURRENTS_API_KEY:
        logger.warning("Currents API ключ не настроен, пропускаем получение новостей")
        return []
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            # Формируем запрос к Currents API
            params = {
                "apiKey": CURRENTS_API_KEY,
                "keywords": topic,
                "language": language,
                "page_size": max_results
            }
            
            response = await http_client.get(
                f"{CURRENTS_API_BASE_URL}/search",
                params=params
            )
            
            # Проверяем статус ответа
            response.raise_for_status()
            data = response.json()
            
            # Извлекаем новости из ответа
            news_articles = data.get("news", [])
            
            # Форматируем новости для использования
            formatted_news = []
            for article in news_articles[:max_results]:
                formatted_news.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", ""),
                    "published": article.get("published", "")
                })
            
            logger.info(f"Получено {len(formatted_news)} новостей по теме '{topic}'")
            return formatted_news
            
    except httpx.TimeoutException:
        logger.error(f"Таймаут при запросе к Currents API для темы '{topic}'")
        return []
    except httpx.HTTPStatusError as e:
        logger.error(f"Ошибка HTTP при запросе к Currents API: {e.response.status_code}")
        return []
    except Exception as e:
        logger.error(f"Неожиданная ошибка при получении новостей: {str(e)}")
        return []


def format_news_context(news_articles: List[Dict]) -> str:
    """
    Форматирует новости в контекст для промпта OpenAI.
    
    Args:
        news_articles: Список новостей
        
    Returns:
        Отформатированная строка с контекстом новостей
    """
    if not news_articles:
        return ""
    
    context_parts = ["Актуальные новости по теме:\n"]
    for i, news in enumerate(news_articles, 1):
        context_parts.append(
            f"{i}. {news.get('title', 'Без заголовка')}\n"
            f"   {news.get('description', 'Без описания')}\n"
        )
    
    return "\n".join(context_parts)


async def generate_post_with_openai(
    topic: str,
    news_context: str = "",
    max_tokens: int = 2048
) -> Dict[str, str]:
    """
    Генерирует блог-пост используя OpenAI API.
    
    Args:
        topic: Тема поста
        news_context: Контекст из новостей (опционально)
        max_tokens: Максимальное количество токенов
        
    Returns:
        Словарь с заголовком, мета-описанием и контентом поста
        
    Raises:
        HTTPException: При ошибках OpenAI API
    """
    if not client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API не настроен. Проверьте переменную окружения OPENAI_API_KEY"
        )
    
    try:
        # Генерация заголовка
        title_prompt = f"Придумайте привлекательный заголовок для поста на тему: {topic}"
        if news_context:
            title_prompt += f"\n\n{news_context}"
        
        logger.info("Генерация заголовка...")
        title_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": title_prompt}],
            max_tokens=50,
            n=1,
            temperature=0.7,
        )
        title = title_response.choices[0].message.content.strip()
        
        # Генерация мета-описания
        meta_prompt = f"Напишите краткое, но информативное мета-описание для поста с заголовком: {title}"
        if news_context:
            meta_prompt += f"\n\nКонтекст:\n{news_context}"
        
        logger.info("Генерация мета-описания...")
        meta_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": meta_prompt}],
            max_tokens=100,
            n=1,
            temperature=0.7,
        )
        meta_description = meta_response.choices[0].message.content.strip()
        
        # Генерация контента поста
        content_prompt = (
            f"Напишите подробный и увлекательный пост для блога на тему: {topic}. "
            "Используйте короткие абзацы, подзаголовки, примеры и ключевые слова для лучшего восприятия и SEO-оптимизации."
        )
        if news_context:
            content_prompt += f"\n\nИспользуйте следующую актуальную информацию как контекст:\n{news_context}\n\n"
            content_prompt += "Интегрируйте актуальную информацию из новостей в пост, но не копируйте их дословно."
        
        logger.info("Генерация контента поста...")
        content_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content_prompt}],
            max_tokens=max_tokens,
            n=1,
            temperature=0.7,
        )
        post_content = content_response.choices[0].message.content.strip()
        
        return {
            "title": title,
            "meta_description": meta_description,
            "post_content": post_content
        }
        
    except openai.APIError as e:
        logger.error(f"Ошибка OpenAI API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ошибка при обращении к OpenAI API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при генерации поста: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


@app.get("/", tags=["Root"])
async def root():
    """
    Корневой эндпоинт с информацией о API.
    """
    return {
        "message": "Blog Post Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Эндпоинт для проверки работоспособности сервиса.
    
    Проверяет:
    - Доступность сервиса
    - Наличие и корректность API ключей
    """
    return HealthResponse(
        status="healthy",
        openai_configured=bool(OPENAI_API_KEY),
        currents_configured=bool(CURRENTS_API_KEY)
    )


@app.post("/generate", response_model=PostResponse, tags=["Generation"])
async def generate_post(request: PostRequest):
    """
    Генерирует блог-пост по заданной теме.
    
    Процесс генерации:
    1. Получает актуальные новости по теме (если включено)
    2. Генерирует заголовок поста
    3. Генерирует мета-описание
    4. Генерирует контент поста с использованием новостей как контекста
    
    Args:
        request: Запрос с параметрами генерации
        
    Returns:
        Сгенерированный пост с заголовком, мета-описанием и контентом
    """
    try:
        news_articles = []
        news_context = ""
        
        # Получаем новости, если это требуется
        if request.include_news:
            logger.info(f"Получение новостей для темы: {request.topic}")
            news_articles = await get_news_from_currents(
                topic=request.topic,
                language=request.language,
                max_results=request.max_news
            )
            news_context = format_news_context(news_articles)
        
        # Генерируем пост
        logger.info(f"Генерация поста для темы: {request.topic}")
        post_data = await generate_post_with_openai(
            topic=request.topic,
            news_context=news_context,
            max_tokens=request.max_tokens
        )
        
        # Экранируем специальные символы для Telegram MarkdownV2
        escaped_title = escape_markdown_v2(post_data["title"])
        escaped_meta_description = escape_markdown_v2(post_data["meta_description"])
        escaped_post_content = escape_markdown_v2(post_data["post_content"])
        
        # Экранируем новости, если они есть
        escaped_news = None
        if news_articles:
            escaped_news = []
            for news in news_articles:
                escaped_news.append({
                    "title": escape_markdown_v2(news.get("title", "")),
                    "description": escape_markdown_v2(news.get("description", "")),
                    "url": news.get("url", ""),  # URL не нужно экранировать
                    "published": news.get("published", "")
                })
        
        # Формируем ответ
        response_data = PostResponse(
            title=escaped_title,
            meta_description=escaped_meta_description,
            post_content=escaped_post_content,
            news_used=escaped_news if escaped_news else None
        )
        
        logger.info(f"Пост успешно сгенерирован для темы: {request.topic}")
        return response_data
        
    except HTTPException:
        # Пробрасываем HTTP исключения как есть
        raise
    except Exception as e:
        logger.error(f"Критическая ошибка при генерации поста: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при генерации поста: {str(e)}"
        )


@app.get("/topics/suggestions", tags=["Topics"])
async def get_topic_suggestions():
    """
    Возвращает список примеров тем для генерации постов.
    """
    return {
        "topics": [
            "Преимущества медитации",
            "Здоровое питание для занятых людей",
            "Советы по управлению временем",
            "Как начать свой бизнес",
            "Путешествия по бюджету",
            "Цифровая трансформация бизнеса",
            "Экологичный образ жизни",
            "Развитие навыков программирования",
            "Инвестиции для начинающих",
            "Здоровый сон и его влияние на продуктивность"
        ]
    }


# Обработчик исключений для глобальной обработки ошибок
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Глобальный обработчик исключений для логирования всех необработанных ошибок.
    """
    logger.error(f"Необработанное исключение: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Внутренняя ошибка сервера. Проверьте логи для деталей."}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Запуск сервера через uvicorn
    uvicorn.run(
        "app:app",  # Используем app:app, так как файл называется app.py
        host="0.0.0.0",
        port=8000,
        reload=True,  # Автоперезагрузка при изменении кода (для разработки)
        log_level="info"
    )
