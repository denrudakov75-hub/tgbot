import json
import logging
import os
import re
import sys
import time
import calendar
import datetime
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
import telebot
from bs4 import BeautifulSoup
from openai import OpenAI
from telebot import types

# Задайте TELEGRAM_TOKEN и MISTRAL_API_KEY в окружении или в файле .env рядом со скриптом.


def _load_env_file() -> None:
    path = Path(__file__).resolve().parent / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if not key:
            continue
        # Подставляем из .env, если переменная не задана или пустая (часто так в Run Configuration)
        if key not in os.environ or not str(os.environ.get(key, "")).strip():
            os.environ[key] = val


_load_env_file()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "").strip()


def _mistral_outbound_proxy_url() -> str | None:
    """Прокси для HTTPS к api.mistral.ai: MISTRAL_PROXY → TELEGRAM_PROXY → HTTP(S)_PROXY."""
    p = os.environ.get("MISTRAL_PROXY", "").strip()
    if p:
        return p
    if os.environ.get("MISTRAL_INHERIT_TELEGRAM_PROXY", "1").strip().lower() not in ("0", "false", "no"):
        tp = os.environ.get("TELEGRAM_PROXY", "").strip()
        if tp:
            return tp
    return (os.environ.get("HTTPS_PROXY", "") or os.environ.get("HTTP_PROXY", "")).strip() or None


def _mistral_http_trust_env(explicit_proxy: str | None) -> bool:
    """
    Если задан явный прокси — по умолчанию trust_env=False (как у Telegram), чтобы не дублировать HTTP_PROXY.
    MISTRAL_TRUST_ENV_PROXY=1 — всегда подмешивать переменные окружения.
    """
    raw = os.environ.get("MISTRAL_TRUST_ENV_PROXY", "").strip().lower()
    if raw in ("1", "true", "yes"):
        return True
    if raw in ("0", "false", "no"):
        return False
    return explicit_proxy is None


def _create_mistral_openai_client() -> OpenAI:
    """OpenAI-совместимый клиент к Mistral с httpx: прокси и таймауты под «проблемную» сеть."""
    import httpx

    proxy = _mistral_outbound_proxy_url()
    trust_env = _mistral_http_trust_env(proxy)

    raw_to = os.environ.get("MISTRAL_TIMEOUT_SEC", "120").strip()
    try:
        timeout_s = float(raw_to)
    except ValueError:
        timeout_s = 120.0
    timeout_s = max(30.0, min(300.0, timeout_s))
    connect_s = min(60.0, timeout_s)

    raw_retries = os.environ.get("MISTRAL_MAX_RETRIES", "3").strip()
    try:
        max_retries = max(0, min(10, int(raw_retries)))
    except ValueError:
        max_retries = 3

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    h_timeout = httpx.Timeout(timeout_s, connect=connect_s)

    if proxy:
        http_client = httpx.Client(
            proxy=proxy,
            timeout=h_timeout,
            limits=limits,
            trust_env=trust_env,
        )
        print(
            f"[INFO] Mistral API: httpx с прокси (TELEGRAM_PROXY/MISTRAL_PROXY), "
            f"trust_env={trust_env}, таймаут {timeout_s:.0f} с."
        )
    else:
        http_client = httpx.Client(timeout=h_timeout, limits=limits, trust_env=True)
        print(f"[INFO] Mistral API: httpx без явного прокси, trust_env=True, таймаут {timeout_s:.0f} с.")

    return OpenAI(
        api_key=MISTRAL_API_KEY,
        base_url="https://api.mistral.ai/v1",
        http_client=http_client,
        max_retries=max_retries,
    )


def _apply_telegram_session_adapters(sess: requests.Session, *, tls12_only: bool) -> None:
    """
    Адаптер для api.telegram.org: опционально только TLS 1.2.
    TELEGRAM_HTTP_MAX_RETRIES — сколько повторов urllib3 (по умолчанию 0: не ждать лишние
    десятки секунд при каждом SSLEOF — infinity_polling и так перезапрашивает).
    """
    import ssl

    from requests.adapters import HTTPAdapter
    from urllib3.util import Retry

    try:
        from urllib3.util.ssl_ import create_urllib3_context
    except ImportError:
        from requests.packages.urllib3.util.ssl_ import create_urllib3_context  # type: ignore

    raw_retries = os.environ.get("TELEGRAM_HTTP_MAX_RETRIES", "").strip()
    if raw_retries == "":
        max_retries = 0
    else:
        try:
            max_retries = max(0, int(raw_retries))
        except ValueError:
            max_retries = 0

    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        redirect=0,
        backoff_factor=0,
        raise_on_status=False,
    )

    class _TelegramAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            if tls12_only:
                ctx = create_urllib3_context()
                try:
                    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
                    ctx.maximum_version = ssl.TLSVersion.TLSv1_2
                except (AttributeError, ValueError):
                    pass
                kwargs["ssl_context"] = ctx
            return super().init_poolmanager(*args, **kwargs)

    adapter = _TelegramAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)


def _sanitize_bot_token_in_text(s: str) -> str:
    """Убирает токен из строки лога (URL вида /bot123:secret/...)."""
    t = TELEGRAM_TOKEN
    if t:
        s = s.replace(t, "<BOT_TOKEN>")
    return re.sub(r"/bot\d+:[A-Za-z0-9_-]+/", "/bot<token>/", s)


_telegram_net_warn_last: float = 0.0
_telegram_net_warn_suppressed: int = 0


def _warn_telegram_network_throttled(exc: BaseException) -> None:
    """Редко печатаем одно и то же SSLEOF — иначе лог бесконечен; токен не светим."""
    global _telegram_net_warn_last, _telegram_net_warn_suppressed
    verbose = os.environ.get("TELEGRAM_VERBOSE_NET", "").strip().lower() in ("1", "true", "yes")
    now = time.monotonic()
    try:
        interval = float(os.environ.get("TELEGRAM_NET_WARN_INTERVAL_SEC", "120") or "120")
    except ValueError:
        interval = 120.0
    interval = max(15.0, interval)

    safe = _sanitize_bot_token_in_text(repr(exc))
    if verbose:
        print(f"[WARN] Сеть Telegram в обработчике (продолжаем): {safe}")
        return

    if now - _telegram_net_warn_last < interval:
        _telegram_net_warn_suppressed += 1
        return

    extra = ""
    if _telegram_net_warn_suppressed:
        extra = f" (пропущено {_telegram_net_warn_suppressed} похожих за {interval:.0f} с)"
        _telegram_net_warn_suppressed = 0
    _telegram_net_warn_last = now
    short = "SSLEOF / обрыв TLS" if "SSLEOF" in safe or "EOF occurred" in safe else "ошибка HTTP(S)"
    print(
        f"[WARN] Сеть Telegram: {short} к api.telegram.org — polling продолжается.{extra} "
        f"Подробный лог: TELEGRAM_VERBOSE_NET=1. Починка сети: TUN без TELEGRAM_PROXY или другой прокси."
    )


def _configure_telegram_http() -> None:
    """
    По умолчанию requests учитывает HTTP_PROXY/HTTPS_PROXY из окружения — так часто работает VPN-клиент.

    TELEGRAM_IGNORE_ENV_PROXY=1 — не использовать эти переменные (отдельная сессия, trust_env=False).
    Нужно, если в системе «битый» прокси и из‑за него ProxyError, а до Telegram хотите ходить напрямую.

    TELEGRAM_USE_SYSTEM_PROXY=1 — то же, что поведение по умолчанию (оставлено для совместимости).

    TELEGRAM_PROXY=http://... или socks5://... — явный прокси только для Telegram API (для socks5: pip install PySocks).
    Если задан TELEGRAM_PROXY, для запросов к api.telegram.org по умолчанию trust_env=False — иначе VPN часто
    подмешивает второй прокси из HTTP(S)_PROXY и ловят SSLEOF / обрыв TLS.
    TELEGRAM_TRUST_ENV_WITH_PROXY=1 — редкий случай: снова подмешивать HTTP(S)_PROXY вместе с TELEGRAM_PROXY.

    TELEGRAM_FORCE_TLS12=1 — только TLS 1.2 для HTTPS к Telegram (после обрыва TLS 1.3 на прокси).

    TELEGRAM_HTTP_MAX_RETRIES — повторы urllib3 внутри одного запроса (по умолчанию 0, чтобы не копить
    задержки при SSLEOF; infinity_polling всё равно перезапрашивает).

    TELEGRAM_NET_WARN_INTERVAL_SEC — не чаще раз в N секунд краткое предупреждение о сети (по умолчанию 120).
    TELEGRAM_VERBOSE_NET=1 — печатать каждое сетевое исключение (с маскировкой токена).

    TELEGRAM_CONNECT_TIMEOUT / TELEGRAM_READ_TIMEOUT — секунды (по умолчанию 45 и 90).
    """
    import telebot.apihelper as apihelper

    def _timeout_env(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return max(5, int(raw))
        except ValueError:
            return default

    apihelper.CONNECT_TIMEOUT = _timeout_env("TELEGRAM_CONNECT_TIMEOUT", 45)
    apihelper.READ_TIMEOUT = _timeout_env("TELEGRAM_READ_TIMEOUT", 90)

    use_system = os.environ.get("TELEGRAM_USE_SYSTEM_PROXY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    ignore_env = os.environ.get("TELEGRAM_IGNORE_ENV_PROXY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    proxy_url = os.environ.get("TELEGRAM_PROXY", "").strip()
    trust_env_with_proxy = os.environ.get("TELEGRAM_TRUST_ENV_WITH_PROXY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    force_tls12 = os.environ.get("TELEGRAM_FORCE_TLS12", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    isolate_session = (ignore_env and not use_system) or (
        bool(proxy_url) and not trust_env_with_proxy
    )
    if isolate_session:
        sess = requests.Session()
        sess.trust_env = False
        apihelper.session = sess
    elif force_tls12:
        apihelper.session = requests.Session()

    if apihelper.session is not None:
        _apply_telegram_session_adapters(apihelper.session, tls12_only=force_tls12)
        if force_tls12:
            print("[INFO] Telegram HTTP: TELEGRAM_FORCE_TLS12=1 — для HTTPS к api.telegram.org используется только TLS 1.2.")

    if proxy_url:
        apihelper.proxy = {"http": proxy_url, "https": proxy_url}

    if ignore_env and not use_system:
        print("[INFO] Telegram HTTP: TELEGRAM_IGNORE_ENV_PROXY=1 — HTTP(S)_PROXY из окружения не используются.")
    elif proxy_url and not trust_env_with_proxy:
        print(
            "[INFO] Telegram HTTP: TELEGRAM_PROXY задан — для api.telegram.org HTTP(S)_PROXY из окружения "
            "не подмешиваются (иначе часто двойной прокси и SSLEOF). Отключить изоляцию: TELEGRAM_TRUST_ENV_WITH_PROXY=1."
        )
    else:
        print(
            "[INFO] Telegram HTTP: используются HTTP_PROXY/HTTPS_PROXY из окружения (как у многих VPN). "
            "Если был ProxyError — добавьте TELEGRAM_IGNORE_ENV_PROXY=1 в .env."
        )


# Если True — бот будет работать только с командами, без AI-ответов
WORK_WITHOUT_AI = False

if not TELEGRAM_TOKEN:
    print(
        "[FATAL] Не задан TELEGRAM_TOKEN.\n"
        "Установите переменную окружения TELEGRAM_TOKEN или добавьте её в .env рядом с bot2.py."
    )
    sys.exit(1)

_configure_telegram_http()


class TelegramNetworkExceptionHandler(telebot.ExceptionHandler):
    """Сетевые сбои (SSL, таймаут, обрыв) в обработчиках не останавливают polling."""

    def handle(self, exception):  # type: ignore[override]
        if isinstance(exception, requests.exceptions.RequestException):
            _warn_telegram_network_throttled(exception)
            return True
        return False


print("[DEBUG] Инициализация бота...")
bot = telebot.TeleBot(TELEGRAM_TOKEN, exception_handler=TelegramNetworkExceptionHandler())

# Пользователи, ожидающие ввода текста предзаказа после /predzakaz без аргумента
_PREORDER_PENDING_USERS: set[int] = set()


def _preorder_admin_chat_id() -> int | None:
    """Куда слать предзаказы: PREORDER_ADMIN_CHAT_ID в .env (ваш user id или id группы)."""
    raw = os.environ.get("PREORDER_ADMIN_CHAT_ID", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


_BOT_COMMANDS = [
    types.BotCommand("start", "Начать общение с ботом"),
    types.BotCommand("help", "Показать список команд"),
    types.BotCommand("site", "Открыть сайт Бродячей Собаки"),
    types.BotCommand("vk", "Открыть группу ВК"),
    types.BotCommand("vk_org", "Мероприятия и концерты в кафе — написать в ВК"),
    types.BotCommand("phone", "Показать телефон для брони"),
    types.BotCommand("whatsapp", "Показать WhatsApp для связи"),
    types.BotCommand("refund", "Возврат билетов (Ticketscloud для зрителей)"),
    types.BotCommand("forget", "Очистить память диалога с нейросетью в этом чате"),
    types.BotCommand("menu", "Файл меню для предзаказа к программе"),
    types.BotCommand("predzakaz", "Отправить предзаказ администратору"),
    types.BotCommand("event", "Выбрать дату и узнать мероприятия"),
]


def _telegram_request_looks_broken(exc: BaseException) -> bool:
    s = f"{type(exc).__name__} {exc!s}"
    return "api.telegram.org" in s or "SSLEOF" in s or "SSLError" in s


def _print_telegram_unreachable_hints() -> None:
    print(
        "[FATAL] До api.telegram.org запросы не проходят — бот не увидит сообщения и команды, пока сеть не починится.\n"
        "    Что попробовать по очереди:\n"
        "    1) Режим VPN «TUN / системный трафик» — уберите строку TELEGRAM_PROXY из .env (пусть трафик идёт через туннель ОС).\n"
        "    2) Другой тип прокси в .env: socks5h://127.0.0.1:ПОРТ вместо http:// (нужен пакет PySocks).\n"
        "    3) В .env: TELEGRAM_FORCE_TLS12=1 — если обрыв TLS на смешанном HTTP-прокси.\n"
        "    4) Порт и тип прокси в Hiddify (смешанный / HTTP / SOCKS) должны совпадать с URL в TELEGRAM_PROXY.\n"
        "    5) Антивирус / фильтр HTTPS: временно отключить проверку SSL или добавить исключение для python.exe.\n"
        "    Пока падает set_my_commands с SSL/таймаутом — polling тоже не получит getUpdates."
    )


def _register_bot_commands() -> None:
    last_exc: BaseException | None = None
    for attempt in range(1, 4):
        try:
            bot.set_my_commands(_BOT_COMMANDS)
            print("[DEBUG] Команды бота зарегистрированы в Telegram.")
            return
        except requests.exceptions.RequestException as e:
            last_exc = e
            print(f"[WARN] set_my_commands: попытка {attempt}/3 — {_sanitize_bot_token_in_text(str(e))}")
            if attempt < 3:
                time.sleep(5 * attempt)
    print(
        "[WARN] Не удалось зарегистрировать команды в Telegram. "
        "Бот всё равно запустится — проверьте сеть и api.telegram.org."
    )
    if last_exc is not None and _telegram_request_looks_broken(last_exc):
        _print_telegram_unreachable_hints()


_register_bot_commands()

_preorder_dest = _preorder_admin_chat_id()
if _preorder_dest:
    print(f"[INFO] Предзаказы (/predzakaz) пересылаются в Telegram, chat_id={_preorder_dest}")
else:
    print("[INFO] Предзаказы администратору выключены — задайте PREORDER_ADMIN_CHAT_ID в .env (ваш Telegram id).")


def _safe_answer_callback_query(callback_query_id: str, text: str | None = None) -> None:
    """answer_callback_query с повторами — иначе SSL/EOF роняет поток polling."""
    for attempt in range(1, 4):
        try:
            bot.answer_callback_query(callback_query_id, text=text)
            return
        except requests.exceptions.RequestException as e:
            print(f"[WARN] answer_callback_query попытка {attempt}/3: {_sanitize_bot_token_in_text(str(e))}")
            if attempt < 3:
                time.sleep(1.5 * attempt)


client: OpenAI | None = None
if MISTRAL_API_KEY:
    print("[DEBUG] Инициализация клиента Mistral...")
    try:
        client = _create_mistral_openai_client()
    except Exception as e:
        print(f"[WARN] Не удалось создать httpx-клиент для Mistral ({e}) — клиент по умолчанию, timeout=120.")
        try:
            client = OpenAI(
                api_key=MISTRAL_API_KEY,
                base_url="https://api.mistral.ai/v1",
                timeout=120.0,
                max_retries=3,
            )
        except Exception as e2:
            print(f"[ERROR] Mistral OpenAI-клиент: {e2}")
            client = None
    if client:
        print("[DEBUG] Бот готов к работе (Mistral).")
else:
    print("[WARN] MISTRAL_API_KEY не задан — ответы нейросети и описание афиши недоступны.")


def _user_display_name(user) -> str:
    parts = [user.first_name, user.last_name]
    return " ".join(p for p in parts if p).strip() or "гость"


MONTHS_GENITIVE = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}

# Родительный падеж для фраз «5 февраля 2026 года»
MONTHS_GENITIVE_NUM = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря",
}

# Предложный падеж для заголовка календаря: «в феврале 2026 года»
MONTHS_PREPOSITIONAL = {
    1: "январе",
    2: "феврале",
    3: "марте",
    4: "апреле",
    5: "мае",
    6: "июне",
    7: "июле",
    8: "августе",
    9: "сентябре",
    10: "октябре",
    11: "ноябре",
    12: "декабре",
}

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SobakaTelegramBot/1.0; +https://sobaka.su)",
}

# ВК: личные сообщения сообщества — вопросы о проведении мероприятий/концертов в кафе
VK_ORG_MESSAGES_URL = "https://vk.com/kabaresobaka"

# Телефон и администраторы (имена не с сайта — фиксированные для ответов гостям)
PHONE_DISPLAY_FULL = "+7 (383) 218-80-70"
PHONE_DISPLAY_SHORT = "218-80-70"
ADMIN_PHONE_NAMES = "Данил и Анастасия"

# Файл меню для гостей (/menu): названия и цены — MENU_FILE или menu.docx / menu.txt рядом с ботом.
# Состав, ингредиенты, граммовки для Mistral — отдельно: sostav.docx или SOSTAV_DOCX_FILE (не путать с menu).
_MENU_DEFAULT_NAMES = ("menu.docx", "menu.pdf", "menu.txt", "menyu.pdf", "меню.pdf")

_OFFLINE_MENU_TEXT_CACHE: tuple[str, float, str] | None = None  # текст, mtime, путь
_SOSTAV_COMPOSITION_TEXT_CACHE: tuple[str, float, str] | None = None

_WIKI_SNIPPET_CACHE: dict[str, tuple[str, float]] = {}
_WIKI_SNIPPET_TTL_SEC = 45 * 60

try:
    from docx import Document as _DocxDocument  # type: ignore
except ImportError:
    _DocxDocument = None

# Хвост к ответам про билеты/афишу на дату (предзаказ + телефон)
_TICKETS_PREORDER_MARKER = "Предзаказ блюд и напитков к программе"


def _menu_file_path() -> Path | None:
    raw = os.environ.get("MENU_FILE", "").strip()
    base = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else base / p)
    for name in _MENU_DEFAULT_NAMES:
        candidates.append(base / name)
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p
    return None


def _sostav_docx_path() -> Path | None:
    """
    Файл состава/граммовок (только .docx). Не совпадает с файлом меню для гостя.
    Приоритет: SOSTAV_DOCX_FILE → sostav.docx → MENU_DOCX_FILE (если это другой файл, не menu).
    """
    base = Path(__file__).resolve().parent
    candidates: list[Path] = []
    for env_key in ("SOSTAV_DOCX_FILE", "MENU_DOCX_FILE"):
        raw = os.environ.get(env_key, "").strip()
        if raw:
            p = Path(raw)
            candidates.append(p if p.is_absolute() else base / p)
    candidates.append(base / "sostav.docx")
    menu_p = _menu_file_path()
    menu_key = str(menu_p.resolve()) if menu_p else ""
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        if not p.is_file() or p.suffix.lower() != ".docx":
            continue
        if menu_key and key == menu_key:
            continue
        return p
    return None


def _sostav_docx_max_chars() -> int:
    for key in ("SOSTAV_DOCX_MAX_CHARS", "MENU_DOCX_MAX_CHARS"):
        raw = os.environ.get(key, "").strip()
        if raw:
            try:
                lim = int(raw or "16000")
                return max(2000, min(32000, lim))
            except ValueError:
                break
    return 16000


def _iter_docx_body_blocks(document) -> list[tuple[str, object]]:
    """Параграфы и таблицы в порядке как в документе (не «все абзацы, потом все таблицы»)."""
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    out: list[tuple[str, object]] = []
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            out.append(("p", Paragraph(child, document)))
        elif isinstance(child, CT_Tbl):
            out.append(("t", Table(child, document)))
    return out


def _sostav_mostly_uppercase_line(t: str) -> bool:
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 3:
        return False
    up = sum(1 for c in letters if c.isupper() or c == "Ё")
    return up / len(letters) >= 0.82


def _sostav_paragraph_is_section_heading(para) -> bool:
    """
    Заголовок раздела: стиль «Заголовок 1» / Heading 1, уровень outline в Word, КАПС-строка категории,
    или короткий жирный абзац (но не стиль «Обычный» / список — иначе названия блюд жирным станут «разделами»).
    """
    try:
        st_raw = para.style.name or ""
        st = st_raw.lower()
    except Exception:
        st_raw = ""
        st = ""
    t = para.text.strip()
    if not t:
        return False
    if st and any(x in st for x in ("toc", "оглавлен", "содержан", "верхний колонтитул", "нижний колон")):
        return False
    # Явные стили заголовков Word (в т.ч. «Заголовок 1», Heading 1 — в имени стиля есть «заголовок» / «heading»)
    if st and any(x in st for x in ("heading", "заголовок")):
        return len(t) <= 400
    if st and "title" in st and "subtitle" not in st:
        return len(t) <= 400
    if len(t) > 200:
        return False
    # Уровень структуры в XML (иногда стиль сбит, а outline остаётся)
    try:
        from docx.oxml.ns import qn

        p_pr = para._element.pPr
        if p_pr is not None:
            ol = p_pr.find(qn("w:outlineLvl"))
            if ol is not None:
                val = ol.get(qn("w:val"))
                if val is not None and str(val).isdigit() and int(val) <= 1:
                    return True
    except Exception:
        pass
    if re.search(r"\d{1,4}\s*(₽|руб\.?)\b", t, re.IGNORECASE):
        return False
    # Обычный текст / список: не считаем жирное название блюда заголовком раздела
    body_like = any(
        x in st
        for x in (
            "normal",
            "обычн",
            "list paragraph",
            "listparagraph",
            "абзац списка",
            "абзац",
            "no spacing",
            "без интервала",
            "caption",
            "подпись",
        )
    )
    # Строка вида «Красное, п/сладкое» в капсе — позиция, не раздел
    if _sostav_mostly_uppercase_line(t) and ("," in t or "/" in t) and len(t) < 90:
        return False
    if _sostav_mostly_uppercase_line(t) and len(t) >= 4:
        return True
    runs = [r for r in para.runs if r.text.strip()]
    if runs and all(r.bold for r in runs) and len(t) < 130 and not body_like:
        return True
    return False


def _build_sostav_docx_structured_plaintext(doc, *, doc_role: str = "меню (названия и цены)") -> str:
    """
    Текст для промпта: [РАЗДЕЛ] заголовок, под ним строки «  • пункт» до следующего раздела.
    Сохраняет порядок из Word (таблицы между абзацами на своих местах).
    """
    lines: list[str] = []
    in_section = False
    for kind, block in _iter_docx_body_blocks(doc):
        if kind == "p":
            para = block
            t = para.text.strip()
            if not t:
                continue
            if _sostav_paragraph_is_section_heading(para):
                lines.append(f"[РАЗДЕЛ] {t}")
                in_section = True
            elif in_section:
                lines.append(f"  • {t}")
            else:
                lines.append(t)
        else:
            tbl = block
            lines.append("")
            lines.append("[ТАБЛИЦА]")
            for row in tbl.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                row_txt = " | ".join(x for x in cells if x)
                if row_txt:
                    lines.append(f"  • {row_txt}")
            lines.append("")
            in_section = True

    note = (
        f"Структура файла ({doc_role}): строка «[РАЗДЕЛ] …» — заголовок категории; все строки «  • …» ниже относятся "
        "только к этому разделу до следующей строки «[РАЗДЕЛ]» или до «[ТАБЛИЦА]». "
        "Блок «[ТАБЛИЦА]» — таблица; каждая строка с «  • » — ряд таблицы.\n\n"
    )
    return note + "\n".join(lines)


_STRUCTURED_SECTION_HEAD_RE = re.compile(r"^\[РАЗДЕЛ\]\s*(.+?)\s*$", re.MULTILINE)
# Строки позиций в разметке из Word (отступы и маркеры могут отличаться)
_MENU_BULLET_LINE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^ {2}•\s+(.+)$"),
    re.compile(r"^[ \t]*[•\u2022\u00b7·]\s+(.+)$"),
    re.compile(r"^[ \t]{0,3}[–—\-]\s+(.+)$"),
)


def _append_bullets_from_body(body: str, seen: set[str], out: list[str]) -> None:
    for ln in body.splitlines():
        for cre in _MENU_BULLET_LINE_RES:
            m = cre.match(ln)
            if not m:
                continue
            line = m.group(1).strip()
            if not line or line.startswith("[ТАБЛИЦА]"):
                break
            if line not in seen:
                seen.add(line)
                out.append(line)
            break

# Служебные слова вопроса — не используются для поиска раздела по заголовку
_SECTION_FOCUS_STOPWORDS = frozenset(
    "какие какой какая какое как каково что у вас есть ли в кафе напишите расскажите подскажите "
    "мне хочу заказать сколько стоит цена стоимость все покажите список назовите только также ещё еще "
    "кратко подробно пожалуйста можно нужно нужен нужна нужны очень ли это этот эта эти того тем "
    "дайте дай дайте посмотреть подскажи расскажи всё всех весь всю всем из под над при про для "
    "сколько грамм порция раздел категорию категории тебе вам бот расскажи скажи напиши хочу "
    "блюдо блюда блюд напиток напитки закуск закуски десерт десерты салат салаты суп супы".split()
)


def _split_structured_doc_into_sections(marked_text: str) -> list[tuple[str, str]]:
    """Разбивает текст с маркерами [РАЗДЕЛ] на пары (заголовок, тело до следующего раздела)."""
    if not marked_text or "[РАЗДЕЛ]" not in marked_text:
        return []
    matches = list(_STRUCTURED_SECTION_HEAD_RE.finditer(marked_text))
    if not matches:
        return []
    out: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(marked_text)
        body = marked_text[start:end].strip()
        out.append((title, body))
    return out


def _keywords_for_section_focus(user_text: str) -> list[str]:
    t = user_text.lower().replace("ё", "е")
    words = re.findall(r"[\w\-]{3,}", t)
    out: list[str] = []
    seen: set[str] = set()
    for w in words:
        if w in _SECTION_FOCUS_STOPWORDS:
            continue
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out[:22]


def _ru_word_stem_overlap(a: str, b: str, *, min_len: int = 5, max_prefix: int = 6) -> bool:
    """Совпадение начала слова (падежи: грузия / грузии / грузину — общий префикс «грузи»)."""
    if len(a) < min_len or len(b) < min_len:
        return False
    n = min(len(a), len(b), max_prefix)
    return a[:n] == b[:n]


def _menu_title_region_substrings_required(user_text: str) -> tuple[str, ...]:
    """
    Если в вопросе указан регион/страна, заголовок [РАЗДЕЛ] должен содержать эту привязку.
    Иначе слово «вина» цепляет любой раздел с «вина» в названии (БЕЛЫЕ ВИНА, ИТАЛИЯ…), и в ответ попадает не тот список.
    """
    t = user_text.lower().replace("ё", "е")
    reqs: list[str] = []
    if re.search(r"грузи", t) or "georgia" in t or "georgian" in t:
        reqs.append("грузи")
    if re.search(r"итал", t) or "italy" in t or "italian" in t:
        reqs.append("итал")
    if re.search(r"франц", t) or "france" in t or "french" in t:
        reqs.append("франц")
    if re.search(r"испан", t) or "spain" in t or "spanish" in t:
        reqs.append("испан")
    if re.search(r"португал", t) or "portugal" in t or "portuguese" in t:
        reqs.append("португал")
    if "champagne" in t or "шампан" in t:
        reqs.append("шампан")
    return tuple(reqs)


def _section_title_matches_region_filter(title: str, required_substrings: tuple[str, ...]) -> bool:
    if not required_substrings:
        return True
    tl = title.lower().replace("ё", "е")
    return any(sub in tl for sub in required_substrings)


def _section_title_matches_user_query(title: str, user_text: str, keywords: list[str]) -> bool:
    """Совпадение слов или биграммы из вопроса с заголовком раздела (например «грузии» → «ВИНА ГРУЗИИ»)."""
    tl = title.lower().replace("ё", "е")
    ut = user_text.lower().replace("ё", "е")
    title_words = re.findall(r"[\w\-]{3,}", tl)
    for kw in keywords:
        if kw in tl:
            return True
        for tw in title_words:
            if len(kw) >= 4 and len(tw) >= 4 and (kw in tw or tw in kw):
                return True
            if _ru_word_stem_overlap(kw, tw):
                return True
    toks = [w for w in re.findall(r"[\w\-]{3,}", ut) if w not in _SECTION_FOCUS_STOPWORDS]
    for i in range(len(toks) - 1):
        bigram = f"{toks[i]} {toks[i + 1]}"
        if len(bigram) >= 7 and bigram in tl:
            return True
    return False


def _build_menu_focus_excerpt_by_headings(menu_marked: str, sostav_marked: str, user_text: str) -> str:
    """
    Вырезает из размеченных файлов разделы, в заголовке [РАЗДЕЛ] которых есть ключевые слова из вопроса,
    до следующего [РАЗДЕЛ] (как в Word: Заголовок 1 → содержимое до следующего Заголовок 1).
    """
    keywords = _keywords_for_section_focus(user_text)
    if not keywords:
        return ""
    regions = _menu_title_region_substrings_required(user_text)
    chunks: list[str] = []
    for label, blob in (("файл меню (цены)", menu_marked), ("файл состава", sostav_marked)):
        if not blob or "[РАЗДЕЛ]" not in blob:
            continue
        parts: list[str] = []
        for title, body in _split_structured_doc_into_sections(blob):
            if not _section_title_matches_region_filter(title, regions):
                continue
            if _section_title_matches_user_query(title, user_text, keywords):
                parts.append(f"[РАЗДЕЛ] {title}\n{body}")
        if parts:
            chunks.append(f"--- {label} ---\n" + "\n\n".join(parts))
    return "\n\n".join(chunks).strip()


def _user_asks_menu_bulk_list(text: str) -> bool:
    """
    Гость хочет перечень позиций в категории (вина, закуски и т.д.), а не цену одной позиции или состав.
    Для таких запросов ответ формируется дословно из файла меню без Mistral.
    """
    t = text.lower().replace("ё", "е")
    if any(
        x in t
        for x in (
            "состав",
            "из чего",
            "ингредиент",
            "аллерген",
            "грамм",
            "кбжу",
            "бжу",
            "калори",
            "рецепт",
            "сколько грамм",
            "сколько вес",
            "порция сколько",
        )
    ):
        return False
    if "есть ли" in t:
        return False
    if re.search(r"сколько\s+стоит", t) or ("стоимость" in t and "список" not in t and "какие" not in t):
        return False
    if any(x in t for x in ("посовет", "порекоменду", "рекоменд", "подойдет", "подойдёт", "лучше к ", "к столу")):
        return False
    indicators = (
        "какие",
        "что у вас",
        "список",
        "перечисли",
        "назови все",
        "назовите все",
        "полный список",
        "все вина",
        "все напитки",
        "что есть из",
        "какие у вас",
        "покажи",
        "покажите",
        "wine",
        "wines",
        "list ",
        "list,",
        "list?",
        "what kind",
        "which ",
        "show me",
    )
    if any(i in t for i in indicators):
        return True
    more = (
        "названия",
        "называют",
        "винная",
        "карта вин",
        "что налива",
        "что по ",
        "по винам",
        "по напиткам",
        "позици",
        "варианты",
        "что прода",
        "что предлага",
        "что у вас по",
        "дайте варианты",
        "дай варианты",
        "перечень",
        "каталог",
    )
    if any(i in t for i in more):
        return True
    # Короткая формулировка про категорию без уточнения одной позиции
    if len(t) < 100 and not re.search(r"\d{3,}\s*р", t):
        if any(x in t for x in ("вина", "вино ", "вино,", "вино?", "вино.", "wine", "wines")):
            return True
        if "напитк" in t and "меню" not in t:
            return True
    return False


def _extract_menu_bullets_best_effort(menu_marked: str, user_text: str) -> list[str]:
    """
    Дословные строки позиций из menu.docx: несколько проходов, чтобы попасть в нужный раздел
    (сочетание региона в вопросе и заголовка, затем только регион, затем только ключевые слова).
    """
    if not menu_marked or "[РАЗДЕЛ]" not in menu_marked:
        return []
    keywords = _keywords_for_section_focus(user_text)
    regions = _menu_title_region_substrings_required(user_text)
    sections = _split_structured_doc_into_sections(menu_marked)

    def run(title_filter) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for title, body in sections:
            if not title_filter(title):
                continue
            _append_bullets_from_body(body, seen, out)
        return out

    # 1) Регион (если задан) + совпадение заголовка с ключевыми словами из вопроса
    if keywords:

        def tf1(title: str) -> bool:
            if not _section_title_matches_region_filter(title, regions):
                return False
            return _section_title_matches_user_query(title, user_text, keywords)

        out1 = run(tf1)
        if out1:
            return out1

    # 2) Только регион в заголовке — если спросили «вина Грузии», а в заголовке нет слова «вина»
    if regions:

        def tf2(title: str) -> bool:
            return _section_title_matches_region_filter(title, regions)

        out2 = run(tf2)
        if out2:
            return out2
        # Регион указан, но разделов с ним в заголовке нет — не падать на «любые вина»
        return []

    # 3) Регион в вопросе не задан: подходят все разделы, где заголовок совпал с ключевыми словами
    if keywords:

        def tf3(title: str) -> bool:
            return _section_title_matches_user_query(title, user_text, keywords)

        return run(tf3)

    return []


def _extract_menu_bullets_for_matching_sections(menu_marked: str, user_text: str) -> list[str]:
    """Совместимость: то же, что _extract_menu_bullets_best_effort."""
    return _extract_menu_bullets_best_effort(menu_marked, user_text)


def load_offline_menu_text_for_mistral() -> str:
    """
    Названия и цены для промпта: тот же файл, что отправляется гостю по /menu.
    Поддержка: .docx и .txt. PDF не читается.
    Лимит: MENU_DOCX_MAX_CHARS / SOSTAV_DOCX_MAX_CHARS (по умолчанию 16000).
    """
    global _OFFLINE_MENU_TEXT_CACHE
    path = _menu_file_path()
    if path is None:
        return ""

    try:
        mtime = path.stat().st_mtime
    except OSError:
        return ""

    path_key = str(path.resolve())
    if _OFFLINE_MENU_TEXT_CACHE and _OFFLINE_MENU_TEXT_CACHE[2] == path_key and _OFFLINE_MENU_TEXT_CACHE[1] == mtime:
        return _OFFLINE_MENU_TEXT_CACHE[0]

    suf = path.suffix.lower()
    text = ""
    try:
        if suf == ".docx":
            if _DocxDocument is None:
                return ""
            doc = _DocxDocument(str(path))
            text = _build_sostav_docx_structured_plaintext(
                doc,
                doc_role="меню для гостя: названия блюд и цены (файл как по /menu)",
            )
        elif suf == ".txt":
            raw = path.read_text(encoding="utf-8", errors="replace").strip()
            text = (
                "Текст меню из файла .txt (разделы не размечены автоматически):\n\n" + raw
                if raw
                else ""
            )
        else:
            text = ""
    except Exception as e:
        print(f"[WARN] Не удалось прочитать меню {path.name}: {e}")
        return ""

    lim = _sostav_docx_max_chars()
    if len(text) > lim:
        text = text[: lim - 20] + "\n…(обрезано по лимиту длины текста меню)"

    _OFFLINE_MENU_TEXT_CACHE = (text, mtime, path_key)
    return text


def load_sostav_composition_text_for_mistral() -> str:
    """
    Состав блюд, ингредиенты, граммовки из sostav.docx (или SOSTAV_DOCX_FILE).
    Только .docx; не должен совпадать с файлом меню для гостя.
    """
    global _SOSTAV_COMPOSITION_TEXT_CACHE
    if _DocxDocument is None:
        return ""
    path = _sostav_docx_path()
    if path is None:
        return ""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return ""
    path_key = str(path.resolve())
    if (
        _SOSTAV_COMPOSITION_TEXT_CACHE
        and _SOSTAV_COMPOSITION_TEXT_CACHE[2] == path_key
        and _SOSTAV_COMPOSITION_TEXT_CACHE[1] == mtime
    ):
        return _SOSTAV_COMPOSITION_TEXT_CACHE[0]
    try:
        doc = _DocxDocument(str(path))
        text = _build_sostav_docx_structured_plaintext(
            doc,
            doc_role="состав блюд, ингредиенты и граммовки (sostav)",
        )
    except Exception as e:
        print(f"[WARN] Не удалось прочитать состав {path.name}: {e}")
        return ""
    lim = _sostav_docx_max_chars()
    if len(text) > lim:
        text = text[: lim - 20] + "\n…(обрезано по лимиту длины файла состава)"
    _SOSTAV_COMPOSITION_TEXT_CACHE = (text, mtime, path_key)
    return text


_WIKI_QUERY_STOPWORDS = frozenset(
    "какой какая какое какие что это скажи расскажи подскажи есть ли у вас в кафе блюдо блюда меню "
    "состав из чего сделан сделано сделаны ингредиенты ингредиент сколько стоит стоимость цена цены "
    "мне нужно хочу заказать порцию грамм порции напишите расскажите объясните подробно кратко очень "
    "там тут здесь они вы нам мне по вашему ваше ваш ваши подают подается рецепт классический обычно "
    "традиционный стандартный".split()
)


def _wiki_search_phrase_from_user_text(text: str) -> str:
    words = re.findall(r"[\w\-]+", text.lower().replace("ё", "е"))
    kept = [w for w in words if w not in _WIKI_QUERY_STOPWORDS and len(w) > 1]
    if not kept:
        s = re.sub(r"\s+", " ", text.strip())
        return s[:120] if s else ""
    return " ".join(kept[:14])[:120]


def fetch_wikipedia_ru_snippet_for_query(user_text: str) -> str:
    """
    Краткий текст из ru.wikipedia.org по запросу гостя (открытый API).
    Отключается: SOSTAV_WIKIPEDIA=0. (Для вопросов про меню бот Википедию не подмешивает.)
    """
    raw = os.environ.get("SOSTAV_WIKIPEDIA", "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return ""
    phrase = _wiki_search_phrase_from_user_text(user_text)
    if len(phrase.strip()) < 2:
        return ""

    now = time.time()
    ck = phrase.casefold()
    hit = _WIKI_SNIPPET_CACHE.get(ck)
    if hit and now - hit[1] < _WIKI_SNIPPET_TTL_SEC:
        return hit[0]

    api = "https://ru.wikipedia.org/w/api.php"
    try:
        r1 = requests.get(
            api,
            params={
                "action": "opensearch",
                "search": phrase,
                "limit": 2,
                "namespace": 0,
                "format": "json",
            },
            timeout=12,
            headers=HTTP_HEADERS,
        )
        r1.raise_for_status()
        data = r1.json()
        titles = data[1] if isinstance(data, list) and len(data) > 1 else []
        if not titles or not isinstance(titles[0], str):
            _WIKI_SNIPPET_CACHE[ck] = ("", now)
            return ""
        title = titles[0]
        r2 = requests.get(
            api,
            params={
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "titles": title,
            },
            timeout=12,
            headers=HTTP_HEADERS,
        )
        r2.raise_for_status()
        q = r2.json().get("query", {}).get("pages", {})
        page = next(iter(q.values()), None)
        ext = (page or {}).get("extract") or ""
        ext = ext.strip()
        if not ext:
            _WIKI_SNIPPET_CACHE[ck] = ("", now)
            return ""
        try:
            max_e = int(os.environ.get("SOSTAV_WIKIPEDIA_MAX_CHARS", "2200") or "2200")
        except ValueError:
            max_e = 2200
        max_e = max(400, min(6000, max_e))
        if len(ext) > max_e:
            ext = ext[: max_e - 25] + "\n…(фрагмент обрезан)"
        out = f"Статья: {title}\n{ext}"
        _WIKI_SNIPPET_CACHE[ck] = (out, now)
        return out
    except Exception as e:
        print(f"[WARN] Wikipedia (состав блюд): {e}")
        _WIKI_SNIPPET_CACHE[ck] = ("", now)
        return ""


def _user_asks_dish_composition(text: str) -> bool:
    """Вопросы про меню, состав, блюда — файлы menu (цены, названия) и sostav.docx (состав, граммы), без сайта."""
    t = text.lower().replace("ё", "е")
    if "билет" in t and not any(
        x in t
        for x in (
            "меню",
            "блюд",
            "блюдо",
            "состав",
            "ингредиент",
            "салат",
            "суп",
            "еду",
            "кухн",
            "закуск",
            "десерт",
            "напиток",
            "горячее",
        )
    ):
        return False
    keys = (
        "состав",
        "ингредиент",
        "из чего",
        "что входит",
        "входит в",
        "аллерген",
        "веган",
        "вегетариан",
        "без глютен",
        "лактоз",
        "остро",
        "острый",
        "калори",
        "кбжу",
        "бжу",
        "порция",
        "грамм",
        "рецепт",
        "меню",
        "блюд",
        "блюдо",
        "закуск",
        "салат",
        "суп",
        "горячее",
        "десерт",
        "напиток",
        "коктейль",
        "вин",
        "пив",
        "какие",
        "что такое",
        "что у вас",
        "есть ли",
        "сколько стоит",
        "цена",
        "стоимость",
        "предложите",
        "порекомендуй",
    )
    # Английские вопросы: в «wine» нет подстроки «вин» — иначе в промпт не попадает файл меню и модель выдумывает вина.
    en_keys = (
        "wine",
        "wines",
        "cocktail",
        "cocktails",
        "beer",
        "appetizer",
        "salad",
        "soup",
        "dish",
        "dishes",
        "drink",
        "drinks",
        "menu",
        "georgian",
        "georgia",
        "ingredient",
        "ingredients",
        "portion",
        "breakfast",
        "brunch",
        "lunch",
        "dinner",
        "champagne",
        "coffee",
        "tea",
        "whiskey",
        "whisky",
        "vodka",
        "rum",
        "cider",
        "juice",
        "snack",
        "snacks",
        "burger",
        "pizza",
        "pasta",
        "steak",
        "beverage",
        "alcohol",
        "kitchen",
    )
    return any(k in t for k in keys) or any(k in t for k in en_keys)


def _user_requests_menu_file_attachment(text: str) -> bool:
    """
    Гость явно хочет получить в чат файл меню (как /menu).
    Вопросы «сколько стоит X», «из чего салат», «какие вина» — без отправки файла.
    Предзаказ с меню обрабатывается отдельным блоком выше (там файл уже уходит).
    """
    t = text.lower().replace("ё", "е")
    phrases = (
        "файл меню",
        "документ меню",
        "меню файл",
        "меню в файле",
        "скачать меню",
        "пришли меню",
        "пришлите меню",
        "отправь меню",
        "отправьте меню",
        "скинь меню",
        "скиньте меню",
        "кинь меню",
        "киньте меню",
        "выгрузи меню",
        "вышли меню",
        "вышлите меню",
        "дай меню файл",
        "дайте меню файл",
        "полное меню",
        "все меню списком",
        "меню целиком",
        "целиком меню",
        "где меню",
        "как получить меню",
        "как скачать меню",
        "покажи файл меню",
        "покажите файл меню",
        "пришли файл",
        "пришлите файл",
        "скинь файл меню",
        "хочу файл меню",
        "нужен файл меню",
        "прайс файл",
        "меню ворд",
        "меню docx",
        "меню pdf",
    )
    if any(p in t for p in phrases):
        return True
    stripped = t.strip().rstrip("!.?")
    if stripped in (
        "меню",
        "меню!",
        "меню?",
        "хочу меню",
        "дай меню",
        "дайте меню",
        "дай файл меню",
        "отправь файл",
    ):
        return True
    # Явно про меню как документ, без конкретного блюда в одном коротком сообщении
    if len(t) < 100 and ("вышли" in t or "пришли" in t) and "меню" in t and "блюд" not in t:
        return True
    return False


MENU_FILE_AUTHORITY_PREFIX = (
    "КРИТИЧНО: позиции и цены для ответа — только дословное копирование строк «  • …» из блока «ФАЙЛ МЕНЮ» (и при необходимости состава из «ФАЙЛ СОСТАВА»). "
    "Запрещено подставлять «стандартные» названия вин латиницей или транслитом (Kindzmarauli, Khvanchkara, Saperavi, Tsinandali, Mukuzani и т.п.), если такого написания нет в этих строках файла. "
    "Если в файле кириллица (например «Мимико …») — в ответе только кириллица как в файле, без перевода на латиницу."
)

MENU_HISTORY_NOT_AUTHORITY = (
    "История переписки ниже не является источником меню. Названия и цены бери только из блоков «ФАЙЛ МЕНЮ» и «ФАЙЛ СОСТАВА» в этом системном сообщении; "
    "если в прошлых репликах были другие вина или цены — игнорируй и отвечай строго по файлу."
)

MENU_VERBATIM_NO_MATCH = (
    "Под этот запрос в файле меню не найдено ни одной позиции в подходящем разделе. "
    "Проверьте заголовки «Заголовок 1» в Word: в названии раздела должны быть слова из вопроса "
    "(для грузинских вин — например «Грузия» / «Грузии»). Позиции должны быть обычным текстом под заголовком или в таблице. "
    "Бот не выдумывает названия и цены. Откройте /menu или обратитесь к администратору."
)

MENU_COMPOSITION_EXTRA_SYSTEM = (
    "Сейчас вопрос про меню, состав блюд, напитки, ингредиенты, граммовки или цены. "
    "Не начинай с «Вот меню…», «Вот основное меню…» и не предлагай «напишите, какой раздел нужен» — сразу ответ по вопросу. "
    "Если есть блок «ВЫРЕЗКА РАЗДЕЛА ПО КЛЮЧЕВЫМ СЛОВАМ» — для вопроса про эту категорию (например вина Грузии) опирайся на него в первую очередь: это ровно один раздел меню до следующего заголовка, как в Word. "
    "Два источника: (1) блок «ФАЙЛ МЕНЮ» — только названия блюд/напитков и цены, как у гостя; (2) блок «ФАЙЛ СОСТАВА» — состав, ингредиенты, веса/граммовки. "
    "Цены и перечень позиций по категориям бери только из блока меню. Состав, из чего сделано, порции в граммах — только из блока состава; если блока состава нет или там нет строки про блюдо — так и скажи, не выдумывай ингредиенты. "
    "Названия блюд, вин и напитков — только копия текста из строк «  • …» блока меню (кириллица/латиница как в файле); не переводи на латиницу и не подставляй из памяти известные сорта или цены. "
    "Если гость спрашивает по-английски — всё равно перечисляй позиции теми словами, как они записаны в русском (или как в файле) блоке меню, без «международных» замен названий. "
    "Разметка в обоих файлах: «[РАЗДЕЛ] …» — заголовок; строки «  • …» — пункты раздела до следующего [РАЗДЕЛ] или [ТАБЛИЦА]. "
    "Если гость спрашивает про раздел целиком (например «какие вина»), ориентируйся на [РАЗДЕЛ] в блоке меню и перечисли только пункты оттуда. "
    "Запрещено называть позиции, которых нет в блоке меню, и ингредиенты, которых нет в блоке состава — ни из памяти, ни с сайта, ни из Википедии. "
    "Не перечисляй всё меню подряд, если спросили про одну категорию или одно блюдо. "
    "В этом запросе нет фрагментов sobaka.su про меню — только файлы ниже. "
    "Если данных нет в нужном файле — так и скажи; не подменяй ответ общими кулинарными сведениями. "
    "Не добавляй в конец призыв предзаказать, телефон и «всё в файле» — это не твоя задача в этом ответе. "
    "Если гость спрашивает только состав или ингредиенты — ответь кратко. "
    "Не комментируй, откуда у гостя файл меню в Telegram; не называй slash-команды — только факты по вопросу. "
    "Не предлагай смотреть меню на сайте ради списка блюд или напитков — для этого у гостя офлайн-файл и этот текст."
)

# Когда в том же апдейте отправлен файл меню — модель не должна дублировать интерфейсные подсказки.
MENU_FILE_JUST_SENT_SYSTEM = (
    "Бот уже отправил гостю документ с меню и при необходимости отдельное сообщение про предзаказ администратору. "
    "В твоём ответе: не направляй в первую очередь на страницу меню на сайте; не дублируй факт, что бот уже прислал документ; не называй slash-команды; "
    "не повторяй телефон, предзаказ и фразы про «цены в файле». Отвечай только по сути вопроса, без рекламы списка блюд. "
    "Ссылку на сайт — только коротко, если уместно про актуальность."
)


def append_tickets_preorder_footer(text: str, max_len: int = 4096) -> str:
    """Добавляет напоминание о предзаказе и телефон к ответам про мероприятия/билеты."""
    if _TICKETS_PREORDER_MARKER in text:
        return text[:max_len] + ("…" if len(text) > max_len else "")
    foot = (
        "\n\n—\n"
        f"{_TICKETS_PREORDER_MARKER} оформляем по телефону {PHONE_DISPLAY_FULL}."
    )
    if len(text) + len(foot) <= max_len:
        return text + foot
    room = max_len - len(foot) - 1
    if room < 1:
        return foot[:max_len]
    trimmed = text[:room] + ("…" if len(text) > room else "")
    out = trimmed + foot
    return out[:max_len]


def send_menu_file_to_chat(chat_id: int) -> None:
    """Отправляет локальный файл меню или подсказку, если файла нет."""
    path = _menu_file_path()
    caption = f"Меню для предзаказа к программе. Заказ по телефону: {PHONE_DISPLAY_FULL}."
    if path is None:
        bot.send_message(
            chat_id,
            f"Файл меню ещё не загружен. Предзаказ к программе — по телефону {PHONE_DISPLAY_FULL} "
            f"(после 16:00, как на сайте).\n\n"
            "Администратору: положите рядом с bot2.py файл menu.docx, menu.pdf или menu.txt "
            "или укажите путь в .env: MENU_FILE=C:\\путь\\к\\menu.docx",
        )
        return
    try:
        with open(path, "rb") as doc:
            bot.send_document(chat_id, doc, visible_file_name=path.name, caption=caption)
    except Exception as e:
        bot.send_message(
            chat_id,
            f"Не удалось отправить файл меню ({path.name}): {e}\nТелефон для предзаказа: {PHONE_DISPLAY_FULL}",
        )


def send_menu_bundle_to_chat(chat_id: int) -> None:
    """Как команда /menu: файл меню + подсказка про /predzakaz при настройке."""
    send_menu_file_to_chat(chat_id)
    if _preorder_admin_chat_id():
        bot.send_message(
            chat_id,
            "Чтобы передать предзаказ администратору прямо из чата, напишите /predzakaz и опишите заказ.",
        )


def _format_preorder_for_admin(m, order_text: str) -> str:
    u = m.from_user
    uname = f"@{u.username}" if u and u.username else "(username не указан)"
    name = _user_display_name(u) if u else "?"
    uid = u.id if u else "?"
    chat_id = m.chat.id
    chat_type = m.chat.type if m.chat else "?"
    return (
        "Новый предзаказ (бот «Бродячая собака»)\n\n"
        f"Гость: {name} {uname}\n"
        f"User ID: {uid}\n"
        f"Чат: {chat_id} ({chat_type})\n\n"
        f"Текст предзаказа:\n{order_text}"
    )


def _notify_admin_preorder(m, order_text: str) -> bool:
    admin_chat = _preorder_admin_chat_id()
    if not admin_chat:
        return False
    body = _format_preorder_for_admin(m, order_text)
    try:
        kwargs: dict = {}
        thread = os.environ.get("PREORDER_ADMIN_MESSAGE_THREAD_ID", "").strip()
        if thread.isdigit():
            kwargs["message_thread_id"] = int(thread)
        bot.send_message(admin_chat, body, **kwargs)
        return True
    except Exception as e:
        print(f"[ERROR] Предзаказ: не удалось отправить администратору в чат {admin_chat}: {e}")
        return False


# Ticketscloud: возврат и сервисы для зрителей (билеты оформляются через их экосистему)
TICKETSCLOUD_FOR_VIEWERS_URL = "https://ticketscloud.com/for-viewers"
SOBAKA_HOME_URL = "https://sobaka.su/"

# Страницы sobaka.su для общего контекста Mistral (кэш). Страницу /menyu не подключаем — список блюд и цен только из файлов menu/sostav.
# Дополнительно: SOBAKA_CONTEXT_URLS в .env через запятую (URL с путём /menyu игнорируются).
DEFAULT_SOBAKA_CONTEXT_URLS: list[str] = [
    "https://sobaka.su/",
]

_SOBAKA_CONTACTS_SNIPPET_CACHE: tuple[str, float] | None = None
_SOBAKA_CONTACTS_TTL_SEC = 45 * 60

_SOBAKA_SITE_BUNDLE_CACHE: tuple[str, float] | None = None
_SOBAKA_SITE_BUNDLE_TTL_SEC = 30 * 60

# История сообщений user/assistant по chat_id (для многоходового диалога; это не дообучение модели).
_CHAT_HISTORY: dict[int, list[dict[str, str]]] = {}
_HISTORY_PATH = Path(__file__).resolve().parent / ".chat_history.json"

# Сколько карточек мероприятий подгружать со страниц /item/...
MAX_ITEM_PAGES = 8
MAX_CHARS_AFISHA_SOURCE = 14000
ITEM_URL_RE = re.compile(
    r"https?://(?:www\.)?sobaka\.su/item/[^\s\"'<>#]+",
    re.IGNORECASE,
)
# Время в анонсах K2: «20:00» или «20.00»
_TIME_IN_ANNOUNCE_RE = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\.\d{2}\b")

GENERAL_SYSTEM_PROMPT = (
    "Ты бот-помощник кабаре-кафе «Бродячая собака» (сайт https://sobaka.su/). "
    "Отвечай только по темам, связанным с этим заведением: мероприятия, репертуар, "
    "покупка билетов, контакты, адрес (Новосибирск, ул. Каменская, 32). "
    "Отвечай по-русски, кратко и по делу. Можно уместно пошутить. "
    "Не начинай ответ с вводных вроде «Вот меню», «Вот основное меню», «Если нужен конкретный раздел — напишите» — сразу по сути, без мета-описания того, что ты сейчас перечислишь. "
    "Если вопрос не про «Бродячую собаку» или https://sobaka.su/ — вежливо скажи, что помогаешь "
    "только по этому заведению и предложи зайти на https://sobaka.su/ или спросить про мероприятия/билеты. "
    "Если просят описать конкретное мероприятие без текста с сайта — не выдумывай детали; "
    "посоветуй открыть афишу на sobaka.su или выбрать дату через /event. "
    "Про наличие билетов и свободные места: у тебя нет доступа к актуальному сайту в реальном времени; "
    "посоветуй открыть афишу на https://sobaka.su/ и оформить билеты там или воспользоваться командой "
    "бота для выбора даты мероприятий. "
    "Если в промпте два офлайн-блока: «ФАЙЛ МЕНЮ» — только названия и цены; «ФАЙЛ СОСТАВА» — состав, ингредиенты и граммовки. "
    "Цены и названия позиций бери только из блока меню; состав и граммы — только из блока состава, если он есть. "
    "Не дополняй из памяти, Википедии или сайта позиций, которых нет в этих блоках. "
    "Если вопрос про меню, а файлов нет в промпте — не придумывай меню; скажи, что данные боту недоступны, и предложи /menu или администратора. "
    "Про еду и меню: не устраивай рекламу — не вываливай длинным списком все категории "
    "(закуски, салаты, бургеры, десерты, напитки и т.д.), если гость этого явно не просил. "
    "Не заканчивай ответ шаблоном вроде «цены и детали в присланном файле» плюс телефон и предзаказ. "
    "Не комментируй наличие меню в переписке и не отсылай к командам бота ради получения файла. "
    "Телефон +7 (383) 218-80-70 и /predzakaz упоминай только если вопрос про предзаказ к программе, как заказать блюда к шоу или связаться по заказу. "
    "Не пересказывай полное меню в ответе — только то, о чём спросили. "
    "Страницу меню на sobaka.su и любые онлайн-списки блюд для ответов не используй — только текст из файлов меню и состава в промпте (как по /menu). "
    "Возврат билетов Ticketscloud: раздел для зрителей https://ticketscloud.com/for-viewers; "
    "заявку на возврат оформляют в личном кабинете зрителя (customer.ticketscloud.com), правила — по ФЗ-193. "
    "Меню и цены для ответов — из офлайн-файла меню (как по /menu); состав и граммовки — из отдельного файла sostav.docx; сайт sobaka.su при таких вопросах не используется. "
    "Вопросы про телефон, во сколько звонить, бронь/заказ мест по телефону, часы работы зала, администраторов на телефоне — "
    "не направляй в группу ВК; дай номер +7 (383) 218-80-70, имена администраторов на линии: Данил и Анастасия, "
    "и опирайся на факты с https://sobaka.su/ (часы и пометку «после 16-00» для звонков по брони), если они переданы "
    "в этом запросе отдельным блоком. "
    "Может передаваться краткая история последних реплик этого чата — учитывай её для связных ответов; "
    "это не дообучение модели, только контекст в рамках лимита сообщений. "
    "Если спрашивают о проведении своего мероприятия или концерта в кафе, аренде площадки, корпоративе как выступление "
    "на сцене «Собаки» — не придумывай условия и цены; вежливо направь писать в личные сообщения сообщества ВКонтакте: "
    f"{VK_ORG_MESSAGES_URL} (там согласуют организационные вопросы)."
)

PHONE_HOURS_EXTRA_SYSTEM = (
    "Сейчас вопрос про звонки или контакты. Жёстко: не предлагай написать в ВК. "
    f"Обязательно укажи телефон {PHONE_DISPLAY_FULL} (допустимо дублировать {PHONE_DISPLAY_SHORT}). "
    f"Назови администраторов, которые ведут телефон: {ADMIN_PHONE_NAMES}. "
    "Часы работы зала и условия звонка по брони возьми только из приведённого ниже фрагмента с главной sobaka.su; "
    "не выдумывай другое время. Если во фрагменте два варианта часов (блок «Контакты» и подвал), кратко укажи оба или скажи, что в подвале указано иначе."
)


def _normalize_item_url(url: str) -> str:
    p = urlsplit(url.strip())
    if not p.netloc or "sobaka.su" not in p.netloc.lower():
        return ""
    path = p.path.rstrip("/") or "/"
    return urlunsplit(("https", "sobaka.su", path, "", ""))


def _http_get(url: str) -> str:
    r = requests.get(url, timeout=20, headers=HTTP_HEADERS)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def fetch_sobaka_contacts_snippet() -> str:
    """
    Телефон и часы с главной sobaka.su (кэш).
    Страница смешанная: строка с «после 16-00» читается из cp1251, интервалы 17.00 — из utf-8.
    """
    global _SOBAKA_CONTACTS_SNIPPET_CACHE
    now = time.monotonic()
    if _SOBAKA_CONTACTS_SNIPPET_CACHE and (now - _SOBAKA_CONTACTS_SNIPPET_CACHE[1]) < _SOBAKA_CONTACTS_TTL_SEC:
        return _SOBAKA_CONTACTS_SNIPPET_CACHE[0]

    fallback = (
        f"Телефон: {PHONE_DISPLAY_FULL}. Заказ мест на сайте: звонить после 16:00. "
        "Часы работы в блоке «Контакты»: Вс–Чт 17:00–23:00, Пт–Сб 17:00–01:00. "
        "В подвале страницы также указано: 17:00–03:00."
    )
    try:
        r = requests.get(SOBAKA_HOME_URL, timeout=20, headers=HTTP_HEADERS)
        r.raise_for_status()
        raw = r.content
        cp = raw.decode("cp1251", errors="replace")
        ut = raw.decode("utf-8", errors="replace")
        parts: list[str] = []

        pm = re.search(r"218-80-70\s*\([^)]+\)", cp)
        if pm:
            parts.append(f"Как на сайте (блок заказа мест): {pm.group(0).strip()}")
        else:
            parts.append(f"Телефон: {PHONE_DISPLAY_FULL} — звонить по брони после 16:00 (как на sobaka.su).")

        if "17.00-23.00" in ut and "17.00-01.00" in ut:
            parts.append("Часы работы зала по главной sobaka.su: Вс–Чт 17:00–23:00; Пт–Сб 17:00–01:00.")
        else:
            parts.append("Часы работы зала — смотри блок контактов на sobaka.su (на странице не подтвердились автоматически).")

        if "17:00-03:00" in ut:
            parts.append("В подвале той же страницы указано: Часы работы 17:00–03:00.")

        snippet = "\n".join(parts)
        _SOBAKA_CONTACTS_SNIPPET_CACHE = (snippet, now)
        return snippet
    except Exception:
        _SOBAKA_CONTACTS_SNIPPET_CACHE = (fallback, now)
        return fallback


def _user_asks_phone_or_hours(text: str) -> bool:
    t = text.lower().replace("ё", "е")
    keys = (
        "звон",
        "позвон",
        "телефон",
        "номер",
        "во сколько",
        "когда можно звон",
        "бронь",
        "забронировать",
        "заказать мест",
        "заказ мест",
        "администра",
        "дозвон",
        "режим работ",
        "часы работ",
        "работаете",
        "открыты",
        "кто снимает",
        "трубк",
        "контакт",
    )
    if "номер" in t and "билет" in t:
        return False
    return any(k in t for k in keys)


def _is_sobaka_online_menu_url(url: str) -> bool:
    """Страница меню на сайте — в промпт Mistral не попадает (меню только из файлов)."""
    try:
        p = urlsplit(url.strip())
        if "sobaka.su" not in (p.netloc or "").lower():
            return False
        path = (p.path or "").rstrip("/").lower()
        return path == "/menyu" or path.endswith("/menyu")
    except Exception:
        return False


def _sobaka_context_url_list() -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for u in DEFAULT_SOBAKA_CONTEXT_URLS:
        if _is_sobaka_online_menu_url(u):
            continue
        if u not in seen:
            seen.add(u)
            urls.append(u)
    extra = os.environ.get("SOBAKA_CONTEXT_URLS", "").strip()
    if extra:
        for part in extra.split(","):
            u = part.strip()
            if u.startswith("http") and "sobaka.su" in u and u not in seen:
                if _is_sobaka_online_menu_url(u):
                    continue
                seen.add(u)
                urls.append(u)
    return urls


def fetch_sobaka_site_context_bundle() -> str:
    """
    Тексты с выбранных страниц sobaka.su (главная и др., без /menyu) для опоры модели.
    Кэш ~30 мин. Отключить: MISTRAL_SITE_CONTEXT=0. Объём: SOBAKA_CONTEXT_MAX_CHARS (по умолчанию 12000).
    """
    if os.environ.get("MISTRAL_SITE_CONTEXT", "1").strip().lower() in ("0", "false", "no", "off"):
        return ""

    global _SOBAKA_SITE_BUNDLE_CACHE
    now = time.monotonic()
    if _SOBAKA_SITE_BUNDLE_CACHE and (now - _SOBAKA_SITE_BUNDLE_CACHE[1]) < _SOBAKA_SITE_BUNDLE_TTL_SEC:
        return _SOBAKA_SITE_BUNDLE_CACHE[0]

    try:
        lim = int(os.environ.get("SOBAKA_CONTEXT_MAX_CHARS", "12000") or "12000")
    except ValueError:
        lim = 12000
    lim = max(2000, min(24000, lim))

    chunks: list[str] = []
    total = 0
    for url in _sobaka_context_url_list():
        try:
            html = _http_get(url)
            text = extract_listing_fallback_text(html)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            block = f"\n--- {url} ---\n{text}\n"
            if total + len(block) > lim:
                remain = lim - total
                if remain > 400:
                    chunks.append(block[:remain] + "\n…(обрезано по лимиту SOBAKA_CONTEXT_MAX_CHARS)\n")
                break
            chunks.append(block)
            total += len(block)
        except Exception as exc:
            chunks.append(f"\n--- {url} ---\n(не удалось загрузить: {exc})\n")

    result = "".join(chunks).strip()
    _SOBAKA_SITE_BUNDLE_CACHE = (result, now)
    return result


def _history_max_messages() -> int:
    """Пар user+assistant в истории (×2 сообщения в API). 0 — без истории."""
    raw = os.environ.get("MISTRAL_HISTORY_TURNS", "8").strip()
    try:
        n = int(raw)
    except ValueError:
        n = 8
    return max(0, min(30, n))


def _history_trim_one(text: str, max_len: int = 2800) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _chat_history_load_disk() -> None:
    global _CHAT_HISTORY
    if os.environ.get("MISTRAL_HISTORY_PERSIST", "").strip().lower() not in ("1", "true", "yes"):
        return
    if not _HISTORY_PATH.is_file():
        return
    try:
        raw = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
        out: dict[int, list[dict[str, str]]] = {}
        for k, v in raw.items():
            try:
                cid = int(k)
            except ValueError:
                continue
            if isinstance(v, list):
                out[cid] = [x for x in v if isinstance(x, dict) and x.get("role") in ("user", "assistant")]
        _CHAT_HISTORY = out
        print(f"[INFO] Загружена история чатов Mistral из {_HISTORY_PATH.name} ({len(out)} чатов).")
    except Exception as e:
        print(f"[WARN] Не удалось загрузить {_HISTORY_PATH.name}: {e}")


def _chat_history_save_disk() -> None:
    if os.environ.get("MISTRAL_HISTORY_PERSIST", "").strip().lower() not in ("1", "true", "yes"):
        return
    try:
        serial = {str(k): v for k, v in _CHAT_HISTORY.items()}
        _HISTORY_PATH.write_text(json.dumps(serial, ensure_ascii=False, indent=0), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Не удалось сохранить историю чатов: {e}")


def _chat_history_get(chat_id: int) -> list[dict[str, str]]:
    return list(_CHAT_HISTORY.get(chat_id, []))


def _chat_history_clear(chat_id: int) -> None:
    _CHAT_HISTORY.pop(chat_id, None)
    _chat_history_save_disk()


def _chat_history_append(chat_id: int, user_text: str, assistant_text: str) -> None:
    max_turns = _history_max_messages()
    if max_turns == 0:
        return
    hist = _CHAT_HISTORY.setdefault(chat_id, [])
    hist.append({"role": "user", "content": _history_trim_one(user_text)})
    hist.append({"role": "assistant", "content": _history_trim_one(assistant_text)})
    max_msgs = max_turns * 2
    if len(hist) > max_msgs:
        del hist[: len(hist) - max_msgs]
    _chat_history_save_disk()


def extract_item_urls_from_listing(html: str, base: str) -> list[str]:
    """
    Со страницы списка на дату собирает ссылки на карточки мероприятий /item/...
    Сначала узкие контейнеры афиши (K2), чтобы не брать ссылки из общего сайдбара.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for sel in ("footer", "header", "aside", ".sidebar", "#sidebar"):
        for el in soup.select(sel):
            el.decompose()

    def collect_from(root) -> list[str]:
        seen: set[str] = set()
        found: list[str] = []
        for a in root.select('a[href*="/item/"]'):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            full = _normalize_item_url(urljoin(base, href.split("#")[0]))
            if not full or full in seen:
                continue
            seen.add(full)
            found.append(full)
            if len(found) >= MAX_ITEM_PAGES:
                break
        return found

    for sel in ("#k2Container", ".itemList", ".itemContainer", ".component"):
        node = soup.select_one(sel)
        if node:
            got = collect_from(node)
            if got:
                return got

    main = soup.select_one("main") or soup.select_one("#content")
    if main:
        got = collect_from(main)
        if got:
            return got

    got = collect_from(soup)
    if got:
        return got

    seen: set[str] = set()
    out: list[str] = []
    for m in ITEM_URL_RE.finditer(html):
        full = _normalize_item_url(m.group(0))
        if full and full not in seen:
            seen.add(full)
            out.append(full)
        if len(out) >= MAX_ITEM_PAGES:
            break
    return out


def _extract_k2_item_datetime_banner(soup: BeautifulSoup) -> str:
    """
    Дата и время начала из шапки карточки K2 (часто вне .itemFullText, поэтому раньше терялись).
    Пример: «Вс 26 Апрель 20:00».
    """
    for sel in (".itemDateCreated", ".catItemDate", ".itemDate", ".createdate"):
        el = soup.select_one(sel)
        if el:
            t = el.get_text(" ", strip=True)
            if t and _TIME_IN_ANNOUNCE_RE.search(t):
                return t
    ih = soup.select_one(".itemHeader")
    if ih:
        t = ih.get_text(" ", strip=True)
        if t and _TIME_IN_ANNOUNCE_RE.search(t) and len(t) < 500:
            return t.strip()
    h1 = soup.select_one("h1.itemTitle, .itemTitle, #k2Container h1")
    if h1:
        sib = h1.find_previous_sibling()
        while sib is not None:
            raw = sib.get_text(" ", strip=True)
            if raw and _TIME_IN_ANNOUNCE_RE.search(raw) and len(raw) < 500:
                return raw.strip()
            sib = sib.find_previous_sibling()
    return ""


def _listing_datetime_line_for_item_url(listing_soup: BeautifulSoup, base: str, item_url: str) -> str:
    """Строка со временем из блока списка на дату, рядом со ссылкой на то же /item/...."""
    path_norm = urlsplit(item_url).path.rstrip("/")
    if "/item/" not in path_norm:
        return ""
    roots: list = []
    for sel in ("#k2Container", ".itemList", ".component", "main"):
        n = listing_soup.select_one(sel)
        if n and n not in roots:
            roots.append(n)
    if not roots:
        roots = [listing_soup]
    for root in roots:
        for a in root.select('a[href*="/item/"]'):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            full = _normalize_item_url(urljoin(base, href.split("#")[0]))
            if not full or urlsplit(full).path.rstrip("/") != path_norm:
                continue
            cur = a
            for _ in range(6):
                if cur is None:
                    break
                blob = cur.get_text(" ", strip=True)
                if blob and _TIME_IN_ANNOUNCE_RE.search(blob):
                    if len(blob) <= 320:
                        return blob.strip()
                    m = _TIME_IN_ANNOUNCE_RE.search(blob)
                    if m:
                        i0 = max(0, m.start() - 80)
                        i1 = min(len(blob), m.end() + 60)
                        return blob[i0:i1].strip()
                cur = cur.parent
            break
    return ""


def extract_item_page_body_text(html: str) -> str:
    """Текст описания программы со страницы одного мероприятия (K2/Joomla)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for sel in (".itemRelated", ".itemVideoBlock", "footer", "aside", ".sidebar"):
        for el in soup.select(sel):
            el.decompose()

    title = ""
    for sel in (".itemTitle", "h1.itemTitle", "#k2Container h1", "header h1", "h1"):
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            title = el.get_text(" ", strip=True)
            break

    body = None
    for sel in (
        ".itemFullText",
        ".itemBody",
        ".catItemBody",
        ".userItemBody",
        ".genericItemBody",
        "#k2Container .item",
    ):
        body = soup.select_one(sel)
        if body and len(body.get_text(strip=True)) > 80:
            break

    if body is None:
        body = soup.select_one("#k2Container") or soup.select_one("main article") or soup.select_one("main")

    if body is None:
        text = soup.get_text("\n", strip=True)
    else:
        text = body.get_text("\n", strip=True)

    lines: list[str] = []
    prev = None
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s == prev and len(s) < 40 and "купить" in s.lower():
            continue
        lines.append(s)
        prev = s
    text = "\n".join(lines)
    core = text
    if title and title.lower() not in core[:300].lower():
        core = f"{title}\n\n{core}"
    dt_banner = _extract_k2_item_datetime_banner(soup)
    if dt_banner and dt_banner.lower() not in core[:900].lower():
        core = f"{dt_banner}\n\n{core}"
    return core


def extract_listing_fallback_text(html: str) -> str:
    """Если ссылок на /item/ нет — забираем максимум текста со страницы даты."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    root = soup.select_one("#k2Container, main, #content") or soup
    text = root.get_text("\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def _mistral_chat_messages(
    system_prompt: str,
    message_list: list[dict],
    *,
    menu_low_temperature: float | None = None,
) -> str:
    if not MISTRAL_API_KEY or not client:
        return (
            "❌ Не указан MISTRAL_API_KEY.\n"
            "Задайте переменную окружения MISTRAL_API_KEY или добавьте её в .env рядом с bot2.py "
            "(ключ: https://console.mistral.ai/)."
        )
    if not message_list:
        return "❌ Пустой список сообщений."
    try:
        last_user = next(
            (m.get("content", "") for m in reversed(message_list) if m.get("role") == "user"),
            "",
        )
        print(f"[DEBUG] Mistral (история {len(message_list)} сообщ.): {last_user[:100]}...")
        create_kw: dict = {
            "model": "mistral-small-latest",
            "messages": [{"role": "system", "content": system_prompt}, *message_list],
        }
        if menu_low_temperature is not None:
            create_kw["temperature"] = float(menu_low_temperature)
        completion = client.chat.completions.create(**create_kw)
        answer = (completion.choices[0].message.content or "").strip()
        return answer
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Ошибка при обращении к Mistral API: {error_msg}")
        low = error_msg.lower()
        hint = ""
        if any(x in low for x in ("connection", "connect", "timeout", "network", "unreachable", "refused")):
            hint = (
                "\n\nСеть до api.mistral.ai недоступна или обрыв. Бот подставляет тот же прокси, что и для Telegram "
                "(TELEGRAM_PROXY), если не задан MISTRAL_PROXY и не отключено MISTRAL_INHERIT_TELEGRAM_PROXY=0. "
                "Попробуйте увеличить MISTRAL_TIMEOUT_SEC=180 или другой VPN/режим TUN."
            )
        return f"Ошибка при обращении к Mistral API: {error_msg}{hint}"


def _mistral_chat(system_prompt: str, user_text: str) -> str:
    return _mistral_chat_messages(system_prompt, [{"role": "user", "content": user_text}])


def ask_gpt(chat_id: int, user_text: str, *, menu_delivered: bool = False) -> str:
    """
    Общий диалог с Mistral: контекст страниц sobaka.su (кроме вопросов про меню), опционально история чата.
    История — не дообучение модели; лимит пар реплик: MISTRAL_HISTORY_TURNS (0 = выкл).
    Параметр menu_delivered: в этом же апдейте уже отправлены файл меню и подсказка про предзаказ (как /menu).
    """
    system = GENERAL_SYSTEM_PROMPT
    user_trim = _history_trim_one(user_text)
    menu_topic = _user_asks_dish_composition(user_text)
    offline_menu_txt = load_offline_menu_text_for_mistral() if menu_topic else ""
    sostav_txt = load_sostav_composition_text_for_mistral() if menu_topic else ""

    # Списки категорий из меню — только строки из файла; Mistral здесь подмешивает вымышленные позиции и цены.
    if (
        menu_topic
        and offline_menu_txt
        and "[РАЗДЕЛ]" in offline_menu_txt
        and _user_asks_menu_bulk_list(user_trim)
        and not _user_asks_phone_or_hours(user_text)
    ):
        verbatim_lines = _extract_menu_bullets_best_effort(offline_menu_txt, user_trim)
        if verbatim_lines:
            answer = "\n".join(f"• {line}" for line in verbatim_lines)
            max_turns = _history_max_messages()
            if max_turns > 0:
                _chat_history_append(chat_id, user_text, answer)
            return answer
        answer = MENU_VERBATIM_NO_MATCH
        max_turns = _history_max_messages()
        if max_turns > 0:
            _chat_history_append(chat_id, user_text, answer)
        return answer

    bundle = fetch_sobaka_site_context_bundle()
    if bundle and not menu_topic:
        system = (
            f"{system}\n\n"
            "=== ИСТОЧНИК С САЙТА sobaka.su (фрагменты страниц, кэш ~30 мин; актуальность уточняй по ссылкам) ===\n"
            f"{bundle}"
        )
    if menu_delivered:
        system = f"{system}\n\n{MENU_FILE_JUST_SENT_SYSTEM}"
    if _user_asks_phone_or_hours(user_text):
        site_block = fetch_sobaka_contacts_snippet()
        system = (
            f"{system}\n\n{PHONE_HOURS_EXTRA_SYSTEM}\n\n"
            f"=== Фрагмент с {SOBAKA_HOME_URL} (телефон и часы) ===\n{site_block}"
        )

    if menu_topic:
        blocks: list[str] = []
        focus_excerpt = _build_menu_focus_excerpt_by_headings(offline_menu_txt, sostav_txt, user_trim)
        if focus_excerpt:
            blocks.append(
                "=== ВЫРЕЗКА РАЗДЕЛА ПО КЛЮЧЕВЫМ СЛОВАМ ИЗ ВОПРОСА ===\n"
                "Ниже только фрагменты, где в строке «[РАЗДЕЛ] …» заголовок совпал со словами из вопроса гостя "
                "(как разделы «Заголовок 1» в Word: всё до следующего такого заголовка). "
                "Если вопрос про эту категорию — отвечай по вырезке: перечисли позиции и цены дословно, без добавлений из памяти и без других разделов.\n\n"
                f"{focus_excerpt}"
            )
        if offline_menu_txt:
            blocks.append(
                "=== ФАЙЛ МЕНЮ (названия блюд и цены; тот же документ, что отправляется гостю по /menu) ===\n"
                "Служебно: сайт sobaka.su в этот запрос не включён — цены и названия позиций только из этого блока.\n\n"
                f"{offline_menu_txt}"
            )
        if sostav_txt:
            blocks.append(
                "=== ФАЙЛ СОСТАВА sostav.docx (ингредиенты, состав блюд, граммовки) ===\n"
                "Состав и веса только из этого блока; цены — только из блока «ФАЙЛ МЕНЮ» выше.\n\n"
                f"{sostav_txt}"
            )
        if blocks:
            extra_menu = (
                f"{MENU_FILE_AUTHORITY_PREFIX}\n\n{MENU_COMPOSITION_EXTRA_SYSTEM}"
                if offline_menu_txt
                else MENU_COMPOSITION_EXTRA_SYSTEM
            )
            system = f"{system}\n\n{extra_menu}\n\n" + "\n\n".join(blocks)
        else:
            system = (
                f"{system}\n\n{MENU_COMPOSITION_EXTRA_SYSTEM}\n\n"
                "Нет данных из файлов: положите рядом с bot2.py menu.docx или menu.txt (названия и цены, как для /menu; MENU_FILE=…) "
                "и sostav.docx (состав и граммовки; при необходимости SOSTAV_DOCX_FILE=…). "
                "Для .docx нужен python-docx. PDF меню бот не читает. "
                "Не выдумывай меню и не опирайся на сайт — коротко скажи, что файлы боту не загружены."
            )

    max_turns = _history_max_messages()
    if menu_topic and offline_menu_txt and max_turns > 0:
        system = f"{system}\n\n{MENU_HISTORY_NOT_AUTHORITY}"
    mt_temp = 0.0 if menu_topic else None
    if max_turns == 0:
        return _mistral_chat_messages(
            system,
            [{"role": "user", "content": user_trim}],
            menu_low_temperature=mt_temp,
        )

    prior = _chat_history_get(chat_id)
    msg_list = prior + [{"role": "user", "content": user_trim}]
    answer = _mistral_chat_messages(system, msg_list, menu_low_temperature=mt_temp)
    err = answer.startswith("❌") or answer.startswith("Ошибка при обращении к Mistral API")
    if not err:
        _chat_history_append(chat_id, user_text, answer)
    return answer


_chat_history_load_disk()
if _history_max_messages() > 0:
    print(
        f"[INFO] Mistral: в запрос подмешивается история чата (до {_history_max_messages()} пар реплик; "
        "MISTRAL_HISTORY_TURNS=0 — выкл). /forget — очистить. Это не обучение модели."
    )
else:
    print("[INFO] Mistral: история чата выключена (MISTRAL_HISTORY_TURNS=0).")

if _DocxDocument is None:
    print(
        "[WARN] Пакет python-docx не установлен — меню в формате .docx для Mistral недоступно. Установите: pip install python-docx"
    )
_menu_probe = load_offline_menu_text_for_mistral()
_menu_p = _menu_file_path()
if _menu_probe:
    print(
        f"[INFO] Меню (названия и цены) для Mistral: ~{len(_menu_probe)} символов "
        f"({_menu_p.name if _menu_p else 'MENU_FILE'} — как для /menu)."
    )
elif _menu_p and _menu_p.suffix.lower() not in (".docx", ".txt"):
    print(
        f"[INFO] Меню для гостей: {_menu_p.name} — для ответов бота нужен .docx или .txt (PDF не читается)."
    )
elif _menu_p and _menu_p.suffix.lower() == ".docx" and _DocxDocument is None:
    print(f"[INFO] Найден {_menu_p.name}, но без python-docx текст для Mistral не извлечь — pip install python-docx.")
elif _menu_p is None:
    print(
        "[INFO] Файл меню не найден — положите menu.docx или menu.txt (или MENU_FILE=…) для названий и цен."
    )

_sostav_probe = load_sostav_composition_text_for_mistral()
_sostav_p = _sostav_docx_path()
if _sostav_probe:
    print(
        f"[INFO] Состав/граммовки для Mistral: ~{len(_sostav_probe)} символов ({_sostav_p.name if _sostav_p else 'sostav.docx'})."
    )
elif _DocxDocument is not None:
    print(
        "[INFO] Файл sostav.docx не найден — для ответов о составе и граммах положите sostav.docx рядом с bot2.py "
        "(или SOSTAV_DOCX_FILE=…). Должен отличаться от файла меню для гостей."
    )


def ask_gpt_afisha_grounded(source_bundle: str, date_url: str, human_date: str) -> str:
    """
    Краткий ответ строго по текстам со страниц сайта (переданы в source_bundle).
    """
    system_prompt = (
        "Ты помогаешь оформить ответ гостю о мероприятиях кабаре-кафе «Бродячая собака». "
        "Ниже в сообщении пользователя дан ИСТОЧНИК — фрагменты текстов, скопированные со страниц сайта sobaka.su "
        "(анонсы и описания программ с официальных страниц мероприятий).\n\n"
        "Жёсткие правила:\n"
        "— Используй только сведения из ИСТОЧНИКА. Не добавляй факты из памяти и не выдумывай: "
        "ни время, ни цены, ни состав артистов, ни жанр, если этого нет в ИСТОЧНИКЕ.\n"
        "— Время начала: если в ИСТОЧНИКЕ есть конкретное время (например 19:00, 20:00 или 20.00) — обязательно укажи его для каждого мероприятия в том же виде или как «в 20:00». "
        "Категорически запрещено заменять точное время размытыми словами «днём», «днем», «в течение дня», «днём будет», «вечером» без цифр, если в тексте источника есть часы:минуты.\n"
        "— Если времени начала в ИСТОЧНИКЕ нет — так и скажи кратко, не придумывай время.\n"
        "— Названия мероприятий и формулировки из анонса сохраняй максимально близко к оригиналу; "
        "допустим только сжатый пересказ без искажения смысла.\n"
        "— Если в ИСТОЧНИКЕ мало текста, честно скажи об этом и перечисли только то, что есть (например название и время, если они в тексте).\n"
        "— Не добавляй общие фразы про «неповторимую атмосферу» и т.п., если их нет в ИСТОЧНИКЕ.\n"
        "— Ответ по-русски, связно, без шуток, 3–12 предложений.\n"
        "— В конце обязательно отдельной строкой напиши: "
        f"Ссылка на афишу на эту дату: {date_url}"
    )
    user_block = (
        f"Дата (для справки): {human_date}.\n\n"
        "=== ИСТОЧНИК (только со страниц sobaka.su) ===\n"
        f"{source_bundle}"
    )
    return _mistral_chat(system_prompt, user_block)


def extract_date_from_text(text: str) -> datetime.date | None:
    """
    Пытается вытащить дату формата '19 февраля' из русского текста.
    Возвращает date с текущим годом или None.
    """
    text = text.lower()
    m = re.search(
        r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)",
        text,
    )
    if not m:
        return None
    day = int(m.group(1))
    month_word = m.group(2)
    month = MONTHS_GENITIVE.get(month_word)
    if not month:
        return None
    year = datetime.date.today().year
    try:
        return datetime.date(year, month, day)
    except ValueError:
        return None


def describe_events_for_date(date_obj: datetime.date) -> str:
    """
    Загружает страницу афиши на дату и страницы мероприятий /item/... с описаниями программ,
    затем формирует ответ, опираясь только на эти тексты.
    """
    base_url = "https://sobaka.su"
    date_url = f"{base_url}/itemlist/date/{date_obj.year}/{date_obj.month}/{date_obj.day}?catid[0]=1"
    human_date = date_obj.strftime("%d.%m.%Y")

    try:
        html = _http_get(date_url)
    except Exception as e:
        return append_tickets_preorder_footer(
            f"Не смог получить информацию о мероприятиях на {human_date}:\n{e}"
        )

    item_urls = extract_item_urls_from_listing(html, base_url)
    listing_soup = BeautifulSoup(html, "html.parser")
    chunks: list[str] = []
    total_len = 0

    for item_url in item_urls:
        try:
            item_html = _http_get(item_url)
            body = extract_item_page_body_text(item_html)
            list_line = _listing_datetime_line_for_item_url(listing_soup, base_url, item_url)
            if list_line and list_line.lower() not in body[:900].lower():
                body = f"{list_line}\n\n{body}"
        except Exception as exc:
            body = f"(не удалось загрузить страницу: {exc})"
        block = f"\n--- Страница мероприятия: {item_url} ---\n{body}\n"
        if total_len + len(block) > MAX_CHARS_AFISHA_SOURCE:
            remain = MAX_CHARS_AFISHA_SOURCE - total_len
            if remain > 200:
                chunks.append(block[:remain] + "\n…(обрезано по лимиту объёма)")
            break
        chunks.append(block)
        total_len += len(block)

    if not chunks:
        fallback = extract_listing_fallback_text(html)
        if len(fallback) < 80:
            return append_tickets_preorder_footer(
                f"На {human_date} на странице афиши не найдено текстовых анонсов. "
                f"Проверьте расписание на сайте:\n{date_url}"
            )
        source_bundle = (
            f"(Со страницы списка на дату, отдельные карточки /item/ не распознаны)\n{fallback}"
        )[:MAX_CHARS_AFISHA_SOURCE]
    else:
        source_bundle = "".join(chunks)[:MAX_CHARS_AFISHA_SOURCE]

    if not MISTRAL_API_KEY or not client:
        # Без API: отдаём сырой текст с сайта, без пересказа моделью
        tail = f"\n\nАфиша на дату: {date_url}"
        raw = source_bundle + tail
        return append_tickets_preorder_footer(raw[:4000] + ("…" if len(raw) > 4000 else ""))

    reply = ask_gpt_afisha_grounded(source_bundle, date_url, human_date)
    if len(reply) > 4000:
        reply = reply[:3990] + "…"
    return append_tickets_preorder_footer(reply)


def _send_link_message(chat_id: int, text: str, button_label: str, url: str) -> None:
    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton(button_label, url=url))
    bot.send_message(chat_id, text, reply_markup=markup)


@bot.message_handler(commands=["site"])
def site(message):
    _send_link_message(
        message.chat.id,
        "Сайт кабаре-кафе «Бродячая собака»:",
        "Открыть sobaka.su",
        "https://sobaka.su/",
    )


@bot.message_handler(commands=["vk"])
def vk(message):
    _send_link_message(
        message.chat.id,
        "Группа ВКонтакте:",
        "Открыть группу ВК",
        "https://vk.com/club4634260",
    )


@bot.message_handler(commands=["vk_org"])
def vk_org(message):
    _send_link_message(
        message.chat.id,
        "По вопросам проведения мероприятий или концертов в кафе, аренды площадки и подобных организационных тем "
        "напишите в личные сообщения сообщества ВК:",
        "Написать в сообщество ВК",
        VK_ORG_MESSAGES_URL,
    )


@bot.message_handler(commands=["refund"])
def refund_cmd(message):
    _send_link_message(
        message.chat.id,
        "Возврат и сервисы для зрителей Ticketscloud (в т.ч. автоматизированный возврат в личном кабинете):",
        "Открыть ticketscloud.com/for-viewers",
        TICKETSCLOUD_FOR_VIEWERS_URL,
    )


@bot.message_handler(commands=["forget"])
def forget_cmd(message):
    _chat_history_clear(message.chat.id)
    bot.send_message(
        message.chat.id,
        "Память диалога с нейросетью в этом чате очищена. "
        "(Это не влияет на саму модель Mistral — только на сохранённые здесь последние реплики.)",
    )


@bot.message_handler(commands=["phone"])
def phone(message):
    block = fetch_sobaka_contacts_snippet()
    bot.send_message(
        message.chat.id,
        f"Телефон: {PHONE_DISPLAY_FULL}\n\n{block}\n\nАдминистраторы на телефоне: {ADMIN_PHONE_NAMES}.",
    )


@bot.message_handler(commands=["whatsapp"])
def whatsapp(message):
    bot.send_message(message.chat.id, "+79930045717")


@bot.message_handler(commands=["start"])
def start(message):
    _PREORDER_PENDING_USERS.discard(message.from_user.id)
    bot.send_message(
        message.chat.id,
        f"Здравствуйте, {_user_display_name(message.from_user)}!",
    )


@bot.message_handler(commands=["help"])
def help_cmd(message):
    bot.send_message(
        message.chat.id,
        "Я бот кабаре-кафе «Бродячая собака» (sobaka.su).\n\n"
        "Команды:\n"
        "/site – ссылка на сайт\n"
        "/vk – группа ВК\n"
        "/vk_org – провести мероприятие или концерт в кафе (личные сообщения ВК)\n"
        "/phone – телефон для брони\n"
        "/whatsapp – WhatsApp\n"
        "/refund – возврат билетов Ticketscloud (для зрителей)\n"
        "/forget – забыть историю сообщений с нейросетью в этом чате\n"
        "/menu – файл меню для предзаказа к программе (если загружен администратором)\n"
        "/predzakaz – отправить текст предзаказа администратору в Telegram\n"
        "/event – выбрать дату и узнать мероприятия, ссылка на билеты\n\n"
        "Напишите «билеты» или «купить билеты» — откроется календарь выбора даты. "
        "Бот помнит последние реплики в чате для связных ответов Mistral (не обучение модели). "
        "Фрагмент главной sobaka.su для Mistral подгружается из кэша (без страницы меню на сайте). "
        "Вопросы о меню и ценах — только по файлу меню (как /menu); состав и граммовки — по sostav.docx. "
        "Отвечаю только по заведению и мероприятиям на sobaka.su.",
    )


def send_calendar(chat_id: int) -> None:
    """
    Отправляет в чат меню с календарём выбора даты (текущий месяц).
    После выбора даты пользователь получит описание мероприятий и ссылку на билеты.
    """
    today = datetime.date.today()
    year, month = today.year, today.month
    last_day = calendar.monthrange(year, month)[1]

    markup = types.InlineKeyboardMarkup()
    title = f"Выберите дату в {MONTHS_PREPOSITIONAL[month]} {year} года:"

    row: list[types.InlineKeyboardButton] = []
    for day in range(1, last_day + 1):
        date_obj = datetime.date(year, month, day)
        btn = types.InlineKeyboardButton(
            text=str(day),
            callback_data=f"event_date:{date_obj.isoformat()}",
        )
        row.append(btn)
        if len(row) == 7:
            markup.row(*row)
            row = []
    if row:
        markup.row(*row)

    bot.send_message(chat_id, title, reply_markup=markup)


@bot.message_handler(commands=["menu"])
def menu_cmd(message):
    """Отправляет загруженный файл меню (предзаказ к программе)."""
    send_menu_bundle_to_chat(message.chat.id)


@bot.message_handler(commands=["predzakaz", "preorder"])
def predzakaz_cmd(message):
    """
    Текст предзаказа уходит администратору в Telegram (нужен PREORDER_ADMIN_CHAT_ID в .env).
    Варианты: /predzakaz дата, номер стола, блюда, имя, телефон — или /predzakaz и следующим сообщением полный текст.
    """
    if not _preorder_admin_chat_id():
        bot.send_message(
            message.chat.id,
            "Отправка предзаказа администратору через бота не настроена. "
            f"Оформите по телефону {PHONE_DISPLAY_FULL} или откройте /menu.\n\n"
            "Владельцу бота: добавьте в .env строку PREORDER_ADMIN_CHAT_ID=ваш_числовой_id "
            "(узнать id: @userinfobot или аналог в Telegram).",
        )
        return

    raw = (message.text or "").strip()
    parts = raw.split(maxsplit=1)
    if len(parts) >= 2 and parts[1].strip():
        order = parts[1].strip()
        if _notify_admin_preorder(message, order):
            bot.send_message(message.chat.id, "Предзаказ отправлен администратору. Спасибо!")
        else:
            bot.send_message(
                message.chat.id,
                f"Не удалось доставить сообщение. Позвоните {PHONE_DISPLAY_FULL} или попробуйте позже.",
            )
        return

    _PREORDER_PENDING_USERS.add(message.from_user.id)
    bot.send_message(
        message.chat.id,
        "Напишите следующим сообщением предзаказ одним текстом: дата программы, номер стола, время, блюда и напитки, "
        f"имя и телефон для связи.\n\nИли сразу одной строкой, например:\n/predzakaz 15 марта, стол 3, время 19:00, 2 борща, Иван +7…",
    )


@bot.message_handler(commands=["event"])
def event_cmd(message):
    """Показывает календарь выбора даты — описание мероприятий и ссылка на билеты с sobaka.su."""
    send_calendar(message.chat.id)


@bot.callback_query_handler(func=lambda call: call.data.startswith("event_date:"))
def handle_event_date_callback(call):
    """
    Обработчик выбора даты из календаря /event.
    """
    try:
        date_str = call.data.split(":", 1)[1]
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except Exception:
        _safe_answer_callback_query(call.id, "Ошибка разбора даты")
        return

    _safe_answer_callback_query(call.id)

    human_date = f"{date_obj.day} {MONTHS_GENITIVE_NUM[date_obj.month]} {date_obj.year} года"
    bot.send_message(call.message.chat.id, f"Секунду, ищу мероприятия на {human_date}...")
    description = describe_events_for_date(date_obj)
    bot.send_message(call.message.chat.id, description)


@bot.message_handler(content_types=["photo"])
def get_photo(message):
    bot.reply_to(
        message,
        "Спасибо за фото. Вопросы по заведению напишите текстом или нажмите /help.",
    )


def _user_awaits_preorder_reply(m) -> bool:
    if not m.text or not m.from_user:
        return False
    if m.text.lstrip().startswith("/"):
        return False
    return m.from_user.id in _PREORDER_PENDING_USERS


@bot.message_handler(func=_user_awaits_preorder_reply)
def preorder_capture(message):
    """Второй шаг после /predzakaz без текста — принимаем текст заказа и шлём администратору."""
    order_text = message.text.strip()
    if len(order_text) < 5:
        bot.send_message(message.chat.id, "Опишите заказ чуть подробнее (минимум несколько слов).")
        return

    _PREORDER_PENDING_USERS.discard(message.from_user.id)

    if not _preorder_admin_chat_id():
        bot.send_message(
            message.chat.id,
            f"Сервис предзаказа отключён. Звоните {PHONE_DISPLAY_FULL}.",
        )
        return

    if _notify_admin_preorder(message, order_text):
        bot.send_message(
            message.chat.id,
            "Спасибо! Предзаказ передан администратору. При необходимости с вами свяжутся.",
        )
    else:
        _PREORDER_PENDING_USERS.add(message.from_user.id)
        bot.send_message(
            message.chat.id,
            f"Не удалось отправить. Попробуйте ещё раз или позвоните {PHONE_DISPLAY_FULL}.",
        )


@bot.message_handler()
def info(message):
    if not message.text:
        return

    text = message.text.lower()

    # Запрос о билетах/мероприятиях: если в тексте есть дата — сразу показываем по ней; иначе — календарь
    refund_phrases = (
        "возврат билет",
        "вернуть билет",
        "вернуть билеты",
        "оформить возврат",
    )
    if any(p in text.replace("ё", "е") for p in refund_phrases):
        refund_cmd(message)
        return

    preorder_phrases = (
        "предзаказ",
        "пред заказ",
        "пред-заказ",
        "меню к программе",
        "меню на программу",
        "заказать к программе",
        "заказ к программе",
        "еда на программу",
        "блюда к программе",
        "напитки к программе",
        "хочу предзаказ",
        "сделать предзаказ",
        "отправь меню",
        "пришли меню",
        "скинь меню",
    )
    if any(p in text.replace("ё", "е") for p in preorder_phrases):
        send_menu_file_to_chat(message.chat.id)
        if _preorder_admin_chat_id():
            bot.send_message(
                message.chat.id,
                "Передать заказ администратору из бота: /predzakaz (опишите дату программы, состав, контакт).",
            )
        return

    ticket_keywords = (
        "билет",
        "купить",
        "бронь",
        "забронировать",
        "мероприят",
        "афиша",
        "репертуар",
        "что идет",
        "что идёт",
        "когда",
    )
    if any(kw in text for kw in ticket_keywords):
        date_obj = extract_date_from_text(text)
        if date_obj:
            human_date = f"{date_obj.day} {MONTHS_GENITIVE_NUM[date_obj.month]} {date_obj.year} года"
            bot.send_message(
                message.chat.id,
                f"Ищу мероприятия на {human_date} на sobaka.su...",
            )
            description = describe_events_for_date(date_obj)
            bot.send_message(message.chat.id, description)
            return
        # Даты нет — первым делом показываем календарь (как /event)
        bot.send_message(
            message.chat.id,
            "Выберите дату — покажу мероприятия и ссылку на покупку билетов на sobaka.su:",
        )
        send_calendar(message.chat.id)
        return

    # Только дата в сообщении (без явных слов про билеты) — всё равно ищем по дате
    if any(month in text for month in MONTHS_GENITIVE.keys()):
        date_obj = extract_date_from_text(text)
        if date_obj:
            human_date = f"{date_obj.day} {MONTHS_GENITIVE_NUM[date_obj.month]} {date_obj.year} года"
            bot.send_message(
                message.chat.id,
                f"Секунду, ищу мероприятия на {human_date} на sobaka.su...",
            )
            description = describe_events_for_date(date_obj)
            bot.send_message(message.chat.id, description)
            return

    if text == "привет":
        bot.send_message(
            message.chat.id,
            f"Здравствуйте, {_user_display_name(message.from_user)}!",
        )
    elif text == "id":
        bot.reply_to(message, f"ID: {message.from_user.id}")
    else:
        # Всё остальное — вопрос к Mistral (если не отключён)
        if WORK_WITHOUT_AI:
            bot.send_message(
                message.chat.id,
                "❌ AI-ответы временно отключены.\n\n"
                "Бот сейчас работает только с командами:\n"
                "/start - начать общение\n"
                "/help - список команд\n"
                "/site - открыть сайт\n"
                "/vk - открыть ВК\n"
                "/vk_org - мероприятия в кафе (ВК)\n"
                "/phone - телефон\n"
                "/whatsapp - WhatsApp\n"
                "/refund - возврат билетов (Ticketscloud)\n"
                "/forget - очистить память диалога с AI\n"
                "/menu - файл меню (предзаказ)\n"
                "/predzakaz - предзаказ администратору\n"
                "/event - выбрать дату и узнать мероприятия",
            )
            return

        print(f"[DEBUG] Получено сообщение от пользователя: {message.text}")
        raw = (message.text or "").strip()
        send_menu_bundle = _user_requests_menu_file_attachment(raw)
        if send_menu_bundle:
            send_menu_bundle_to_chat(message.chat.id)
        else:
            bot.send_message(message.chat.id, "Секунду — загляну в суфлёр и отвечу…")
        answer = ask_gpt(message.chat.id, message.text, menu_delivered=send_menu_bundle)
        bot.send_message(message.chat.id, answer)


def main() -> None:
    """
    Обычный polling() сразу вызывает getMe и падает при таймауте до api.telegram.org.
    infinity_polling внутри цикла ловит ConnectTimeout/SSL и повторяет запросы (~каждые 3 с).

    logger_level=None — иначе pyTelegramBotAPI на каждый таймаут пишет ERROR и полный traceback.
    TELEGRAM_VERBOSE_POLLING=1 — снова показывать эти логи (отладка).
    """
    verbose = os.environ.get("TELEGRAM_VERBOSE_POLLING", "").strip().lower() in ("1", "true", "yes")
    poll_log_level = logging.ERROR if verbose else None

    print(
        "[INFO] Нужен доступ к https://api.telegram.org (443).\n"
        "    Локальный прокси VPN: TELEGRAM_PROXY=http://127.0.0.1:ПОРТ или socks5://… (PySocks).\n"
        "    При SSLEOF с TELEGRAM_PROXY не задавайте одновременно лишний HTTP(S)_PROXY для тех же запросов — бот отключает trust_env.\n"
        "    Без TELEGRAM_PROXY: учитываются HTTP(S)_PROXY из окружения.\n"
        + (
            ""
            if verbose
            else "[INFO] Пока нет связи, длинные трейсбеки TeleBot отключены (см. TELEGRAM_VERBOSE_POLLING=1).\n"
        )
    )
    restart_after = 15
    while True:
        try:
            bot.infinity_polling(
                skip_pending=False,
                timeout=60,
                long_polling_timeout=30,
                logger_level=poll_log_level,
            )
            print(f"[WARN] Цикл polling завершился — пауза {restart_after} с и снова.")
        except KeyboardInterrupt:
            print("[INFO] Остановка по Ctrl+C.")
            break
        except Exception as e:
            print(f"[WARN] {_sanitize_bot_token_in_text(repr(e))} — пауза {restart_after} с и повтор.")
        time.sleep(restart_after)


if __name__ == "__main__":
    main()
