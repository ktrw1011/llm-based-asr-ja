# ruff: noqa
import unicodedata

import mojimoji

DELETION_SYMBOLS = [
    "?",
    "!",
    ":",
    "\\",
    "〔",
    "□",
    "*",
    "Ⅺ",
    "┘",
    "'",
    "⑺",
    "♀",
    "Ⅰ",
    "▼",
    "※",
    "▶",
    "▲",
    "゜",
    "○",
    "Ⅴ",
    "」",
    "・",
    "♭",
    "』",
    "【",
    "《",
    "┐",
    "△",
    "◆",
    "▽",
    "『",
    "★",
    "_",
    ";",
    "♦",
    "〉",
    "‥",
    "◎",
    "♠",
    "⑿",
    "♪",
    "㈪",
    "〈",
    "`",
    "⑸",
    "Ⅶ",
    "♥",
    "〇",
    "─",
    "♣",
    "♯",
    "》",
    "/",
    ",",
    "☎",
    "】",
    "―",
    "゛",
    "⑾",
    "☆",
    '"',
    "Ⅱ",
    "#",
    "⑷",
    "‐",
    "‼",
    "^",
    "㈭",
    "@",
    "●",
    '"',
    "〕",
    "−",
    "♂",
    "Ⅳ",
    "「",
    "⁉",
    "(",
    "|",
    "]",
    "{",
    "[",
    ")",
]


def normalize_text(text: str) -> str:
    text = mojimoji.zen_to_han(text, kana=False)
    text = mojimoji.han_to_zen(text, digit=False, ascii=False)
    text = unicodedata.normalize("NFKC", text)
    return text


def delete_symbols(text: str) -> str:
    for symbol in DELETION_SYMBOLS:
        text = text.replace(symbol, "")
    return text
