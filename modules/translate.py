import os
import re  # [THAY DOI - CHUNK] tach dong "N. noi dung" trong ket qua dich
import time

import pysrt

from utils.llm_utils import get_completion  # [THAY DOI - CHUNK] goi dich truc tiep theo chunk
from utils.translation_utils import translate_text


# [THAY DOI] Them tham so source_lang (ngon ngu goc) thay vi co dinh "Chinese".
# Mac dinh van la "Chinese" de KHONG anh huong cach goi cu (tuong thich nguoc).
def chatgpt_translate(text: str, language: str, source_lang: str = "Chinese"):
    """
    使用ChatGPT模型翻译字幕文本。

    参数:
    - text: str，需要翻译的字幕文本。
    - language: str，目标翻译语言，如"English"。
    - source_lang: str，源语言，默认 "Chinese"。

    返回:
    - str，翻译后的字幕文本或错误信息。
    """
    content = ""
    try:
        # [THAY DOI] dung source_lang dong thay cho "Chinese" co dinh
        content = translate_text(
            source_lang=source_lang, target_lang=language, source_text=text
        )
    except Exception as e:
        print(f"chatgpt translate error:" + str(e))
    return content


def check_timeline(srt, new_srt_text):
    """
    检查字幕时间线是否一致。

    本函数通过比较原始字幕对象和新的字幕文本中每个字幕片段的开始和结束时间，
    来验证两者的时间线是否完全一致。这用于确保字幕内容经过修改后（如翻译或格式调整），
    其时间戳未发生任何变化，这对于保持字幕与视频内容的同步至关重要。

    参数:
    - srt: pysrt.SubRipFile对象，代表原始的字幕数据。
    - new_srt_text: 字符串，表示经过修改但需要验证时间线的新字幕文本。

    返回:
    - 布尔值：如果所有字幕片段的开始和结束时间都相同，则返回True，表示时间线一致；
      否则，在发现不一致时立即返回False。
    """
    new_srt = pysrt.SubRipFile().from_string(new_srt_text)

    for s, new_s in zip(srt, new_srt):
        if s.start != new_s.start:
            return False
        if s.end != new_s.end:
            return False

    return True


# ============================================================
# [THAY DOI - CHUNK] Cac ham ho tro dich theo TUNG CHUNK NHO.
# Y tuong: chi dich phan CHU cua phu de, GIU NGUYEN timestamp goc.
# -> Khong bao gio lech timeline, va chia nho content de model khong bi qua tai.
# ============================================================
def _translate_chunk_texts(texts: list, target_language: str, source_lang: str) -> str:
    """
    Dich mot nhom (chunk) cac dong phu de. Dau vao la danh sach text,
    tra ve chuoi dang danh so "N. ban dich" de tach lai.

    Goi 1 lan API cho ca chunk (re va on dinh hon cach dich tung dong).
    """
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    system_message = (
        f"You are a professional subtitle translator from {source_lang} to {target_language}."
    )
    prompt = (
        f"Translate the following numbered subtitle lines from {source_lang} to {target_language}.\n"
        f"Strict rules:\n"
        f"- Keep EXACTLY the same numbering and the same number of lines.\n"
        f"- Output ONLY the translated lines, each in the format 'N. translation'.\n"
        f"- Do NOT merge, split, add or remove any line. Do NOT add explanations.\n\n"
        f"{numbered}"
    )
    return get_completion(prompt, system_message=system_message)


def _parse_numbered_translation(text: str, expected: int):
    """
    Tach ket qua dang "N. noi dung" thanh danh sach theo dung thu tu.

    Tra ve:
    - list[str] do dai = expected neu tach du; None neu thieu (de goi thu lai).
    """
    if not text:
        return None
    result = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\s*[\.\)、:]\s*(.+)$", line)
        if m:
            result[int(m.group(1))] = m.group(2).strip()
    out = []
    for i in range(1, expected + 1):
        if i not in result:
            return None
        out.append(result[i])
    return out


# [THAY DOI] Them tham so source_lang de truyen xuong chatgpt_translate.
def translate_subtitles(
    srt_path: str,
    target_language: str,
    try_times: int = 5,
    source_lang: str = "Chinese",
    chunk_size: int = 40,
):
    """
    将字幕翻译成目标语言并保存。

    该函数读取字幕文件，使用 chatgpt_translate 函数翻译内容，并将翻译后的字幕保存到新文件中。
    如果经过多次尝试后翻译的内容行数或时间格式与原始字幕不符，仍然会返回翻译后字幕文件的路径。

    :param srt_path: 字幕文件的路径。
    :param target_language: 目标语言代码，用于翻译。
    :param try_times: 重试次数，默认为 5 次。
    :return: 翻译后字幕文件的路径。
    """
    srt_path_english = srt_path.replace("_zh", f"_{target_language}")
    if os.path.exists(srt_path_english):
        return srt_path_english

    # ============================================================
    # [GIU NGUYEN - DA COMMENT] Cach cu: dich CA FILE trong 1 lan goi.
    # Nhuoc diem: video dai -> content qua nhieu -> model tra ve lech so dong /
    # lech timestamp -> "try again" lien tuc roi that bai. Giu lai de tham khao.
    # ------------------------------------------------------------
    # with open(srt_path, "r", encoding="utf-8") as f:
    #     subtitles = f.read().strip()
    # srt = pysrt.open(srt_path)
    #
    # def _count_content_lines(text: str) -> int:
    #     return len([ln for ln in text.splitlines() if ln.strip()])
    #
    # lines = _count_content_lines(subtitles)
    # for i in range(try_times):
    #     translated_subtitles = chatgpt_translate(
    #         subtitles, target_language, source_lang=source_lang
    #     )
    #     if translated_subtitles:
    #         if _count_content_lines(translated_subtitles) != lines:
    #             print(f"chatgpt translate lines not match, try again! {i + 1}")
    #         elif not check_timeline(srt, translated_subtitles):
    #             print(f"chatgpt translate timeline not match, try again! {i + 1}")
    #         else:
    #             with open(srt_path_english, "w", encoding="utf-8") as f:
    #                 f.write(translated_subtitles)
    #             return srt_path_english
    #     time.sleep(1)
    # print("[CANH BAO] Dich that bai ...")
    # return srt_path
    # ============================================================

    # ============================================================
    # [THAY DOI - CHUNK] Cach moi: dich theo TUNG CHUNK NHO, chi doi phan CHU,
    # GIU NGUYEN timestamp goc (nen khong bao gio lech timeline).
    # ------------------------------------------------------------
    srt = pysrt.open(srt_path)
    if len(srt) == 0:
        # Khong co phu de nao -> tra ve file goc.
        return srt_path

    items = list(srt)
    total_chunks = (len(items) + chunk_size - 1) // chunk_size
    n_fail = 0

    for start in range(0, len(items), chunk_size):
        chunk_index = start // chunk_size + 1
        group = items[start : start + chunk_size]
        # Gop text moi dong (gop xuong dong trong 1 phu de thanh dau cach de de danh so)
        src_texts = [it.text.strip().replace("\n", " ") for it in group]

        translated = None
        for attempt in range(try_times):
            try:
                out = _translate_chunk_texts(src_texts, target_language, source_lang)
            except Exception as e:
                print(f"chunk {chunk_index}/{total_chunks} loi goi API: {e}")
                out = ""
            parsed = _parse_numbered_translation(out, len(group))
            if parsed is not None:
                translated = parsed
                break
            print(
                f"chunk {chunk_index}/{total_chunks} dich chua khop so dong, thu lai! {attempt + 1}"
            )
            time.sleep(1)

        if translated is None:
            # Chunk nay that bai -> GIU NGUYEN text goc cho cac dong trong chunk.
            n_fail += 1
            continue
        for it, new_text in zip(group, translated):
            it.text = new_text

    # Luu lai: pysrt giu nguyen toan bo index + timestamp, chi thay phan chu.
    srt.save(srt_path_english, encoding="utf-8")

    if n_fail:
        print(
            f"[CANH BAO] {n_fail}/{total_chunks} doan (chunk) dich that bai -> "
            f"giu phu de GOC cho cac doan do. Kiem tra lai API key/model neu can."
        )
    return srt_path_english
    # ============================================================
