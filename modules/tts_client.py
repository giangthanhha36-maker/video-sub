"""
CODE MỚI: Client gọi sang service OmniVoice (audio.py) để tạo audio lồng tiếng.

Vì OmniVoice cần môi trường riêng (python 3.12 + torch cu128) xung đột với
paddleocr của pipeline xoá phụ đề, ta chạy audio.py như 1 service Gradio riêng
và gọi sang qua API `gradio_client`. Model nạp 1 lần, dùng lại cho nhiều video
-> tiết kiệm thời gian/chi phí GPU.

Service phải được khởi động trước, ví dụ:
    python audio.py --model k2-fsa/OmniVoice --device cuda --ip 0.0.0.0 --port 7861

Endpoint sử dụng: api_name="/srt_to_speech" (khai báo trong audio.py).
Thứ tự tham số PHẢI khớp đúng inputs của nút Generate trong audio.py.
"""

import os
import shutil


def generate_dub(
    srt_path: str,
    server_url: str = "http://127.0.0.1:7861",
    ref_audio: str = None,
    ref_text: str = None,
    language: str = "Auto",
    num_step: int = 32,
    guidance_scale: float = 2.0,
    denoise: bool = True,
    max_speed: float = 1.3,
    hard_sync: bool = True,
    out_wav: str = None,
) -> str:
    """
    Gọi service OmniVoice để chuyển file .srt -> 1 file audio lồng tiếng (.wav)
    khớp đúng timeline.

    Tham số:
    - srt_path: đường dẫn file .srt đã dịch.
    - server_url: địa chỉ service OmniVoice (mặc định cổng 7861 trên cùng máy).
    - ref_audio: file giọng mẫu (3-10s) để clone giọng. Bỏ trống = giọng mặc định.
    - ref_text: lời thoại của giọng mẫu (giúp clone chuẩn hơn). Có thể bỏ trống.
    - language: ngôn ngữ đích (vd "Vietnamese"); "Auto" để model tự dò.
    - num_step / guidance_scale / denoise / max_speed: tham số sinh audio.
    - hard_sync: True = neo cứng timeline (không trượt giờ) - nên để True khi ghép video.
    - out_wav: nơi lưu file kết quả. Bỏ trống = tự đặt cạnh file srt.

    Trả về:
    - đường dẫn file .wav đã tạo.
    """
    # import muộn để pipeline không bắt buộc cài gradio_client nếu không lồng tiếng
    from gradio_client import Client, handle_file

    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"Không tìm thấy file SRT: {srt_path}")

    if out_wav is None:
        base, _ = os.path.splitext(srt_path)
        out_wav = f"{base}_dub.wav"

    client = Client(server_url)

    srt_arg = handle_file(srt_path)
    ref_arg = handle_file(ref_audio) if ref_audio else None

    # Thứ tự đối số khớp đúng inputs của _srt_dispatch trong audio.py:
    # [srt_file, language, ref_audio, ref_text, instruct, num_step,
    #  guidance_scale, denoise, speed, duration, preprocess_prompt,
    #  postprocess_output, fit_timeline, max_speed, hard_sync]
    result = client.predict(
        srt_arg,                       # srt_file
        language or "Auto",            # language
        ref_arg,                       # ref_audio
        ref_text or "",                # ref_text
        "",                            # instruct
        int(num_step),                 # num_step
        float(guidance_scale),         # guidance_scale
        bool(denoise),                 # denoise
        1.0,                           # speed (hard-sync tự lo, để 1.0)
        0,                             # duration (0 = tự động)
        True,                          # preprocess_prompt
        True,                          # postprocess_output
        True,                          # fit_timeline
        float(max_speed),              # max_speed
        bool(hard_sync),               # hard_sync
        api_name="/srt_to_speech",
    )

    # result thường là (audio, status). audio có thể là đường dẫn (str) hoặc dict.
    audio_part = result[0] if isinstance(result, (list, tuple)) else result
    audio_src = _extract_path(audio_part)
    if not audio_src or not os.path.exists(audio_src):
        raise RuntimeError(
            f"Service OmniVoice không trả về file audio hợp lệ (nhận: {audio_part!r})."
        )

    shutil.copy(audio_src, out_wav)
    return out_wav


def _extract_path(audio_part):
    """Lấy đường dẫn file từ kết quả Audio mà gradio_client trả về.

    Tuỳ phiên bản Gradio, kết quả có thể là:
    - str: đường dẫn file
    - dict: {"path": ...} hoặc {"name": ...} hoặc {"value": ...}
    - tuple/list: (path, ...)
    """
    if audio_part is None:
        return None
    if isinstance(audio_part, str):
        return audio_part
    if isinstance(audio_part, dict):
        for key in ("path", "name", "value", "url"):
            if audio_part.get(key):
                return audio_part[key]
        return None
    if isinstance(audio_part, (list, tuple)) and audio_part:
        return _extract_path(audio_part[0])
    return None
