"""
CODE MỚI: Tiện ích trộn (mix) audio gốc với audio lồng tiếng rồi ghép vào video.

Mục tiêu nghiệp vụ: giữ lại 1 phần tiếng gốc (nhạc nền, hiệu ứng) và đưa giọng
dịch lên làm chủ đạo. Mặc định: tiếng gốc 30% + giọng dịch 70% (tỉ lệ 3/7).

Dùng ffmpeg (đã có sẵn trong pipeline) với bộ lọc amix. Tham số normalize=0 để
ffmpeg KHÔNG tự chuẩn hoá âm lượng theo số input -> giữ đúng tỉ lệ ta đặt.
"""

import os

# Tái sử dụng helper chạy ffmpeg đã có (tự thêm -hide_banner -loglevel error).
from utils.video_utils import run_ffmpeg


def mix_and_mux(
    video_in: str,
    tts_wav: str,
    output: str,
    orig_gain: float = 0.3,
    tts_gain: float = 0.7,
) -> str:
    """
    Trộn tiếng gốc của video với audio lồng tiếng, rồi xuất video mới.

    Tham số:
    - video_in: video nguồn (đã cháy phụ đề + còn tiếng gốc).
    - tts_wav: file audio lồng tiếng (đã khớp timeline).
    - output: đường dẫn video kết quả.
    - orig_gain: hệ số âm lượng tiếng gốc (mặc định 0.3 = 30%).
    - tts_gain: hệ số âm lượng giọng dịch (mặc định 0.7 = 70%).

    Trả về:
    - đường dẫn video kết quả (output).

    Ghi chú:
    - Video stream được copy thẳng (-c:v copy) nên KHÔNG mã hoá lại hình
      -> rất nhanh và không giảm chất lượng hình.
    - duration=first: độ dài audio bám theo tiếng gốc của video (tránh dài hơn video).
    """
    if not os.path.exists(video_in):
        raise FileNotFoundError(f"Không tìm thấy video nguồn: {video_in}")
    if not os.path.exists(tts_wav):
        raise FileNotFoundError(f"Không tìm thấy audio lồng tiếng: {tts_wav}")

    filter_complex = (
        f"[0:a]volume={orig_gain}[a0];"
        f"[1:a]volume={tts_gain}[a1];"
        f"[a0][a1]amix=inputs=2:normalize=0:duration=first[aout]"
    )

    commands = [
        "-y",
        "-i", video_in,
        "-i", tts_wav,
        "-filter_complex", filter_complex,
        "-map", "0:v:0",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        output,
    ]

    ok = run_ffmpeg(commands)
    if not ok:
        raise RuntimeError(
            "ffmpeg trộn audio thất bại. Kiểm tra: video có sẵn luồng audio gốc không, "
            "và file tts_wav có hợp lệ không."
        )
    return output
