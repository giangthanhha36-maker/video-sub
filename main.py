import argparse
import json  # [THAY DOI - RESUME] doc file _ocr_center.json khi chay tiep
import os
import shutil

from modules.config import load_config
from modules.embed import embed_subtitles
from modules.erase import remove_subtitles
# [THAY DOI - RESUME] them load_ocr_result + compute_center_from_result
# de doc lai ket qua OCR cu va tinh lai center khi thieu file center.
from modules.ocr import (
    compute_center_from_result,
    extract_subtitles,
    load_ocr_result,
)
from modules.subtitle import get_subtitles
from modules.translate import translate_subtitles
# [THAY DOI - LONG TIENG] them client goi OmniVoice + ham tron audio.
from modules import tts_client
from utils.audio_utils import mix_and_mux
from utils.logging_utils import update_status
from utils.video_utils import (
    create_video,
    detect_fps,
    extract_frames,
    get_temp_directory_path,
    get_temp_frame_paths,
)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="SubErase-Translate-Embed: A tool for erasing, translating, and embedding subtitles."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--language", required=True, help="Target language code for translation."
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Whether to delete the temporary directory after processing.",
    )
    # [THAY DOI - LONG TIENG] tham so cho buoc long tieng (dub) bang OmniVoice.
    parser.add_argument(
        "--ref-audio",
        default=None,
        help="Duong dan file giong mau (3-10s) de clone giong khi long tieng.",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Loi thoai cua giong mau (tuy chon, giup clone chuan hon).",
    )
    parser.add_argument(
        "--no-dub",
        action="store_true",
        help="Tat buoc long tieng (chi cay phu de, khong tao giong dich).",
    )
    args = parser.parse_args()

    # 开始处理
    update_status(f"Start! {args.video}")
    config = load_config()
    file_name, ext = os.path.splitext(args.video)
    fps = detect_fps(args.video)

    # ============================================================
    # [GIU NGUYEN - DA COMMENT] Pipeline GOC luon chay lai tu dau (khong resume).
    # Giu lai de tham khao/quay ve neu can.
    # ------------------------------------------------------------
    # # 提取视频帧
    # update_status(f"Source: extracting frames with {fps} FPS...")
    # extract_frames(args.video, fps)
    # temp_directory_path = get_temp_directory_path(args.video)
    # frame_paths = get_temp_frame_paths(temp_directory_path)
    #
    # # 使用 OCR 提取字幕
    # update_status("OCR: extracting subtitles...")
    # ocr_result, y_center = extract_subtitles(frame_paths, config, fps)
    #
    # # 生成字幕
    # srt_path = get_subtitles(ocr_result, config, fps, file_name)
    #
    # # 擦除原有字幕
    # update_status("Erase: removing subtitles...")
    # remove_subtitles(ocr_result, fps, len(frame_paths), config)
    # output_path = f"{file_name}_output{ext}"
    # create_video(args.video, output_path, fps)
    #
    # # 翻译字幕
    # update_status("Translate: translating subtitles...")
    # srt_lang_path = translate_subtitles(srt_path, args.language)
    #
    # # 将翻译后的字幕嵌入视频
    # update_status("Embed: embedding subtitles...")
    # output_file = f"{file_name}_{args.language}{ext}"
    # embed_subtitles(output_path, srt_lang_path, y_center, output_file, config)
    # ============================================================

    # ============================================================
    # [THAY DOI - RESUME] Pipeline MOI: chay tiep tu cho da lam xong,
    # tan dung cac file ket qua da tao ra (khong lam lai cac buoc nang).
    # ------------------------------------------------------------
    output_path = f"{file_name}_output{ext}"  # video da xoa phu de
    output_file = f"{file_name}_{args.language}{ext}"  # video ket qua cuoi cung
    ocr_check_path = f"{file_name}_ocr_check.json"
    ocr_center_path = f"{file_name}_ocr_center.json"

    # Truong hop 1: da xoa phu de xong (co _output) va da co ket qua OCR (_ocr_check.json)
    #   -> BO QUA: tach khung hinh + OCR + xoa phu de (3 buoc nang nhat).
    # [THAY DOI - RESUME] Khong bat buoc phai co _ocr_center.json: neu thieu thi tinh lai
    # center tu _ocr_check.json (giup dung lai ca ket qua tao boi ban code cu).
    if os.path.exists(output_path) and os.path.exists(ocr_check_path):
        update_status(
            "Resume: da co video xoa phu de + ket qua OCR -> bo qua tach khung hinh, OCR, xoa phu de."
        )
        ocr_result = load_ocr_result(ocr_check_path)
        if os.path.exists(ocr_center_path):
            with open(ocr_center_path, "r", encoding="utf-8") as f:
                y_center = json.load(f).get("center", 0)
        else:
            # Thieu file center (vd tao boi ban cu) -> tinh lai, khoi OCR lai.
            y_center = compute_center_from_result(
                ocr_result, config["video"]["groups_tolerance"]
            )
            with open(ocr_center_path, "w", encoding="utf-8") as f:
                json.dump({"center": float(y_center)}, f)
        srt_path = get_subtitles(ocr_result, config, fps, file_name)
    else:
        # Tach khung hinh (extract_frames da them -y nen khong bi treo khi chay lai)
        update_status(f"Source: extracting frames with {fps} FPS...")
        extract_frames(args.video, fps)
        temp_directory_path = get_temp_directory_path(args.video)
        frame_paths = get_temp_frame_paths(temp_directory_path)

        # OCR: tu dong dung lai _ocr_check.json neu da co (xem modules/ocr.py)
        update_status("OCR: extracting subtitles...")
        ocr_result, y_center = extract_subtitles(frame_paths, config, fps)

        # Tao file phu de goc
        srt_path = get_subtitles(ocr_result, config, fps, file_name)

        # Xoa phu de goc roi ghep thanh video _output
        update_status("Erase: removing subtitles...")
        remove_subtitles(ocr_result, fps, len(frame_paths), config)
        create_video(args.video, output_path, fps)

    # Dich phu de (ham translate_subtitles da co san co che bo qua neu file dich da ton tai)
    update_status("Translate: translating subtitles...")
    srt_lang_path = translate_subtitles(srt_path, args.language)

    # Chen phu de da dich vao video. Bo qua neu video ket qua da ton tai.
    if os.path.exists(output_file):
        update_status("Resume: video ket qua da ton tai -> bo qua buoc chen phu de.")
    else:
        update_status("Embed: embedding subtitles...")
        embed_subtitles(output_path, srt_lang_path, y_center, output_file, config)
    # ============================================================

    # ===================== [THAY DOI - LONG TIENG] =====================
    # Tao giong long tieng (OmniVoice) + tron tieng goc 0.3 / giong dich 0.7
    # roi ghep vao video. Bo qua neu tat (--no-dub) hoac dub.enable = false,
    # va bo qua neu video long tieng da ton tai (resume).
    dub_cfg = config.get("dub", {})
    tts_cfg = config.get("tts", {})
    dub_video = f"{file_name}_{args.language}_dub{ext}"
    if (not args.no_dub) and dub_cfg.get("enable", True):
        if os.path.exists(dub_video):
            update_status("Resume: video long tieng da ton tai -> bo qua buoc long tieng.")
        else:
            try:
                update_status("Dub: tao giong long tieng bang OmniVoice...")
                tts_wav = f"{file_name}_dub.wav"
                tts_client.generate_dub(
                    srt_path=srt_lang_path,
                    server_url=tts_cfg.get("server_url", "http://127.0.0.1:7861"),
                    ref_audio=args.ref_audio,
                    ref_text=args.ref_text,
                    language=args.language,
                    num_step=int(tts_cfg.get("num_step", 32)),
                    max_speed=float(tts_cfg.get("max_speed", 1.3)),
                    hard_sync=bool(tts_cfg.get("hard_sync", True)),
                    out_wav=tts_wav,
                )
                update_status("Dub: tron tieng goc + giong dich roi ghep vao video...")
                mix_and_mux(
                    video_in=output_file,
                    tts_wav=tts_wav,
                    output=dub_video,
                    orig_gain=float(dub_cfg.get("orig_gain", 0.3)),
                    tts_gain=float(dub_cfg.get("tts_gain", 0.7)),
                )
                update_status(f"Dub: xong -> {dub_video}")
            except Exception as e:
                # Long tieng loi -> van giu video co phu de (khong lam hong ket qua).
                update_status(
                    f"Canh bao: long tieng that bai ({e}). "
                    "Giu lai video co phu de (chua long tieng)."
                )
    # ===================================================================

    if args.delete:
        if os.path.exists(file_name):
            shutil.rmtree(file_name)
            update_status("Temporary request directory {} deleted".format(file_name))

    update_status(f"Done! {args.video}")


if __name__ == "__main__":
    main()
