"""
Giao dien Gradio cho SubErase-Translate-Embed.

Cong dung:
- Nhap video + cau hinh qua web thay vi go lenh.
- Chay day du pipeline: tach khung hinh -> OCR -> tao phu de -> xoa phu de goc
  -> dich -> chen phu de moi vao video.
- Tu dong upload cac file ket qua len Google Drive (neu bat).

Luu y bao mat:
- API key va file Service Account JSON chi nam trong bo nho phien chay,
  KHONG ghi ra config.yaml hay commit len git.

Chay:
    python app.py
hoac dung run_ui.bat tren Windows.
"""

import copy
import os
import shutil
import traceback
from datetime import datetime

import gradio as gr
import yaml

# Cac ngon ngu dich pho bien (co the go tay neu khong co trong danh sach)
TARGET_LANGUAGES = [
    "English",
    "Vietnamese",
    "Chinese",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
]

SOURCE_LANGUAGES = [
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "Vietnamese",
]


def _load_base_config():
    """Doc config.yaml neu co, neu khong thi doc config-template.yaml."""
    for path in ("config.yaml", "config-template.yaml"):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("Khong tim thay config.yaml hoac config-template.yaml")


def process_video(
    video_path,
    target_language,
    source_language,
    api_key,
    api_base_url,
    model,
    sa_file,
    drive_folder_id,
    drive_enable,
    mask_expand,
    font_size,
    font_color,
    position,
    target_size,
    min_duration,
    delete_temp,
    # CODE MỚI: tham số cho bước lồng tiếng (dub) bằng OmniVoice.
    dub_enable,
    ref_audio,
    ref_text,
    omnivoice_url,
    orig_gain,
    tts_gain,
    dub_num_step,
    dub_max_speed,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Ham xu ly chinh, la mot generator de cap nhat tien do len giao dien.

    Moi lan `yield` se cap nhat 3 o dau ra: (nhat ky, video ket qua, link Drive).
    """
    log_lines = []

    def log(msg):
        """Ghi them mot dong vao nhat ky va tra ve chuoi day du."""
        stamp = datetime.now().strftime("%H:%M:%S")
        log_lines.append(f"[{stamp}] {msg}")
        return "\n".join(log_lines)

    # --- Kiem tra dau vao bat buoc ---
    if not video_path:
        yield log("Loi: chua chon video."), None, ""
        return
    if not api_key or not api_key.strip():
        yield log("Loi: chua nhap API key o tab 'Cai dat'."), None, ""
        return

    try:
        # --- Import muon (lazy) de giao dien khoi dong nhanh, tranh loi env luc mo trang ---
        from modules.config import load_config  # noqa: F401 (giu de tham khao)
        from modules.embed import embed_subtitles
        from modules.erase import remove_subtitles
        from modules.ocr import extract_subtitles
        from modules.subtitle import get_subtitles
        from modules.translate import translate_subtitles
        from utils import llm_utils
        from utils.video_utils import (
            create_video,
            detect_fps,
            extract_frames,
            get_temp_directory_path,
            get_temp_frame_paths,
        )

        # CODE MỚI: import muộn cho bước lồng tiếng (chỉ cần khi dub_enable=True).
        from modules import tts_client
        from utils.audio_utils import mix_and_mux

        # --- Chuan bi thu muc lam viec rieng cho lan chay nay ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        work_dir = os.path.join("output", f"{base_name}_{timestamp}")
        os.makedirs(work_dir, exist_ok=True)

        ext = os.path.splitext(video_path)[1] or ".mp4"
        local_video = os.path.join(work_dir, base_name + ext)
        shutil.copy(video_path, local_video)
        yield log(f"Da copy video vao: {local_video}"), None, ""

        # --- Lay config goc va ghi de bang thong so tu giao dien ---
        config = copy.deepcopy(_load_base_config())
        config["translation"]["api_key"] = api_key.strip()
        if api_base_url and api_base_url.strip():
            config["translation"]["api_base_url"] = api_base_url.strip()
        if model and model.strip():
            config["translation"]["model"] = model.strip()

        config["erase"]["mask_expand"] = int(mask_expand)
        config["subtitle"]["font_size"] = int(font_size)
        config["subtitle"]["font_color"] = font_color
        config["subtitle"]["position"] = float(position)
        config["output"]["target_size"] = int(target_size)
        config["video"]["min_duration"] = float(min_duration)

        # Cho phep cac module dung chung (translate) lay dung API key/model luc chay
        llm_utils.set_runtime(
            api_key=api_key.strip(),
            api_base_url=config["translation"]["api_base_url"],
            model=config["translation"]["model"],
        )

        # --- Chuan bi Google Drive (neu bat) ---
        drive_service = None
        drive_job_folder = ""
        drive_links = []
        if drive_enable:
            if sa_file:
                try:
                    from modules import drive as drive_mod

                    drive_service = drive_mod.build_service(sa_file)
                    drive_job_folder = drive_mod.ensure_job_folder(
                        drive_service,
                        (drive_folder_id or "").strip(),
                        base_name,
                    )
                    yield log("Da ket noi Google Drive, tao thu muc job xong."), None, ""
                except Exception as e:
                    drive_service = None
                    yield log(f"Canh bao: khong ket noi duoc Drive ({e}). Bo qua upload."), None, ""
            else:
                yield log("Canh bao: bat Drive nhung chua tai file Service Account JSON. Bo qua upload."), None, ""

        def upload_if_enabled(local_path):
            """Upload mot file len Drive neu da bat, va ghi link vao danh sach."""
            if drive_service and local_path and os.path.exists(local_path):
                try:
                    from modules import drive as drive_mod

                    info = drive_mod.upload_file(drive_service, local_path, drive_job_folder)
                    link = info.get("link") or info.get("id")
                    drive_links.append(f"- [{info['name']}]({link})")
                    return log(f"Da upload len Drive: {info['name']}")
                except Exception as e:
                    return log(f"Loi upload Drive ({os.path.basename(local_path)}): {e}")
            return "\n".join(log_lines)

        def links_md():
            if not drive_links:
                return ""
            return "### File tren Google Drive\n" + "\n".join(drive_links)

        # ===================== BAT DAU PIPELINE =====================
        file_name = os.path.join(work_dir, base_name)

        fps = detect_fps(local_video)
        yield log(f"FPS phat hien: {fps}"), None, links_md()

        yield log("Buoc 1/6: tach khung hinh tu video..."), None, links_md()
        extract_frames(local_video, fps)
        temp_directory_path = get_temp_directory_path(local_video)
        frame_paths = get_temp_frame_paths(temp_directory_path)
        yield log(f"Da tach {len(frame_paths)} khung hinh."), None, links_md()

        yield log("Buoc 2/6: OCR nhan dien phu de..."), None, links_md()
        ocr_result, y_center = extract_subtitles(frame_paths, config, fps)

        yield log("Buoc 3/6: tao file phu de goc (.srt)..."), None, links_md()
        srt_path = get_subtitles(ocr_result, config, fps, file_name)
        yield upload_if_enabled(srt_path), None, links_md()

        yield log("Buoc 4/6: xoa phu de goc khoi video (STTN)..."), None, links_md()
        remove_subtitles(ocr_result, fps, len(frame_paths), config)
        output_path = f"{file_name}_output{ext}"
        create_video(local_video, output_path, fps)
        yield upload_if_enabled(output_path), None, links_md()

        yield log("Buoc 5/6: dich phu de (co the mat vai phut)..."), None, links_md()
        srt_lang_path = translate_subtitles(
            srt_path, target_language, source_lang=source_language
        )
        yield upload_if_enabled(srt_lang_path), None, links_md()

        yield log("Buoc 6/7: chen phu de da dich vao video..."), None, links_md()
        output_file = f"{file_name}_{target_language}{ext}"
        embed_subtitles(output_path, srt_lang_path, y_center, output_file, config)
        yield upload_if_enabled(output_file), None, links_md()

        # ===================== CODE MỚI: BƯỚC LỒNG TIẾNG =====================
        # Video kết quả mặc định = video đã cháy phụ đề (còn tiếng gốc).
        final_output = output_file
        if dub_enable:
            try:
                yield log(
                    "Buoc 7/7: tao giong long tieng bang OmniVoice "
                    f"(service: {omnivoice_url})..."
                ), None, links_md()
                tts_wav = f"{file_name}_dub.wav"
                tts_client.generate_dub(
                    srt_path=srt_lang_path,
                    server_url=(omnivoice_url or "").strip()
                    or "http://127.0.0.1:7861",
                    ref_audio=ref_audio,
                    ref_text=(ref_text or "").strip() or None,
                    language=target_language,
                    num_step=int(dub_num_step),
                    max_speed=float(dub_max_speed),
                    hard_sync=True,  # neo cung timeline: khong truot gio
                    out_wav=tts_wav,
                )
                yield upload_if_enabled(tts_wav), None, links_md()

                yield log(
                    "Tron tieng goc + giong dich roi ghep vao video..."
                ), None, links_md()
                dub_video = f"{file_name}_{target_language}_dub{ext}"
                mix_and_mux(
                    video_in=output_file,
                    tts_wav=tts_wav,
                    output=dub_video,
                    orig_gain=float(orig_gain),
                    tts_gain=float(tts_gain),
                )
                final_output = dub_video
                yield upload_if_enabled(dub_video), None, links_md()
            except Exception as e:
                # Long tieng loi -> van giu video co phu de (khong lam hong ket qua).
                yield log(
                    f"Canh bao: long tieng that bai ({e}). "
                    "Giu lai video co phu de (chua long tieng)."
                ), None, links_md()
        # ===================================================================

        # --- Don dep thu muc khung hinh tam neu chon ---
        if delete_temp and os.path.isdir(temp_directory_path):
            shutil.rmtree(temp_directory_path, ignore_errors=True)
            log(f"Da xoa thu muc tam: {temp_directory_path}")

        yield log("HOAN TAT!"), final_output, links_md()

    except Exception as e:
        err = traceback.format_exc()
        yield log(f"Loi: {e}\n{err}"), None, ""


def build_ui():
    with gr.Blocks(title="SubErase-Translate-Embed") as demo:
        gr.Markdown(
            "# SubErase-Translate-Embed\n"
            "Xoa phu de goc, dich va chen phu de moi vao video. "
            "Nhap API key / Google Drive o tab **Cai dat** truoc khi chay."
        )

        with gr.Tabs():
            # ---------------- TAB XU LY ----------------
            with gr.Tab("Xu ly"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_in = gr.Video(label="Video dau vao")
                        target_lang = gr.Dropdown(
                            choices=TARGET_LANGUAGES,
                            value="Vietnamese",
                            label="Ngon ngu dich (target)",
                            allow_custom_value=True,
                        )

                        with gr.Accordion("Tuy chon nang cao (khong bat buoc)", open=False):
                            mask_expand = gr.Slider(
                                0, 60, value=20, step=1,
                                label="mask_expand (px no rong vung xoa)",
                            )
                            font_size = gr.Slider(
                                0, 80, value=20, step=1,
                                label="font_size (0 = tu tinh)",
                            )
                            font_color = gr.ColorPicker(
                                value="#FFFFFF", label="Mau chu phu de"
                            )
                            position = gr.Slider(
                                0.0, 1.0, value=0.0, step=0.01,
                                label="position (0 = vi tri phu de goc)",
                            )
                            target_size = gr.Slider(
                                5, 200, value=30, step=1,
                                label="Dung luong video xuat (MB)",
                            )
                            min_duration = gr.Slider(
                                0.05, 2.0, value=0.1, step=0.05,
                                label="min_duration (giay) - thoi luong toi thieu 1 phu de",
                            )
                            delete_temp = gr.Checkbox(
                                value=True, label="Xoa khung hinh tam sau khi chay"
                            )

                        run_btn = gr.Button("Bat dau xu ly", variant="primary")

                    with gr.Column(scale=1):
                        status_out = gr.Textbox(
                            label="Nhat ky tien do", lines=18, max_lines=30
                        )
                        video_out = gr.Video(label="Video ket qua")
                        links_out = gr.Markdown(label="Link Google Drive")

            # ---------------- TAB CAI DAT ----------------
            with gr.Tab("Cai dat"):
                gr.Markdown(
                    "Cac thong tin nhay cam ben duoi **chi luu trong phien chay**, "
                    "khong ghi ra file. Nhap lai moi lan mo trang."
                )
                api_key = gr.Textbox(
                    label="API key (OpenAI / chatanywhere ...)",
                    type="password",
                    placeholder="sk-...",
                )
                api_base_url = gr.Textbox(
                    label="API base URL",
                    value="https://api.openai.com/v1",
                )
                model = gr.Textbox(label="Model", value="gpt-4o-mini")
                source_lang = gr.Dropdown(
                    choices=SOURCE_LANGUAGES,
                    value="Chinese",
                    label="Ngon ngu goc cua phu de (source)",
                    allow_custom_value=True,
                )

                gr.Markdown("---\n### Google Drive")
                drive_enable = gr.Checkbox(
                    value=True, label="Upload ket qua len Google Drive"
                )
                sa_file = gr.File(
                    label="File Service Account JSON",
                    file_types=[".json"],
                    type="filepath",
                )
                drive_folder_id = gr.Textbox(
                    label="Folder ID tren Drive (da share cho email service account)",
                    placeholder="vd: 1AbCxxxxxxxxxxxxxxxx",
                )

                # ---------------- CODE MỚI: LỒNG TIẾNG (OmniVoice) ----------------
                gr.Markdown(
                    "---\n### Long tieng (OmniVoice)\n"
                    "Chay service OmniVoice (audio.py) o moi truong rieng truoc, "
                    "vi du cong 7861. Bat o duoi de tu dong tao giong dich va tron "
                    "vao video."
                )
                dub_enable = gr.Checkbox(
                    value=True, label="Bat long tieng (dich giong) vao video"
                )
                omnivoice_url = gr.Textbox(
                    label="OmniVoice server URL",
                    value="http://127.0.0.1:7861",
                )
                ref_audio = gr.Audio(
                    label="Giong mau (upload 3-10s) de clone giong",
                    type="filepath",
                )
                ref_text = gr.Textbox(
                    label="Loi thoai cua giong mau (tuy chon, giup clone chuan hon)",
                    lines=2,
                )
                with gr.Row():
                    orig_gain = gr.Slider(
                        0.0, 1.0, value=0.3, step=0.05,
                        label="Am luong tieng goc (mac dinh 0.3 = 30%)",
                    )
                    tts_gain = gr.Slider(
                        0.0, 1.0, value=0.7, step=0.05,
                        label="Am luong giong dich (mac dinh 0.7 = 70%)",
                    )
                with gr.Row():
                    dub_num_step = gr.Slider(
                        1, 64, value=32, step=1,
                        label="num_step (cao = hay hon, cham hon)",
                    )
                    dub_max_speed = gr.Slider(
                        1.0, 2.0, value=1.3, step=0.05,
                        label="Gioi han tang toc cau dai (MAX_SPEED)",
                    )

        run_btn.click(
            fn=process_video,
            inputs=[
                video_in,
                target_lang,
                source_lang,
                api_key,
                api_base_url,
                model,
                sa_file,
                drive_folder_id,
                drive_enable,
                mask_expand,
                font_size,
                font_color,
                position,
                target_size,
                min_duration,
                delete_temp,
                # CODE MỚI: cac input cho buoc long tieng (dung thu tu voi process_video)
                dub_enable,
                ref_audio,
                ref_text,
                omnivoice_url,
                orig_gain,
                tts_gain,
                dub_num_step,
                dub_max_speed,
            ],
            outputs=[status_out, video_out, links_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    # Cau hinh khoi chay phu hop chay tren VPS Linux (khong tu mo trinh duyet).
    #  - GRADIO_PORT : doi cong (mac dinh 7860)
    #  - GRADIO_SHARE=1 : tao link public *.gradio.live (huu ich neu VPS chua mo cong)
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        inbrowser=False,
    )
