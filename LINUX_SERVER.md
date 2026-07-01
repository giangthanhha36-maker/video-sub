# Triển khai trên Linux GPU Server (headless — không có màn hình)

Hướng dẫn này dành cho server Linux **chỉ có GPU, không có card màn hình**, bạn thao tác hoàn toàn qua **terminal** (SSH). Giao diện web vẫn dùng được qua trình duyệt trên máy cá nhân.

---

## 1. Tóm tắt cổng (quan trọng)

| Cổng | Dịch vụ | Mở ra internet? | Mục đích |
|------|---------|-------------------|----------|
| **7860** | `app.py` — giao diện chính | **CÓ** (cổng bạn truy cập) | Upload video, cấu hình, xem kết quả |
| **7861** | `audio.py` — OmniVoice API | **KHÔNG** (chỉ localhost) | Pipeline gọi nội bộ để lồng tiếng |

**URL truy cập giao diện:**

```text
http://<IP-của-server>:7860
```

Ví dụ server có IP `203.0.113.50` → mở trình duyệt: `http://203.0.113.50:7860`

**Không cần mở cổng 7861** ra ngoài. Service OmniVoice chỉ lắng nghe để `app.py` gọi qua `http://127.0.0.1:7861`.

---

## 2. Yêu cầu hệ thống

- Ubuntu 20.04 / 22.04 / 24.04 (hoặc distro Linux tương đương)
- GPU NVIDIA + driver (`nvidia-smi` chạy được)
- RAM khuyến nghị ≥ 16 GB, VRAM ≥ 12 GB (16 GB+ nếu chạy đồng thời OCR + OmniVoice)
- Ổ đĩa trống ≥ 30 GB (model + video tạm)
- Công cụ: `git`, `conda` (Miniconda), `ffmpeg`, `imagemagick`

Cài phụ thuộc hệ thống (chạy 1 lần):

```bash
sudo apt update
sudo apt install -y git ffmpeg imagemagick wget
```

Cài Miniconda (nếu chưa có):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# dong terminal, mo lai, hoac: source ~/.bashrc
```

**Conda 26+** bắt chấp nhận Terms of Service trước khi tạo môi trường. Script `setup_linux_gpu.sh` tự xử lý; nếu chạy `conda create` thủ công và báo lỗi ToS:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

---

## 3. Clone repo và cài môi trường

```bash
git clone --recursive https://github.com/chenwr727/SubErase-Translate-Embed.git
cd SubErase-Translate-Embed

chmod +x setup_linux_gpu.sh run_ui.sh run_omnivoice.sh start_linux.sh stop_linux.sh
./setup_linux_gpu.sh
```

source ~/miniconda3/etc/profile.d/conda.sh   # đổi path nếu cần
./setup_linux_gpu.sh

Script `setup_linux_gpu.sh` tự tạo 2 môi trường conda và **ghim phiên bản** Gradio/Paddle/cuDNN để tránh lỗi thường gặp trên Docker.

| Môi trường | Python | Vai trò |
|------------|--------|---------|
| `ste` | 3.10 | Pipeline chính + giao diện `app.py` |
| `omnivoice` | 3.12 | Service lồng tiếng `audio.py` |

**Đã cài rồi nhưng gặp lỗi** (cuDNN, Gradio, Cython…)? Chạy sửa nhanh:

```bash
./scripts/fix_ste_env.sh
./stop_linux.sh && ./start_linux.sh --share
```

> Card **50-series** (RTX 5090…): nếu setup báo lỗi Paddle, xem **CASE B** trong [index.md](index.md) mục 4 và cài lại torch/paddle thủ công trong env `ste`.

Sửa cấu hình:

```bash
cp config-template.yaml config.yaml   # neu setup chua tao
nano config.yaml                      # dien API key dich (translation.api_key)
```

---

## 4. Tải model AI (bắt buộc)

Tạo thư mục `models/` và tải (xem chi tiết [index.md](index.md) mục 6):

```bash
mkdir -p models && cd models

# PaddleOCR (giai nen sau khi tai)
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar
wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer.tar
tar -xf ch_PP-OCRv4_det_server_infer.tar
tar -xf ch_PP-OCRv4_rec_server_infer.tar

# STTN (xoa phu de) - tai tu Google Drive, dat ten sttn.pth
# https://drive.google.com/file/d/1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv/view

cd ..
```

Cấu trúc đúng:

```text
models/
├── ch_PP-OCRv4_det_server_infer/
├── ch_PP-OCRv4_rec_server_infer/
└── sttn.pth
```

---

## 5. Khởi động dịch vụ (terminal)

### Cách 1 — Một lệnh (khuyến nghị)

```bash
./start_linux.sh
```

Hoặc kèm link public Gradio (không cần mở firewall):

```bash
./start_linux.sh --share
```

Sau vài phút, xem log:

```bash
tail -f logs/ui.log
# Tim dong URL, vi du: Running on local URL: http://0.0.0.0:7860
# Neu --share: them dong https://xxxxx.gradio.live
```

Dừng dịch vụ:

```bash
./stop_linux.sh
```

### Cách 2 — Hai terminal / tmux (kiểm soát từng service)

**Phiên 1** — OmniVoice (đợi `Model loaded.`):

```bash
./run_omnivoice.sh
```

**Phiên 2** — Giao diện chính:

```bash
export GRADIO_SHARE=0    # hoac 1 neu muon link gradio.live
./run_ui.sh
```

### Dùng tmux (giữ chạy sau khi thoát SSH)

```bash
tmux new -s ste
./start_linux.sh
# Ctrl+B roi D de detach
# Quay lai: tmux attach -t ste
```

---

## 6. Truy cập giao diện từ máy cá nhân

### Cách A — Mở firewall cổng 7860

Trên server:

```bash
sudo ufw allow 7860/tcp
sudo ufw status
```

Trên máy bạn: mở trình duyệt → `http://<IP-server>:7860`

> Nhà cung cấp VPS (AWS, GCP, RunPod…) thường có **Security Group / Firewall** riêng trên web — cần mở **TCP 7860** ở đó nữa.

### Cách B — SSH tunnel (an toàn, không cần mở cổng public)

Trên **máy cá nhân**:

```bash
ssh -L 7860:localhost:7860 user@<IP-server>
```

Giữ phiên SSH mở, mở trình duyệt: `http://localhost:7860`

### Cách C — Link Gradio public (`*.gradio.live`)

```bash
GRADIO_SHARE=1 ./start_linux.sh
# hoac: ./start_linux.sh --share
```

Xem link trong `logs/ui.log` (dạng `https://xxxxx.gradio.live`). Không cần mở firewall; phù hợp thuê server theo giờ.

---

## 7. Cấu hình lồng tiếng trên giao diện web

Tab **Cài đặt** → mục **Long tieng (OmniVoice)**:

| Mục | Giá trị gợi ý |
|-----|----------------|
| Bật lồng tiếng | Bật |
| OmniVoice server URL | `http://127.0.0.1:7861` (mặc định — đúng khi chạy cùng server) |
| Giọng mẫu | Upload 3–10 giây |
| Tỉ lệ trộn | Gốc 0.3 / Dịch 0.7 |

Đảm bảo `run_omnivoice.sh` đã chạy và log có `Model loaded.` trước khi xử lý video có lồng tiếng.

---

## 8. Kiểm tra GPU trước khi chạy thật

```bash
conda activate ste
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import paddle; paddle.utils.run_check()"
conda deactivate
```

---

## 9. Tối ưu chi phí thuê server theo giờ

1. **Snapshot / image** sau khi cài xong lần đầu — lần sau khởi tạo VPS từ snapshot, chỉ cần `./start_linux.sh`.
2. **Cache `models/`** trên ổ persistent hoặc cloud — đừng tải lại mỗi lần thuê máy mới.
3. Giữ **cả hai service chạy nền** — model nạp 1 lần, xử lý nhiều video không phải load lại.
4. VRAM thấp: chạy lần lượt — xong bước xóa phụ đề rồi mới bật OmniVoice (`./run_omnivoice.sh`).
5. Giảm `num_step` OmniVoice (16–24) trên UI nếu chấp nhận chất lượng thấp hơn một chút.

---

## 10. Xử lý sự cố thường gặp

| Triệu chứng | Cách xử lý |
|-------------|------------|
| Không vào được `:7860` | Kiểm tra `ufw` + Security Group VPS; thử SSH tunnel hoặc `GRADIO_SHARE=1` |
| Lồng tiếng báo lỗi kết nối | `curl http://127.0.0.1:7861` trên server; chạy lại `./run_omnivoice.sh` |
| `Model loaded.` không xuất hiện | Xem `logs/omnivoice.log`; thiếu VRAM → tắt service khác hoặc `OMNIVOICE_NO_ASR=1 ./run_omnivoice.sh` |
| MoviePy / ImageMagick lỗi | `sudo apt install imagemagick`; đảm bảo `magick` hoặc `convert` trong PATH |
| OCR/Paddle lỗi GPU / `libcudnn.so` | Chạy `./scripts/fix_ste_env.sh`; hoặc xem [index.md](index.md) CASE A/B |
| `Out of memory` / `ResourceExhaustedError` GPU | OmniVoice + OCR cùng GPU 0 → restart: `./stop_linux.sh && ./start_linux.sh` (tự tách GPU 1 cho OmniVoice nếu có 2 card); hoặc tắt lồng tiếng khi xử lý |
| `CondaToSNonInteractiveError` | `conda tos accept` (xem mục 2) hoặc chạy lại `./setup_linux_gpu.sh` |
| Gradio `HfFolder` / `pydantic` / `jinja2` | Chạy `./scripts/fix_ste_env.sh` (đã ghim trong `requirements-ui.txt`) |

---

## 11. Tóm tắt lệnh nhanh

```bash
# Cai lan dau
./setup_linux_gpu.sh
# (tai models/, sua config.yaml)

# Chay
./start_linux.sh              # hoac ./start_linux.sh --share

# Truy cap
# http://<IP-server>:7860  HOAC  ssh -L 7860:localhost:7860 ...

# Dung
./stop_linux.sh
```

**Cổng bạn cần nhớ:** chỉ **7860** là giao diện public; **7861** là API nội bộ, không expose ra internet.
