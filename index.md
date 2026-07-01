# Hướng dẫn cài đặt SubErase-Translate-Embed (Windows & Linux, có GPU NVIDIA)

> Repo này dùng GPU để chạy AI (OCR nhận diện chữ bằng PaddleOCR + STTN xóa phụ đề + PyTorch).
> Hướng dẫn được chia theo **đời card đồ họa** vì đây là nguyên nhân chính gây lỗi cài đặt.

---

## 0. Hiểu nhanh trước khi cài (rất quan trọng)

Hai khái niệm hay bị nhầm:

1. **Dòng `CUDA Version: 13.x` trong `nvidia-smi` KHÔNG phải toolkit bạn phải cài.**
   Đó chỉ là *mức CUDA tối đa mà driver hỗ trợ*. Các gói `torch`/`paddlepaddle` cài qua `pip`
   đã **gói sẵn CUDA runtime riêng** bên trong, nên driver hiển thị 13.2/13.3 vẫn chạy bình thường.
   => Bạn KHÔNG cần hạ driver hay cài đúng CUDA 12.x.

2. **Lỗi PaddlePaddle bạn gặp là do "kiến trúc card" (compute capability), không phải do CUDA driver.**
   Mỗi đời card có một mã kiến trúc `sm_xx`. Gói AI phải được build sẵn cho đúng mã đó:
   - Card 30/40-series: `sm_86` / `sm_89` -> bản PaddlePaddle/PyTorch phổ thông hỗ trợ sẵn.
   - Card 50-series (5090/5080/5070...): `sm_120` -> bản phổ thông CHƯA build sẵn,
     nên báo lỗi kiểu: `compiled for 75 80 86 89, but your current GPU is 120`
     hoặc `no kernel image is available for execution on the device`.

> Thuật ngữ:
> - **CUDA**: nền tảng của NVIDIA cho phép phần mềm dùng GPU để tính toán.
> - **conda**: công cụ tạo "môi trường ảo" Python tách biệt, tránh xung đột thư viện.
> - **compute capability / sm_xx**: "mã đời" của GPU. Phần mềm GPU phải build đúng mã này mới chạy được.

### Cách xác định card của bạn thuộc CASE nào

Mở PowerShell/CMD (Windows) hoặc Terminal (Linux), gõ:

```bash
nvidia-smi
```

Nhìn tên card ở cột "Name", rồi đối chiếu:

| Đời card (ví dụ)                         | Kiến trúc | Mã sm     | Làm theo |
| ---------------------------------------- | --------- | --------- | -------- |
| RTX 3060/3070/3080/3090, A10, A100...     | Ampere    | sm_80/86  | CASE A   |
| RTX 4070/4080/4090, 4090 Ti, L4, L40...   | Ada       | sm_89     | CASE A   |
| RTX 5060/5060 Ti/5070/5080/5090, 5090 Ti  | Blackwell | sm_120    | CASE B   |

> Nếu chưa chắc, sau khi cài xong PyTorch bạn có thể kiểm tra ở mục 7.
> Card báo "sm_120" thì bắt buộc đi theo **CASE B**.

---

## 1. Chuẩn bị chung (cả Windows lẫn Linux)

Bạn cần cài sẵn:

- **NVIDIA Driver mới nhất** -> tải tại trang [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx).
  (Driver mới càng tốt; card 50-series cần driver mới hỗ trợ CUDA 12.8+.)
- **Miniconda/Anaconda** (tạo môi trường Python riêng):
  - Windows: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) hoặc [Anaconda](https://www.anaconda.com/download/success).
  - Linux: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh`
- **Git** -> [git-scm.com](https://git-scm.com/downloads).
- **Git LFS** (để tải model lớn, nên có nhưng không bắt buộc).

Kiểm tra GPU đã nhận chưa:

```bash
nvidia-smi
```

Nếu hiện bảng thông tin card là OK (bỏ qua con số CUDA Version như đã giải thích ở mục 0).

---

## 2. Tải mã nguồn (Clone Repository)

```bash
git clone --recursive https://github.com/chenwr727/SubErase-Translate-Embed.git
cd SubErase-Translate-Embed
```

Lưu ý: phải có `--recursive` vì repo dùng submodule (module con). Nếu lỡ clone thiếu:

```bash
git submodule update --init --recursive
```

---

## 3. Tạo môi trường Python

```bash
conda create -n ste python=3.10 -y
conda activate ste
```

> Giữ Python 3.10 vì các wheel GPU bên dưới đều có sẵn bản cho 3.10.

---

## 4. Cài PyTorch + PaddlePaddle theo ĐÚNG CASE của bạn

> Quan trọng về thứ tự: hãy cài **torch + paddlepaddle GPU TRƯỚC**, rồi mới
> `pip install -r requirements.txt` (mục 5). File `requirements.txt` đã được comment
> 2 dòng `torch`/`torchvision` để không vô tình cài đè lại bản CPU.

### CASE A — Card 30/40-series (sm_86 / sm_89), ví dụ RTX 3090, 4090, 4090 Ti

PyTorch (bản GPU CUDA 12.1):

```bash
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
```

PaddlePaddle GPU (CUDA 12.0):

- Windows:

```bash
python -m pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

- Linux:

```bash
python -m pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

### CASE B — Card 50-series (sm_120), ví dụ RTX 5070/5080/5090, 5090 Ti

> Đây là trường hợp gây lỗi mà bạn đang gặp. Cần bản PyTorch và PaddlePaddle MỚI hơn,
> được build cho `sm_120` (CUDA 12.8/12.9).

PyTorch (bản GPU cho sm_120). QUAN TRỌNG: hãy cài bản **CUDA 12.9 (cu129)** để KHỚP
với PaddlePaddle 12.9 ở dưới, tránh lỗi xung đột DLL (xem mục "Lỗi WinError 127" cuối CASE B):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

> Nếu link cu129 báo không có gói, dùng bản nightly:
> `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu129`
> (Bản cu128 vẫn chạy được sm_120, nhưng dễ ĐÁ NHAU với Paddle 12.9 -> nên ưu tiên cu129.)

PaddlePaddle GPU:

- **Windows**: bản phổ thông CHƯA hỗ trợ 50-series, phải dùng wheel chuyên dụng (CUDA 12.9)
  của đội PaddleX. Chọn đúng theo phiên bản Python (ở đây là 3.10 = `cp310`):

```bash
python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp310-cp310-win_amd64.whl
```

> Nếu dùng Python khác, đổi `cp310` thành `cp39`/`cp311`/`cp312` tương ứng.
> Trang gốc (luôn cập nhật link mới nhất): https://paddlepaddle.github.io/PaddleX/latest/en/installation/paddlepaddle_install.html
> Lưu ý: wheel này có ghi chú "lỗi khi TRAIN model nhận dạng". App này CHỈ chạy
> inference (nhận diện), KHÔNG train, nên không bị ảnh hưởng.

- **Linux**: thử bản chính thức CUDA 12.9 trước:

```bash
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

Nếu vẫn báo lỗi kiến trúc `sm_120` (vì wheel chính thức chưa bật sẵn), dùng wheel cộng đồng
build sẵn cho `sm_120` (CUDA 13.0, Python 3.10):

```bash
# Tải wheel ở trang Releases rồi cài (xem README repo dưới đây)
# https://github.com/horhe-dvlp/paddlepaddle-sm120-wheels
pip install paddlepaddle_gpu-3.3.0.dev20251209-cp310-cp310-linux_x86_64.whl --force-reinstall
```

### Lỗi WinError 127 (cublas64_12.dll) — xung đột giữa PyTorch và PaddlePaddle

Nếu khi chạy bạn gặp lỗi kiểu:

```
OSError: [WinError 127] The specified procedure could not be found.
Error loading "...\paddle\..\nvidia\cublas\bin\cublas64_12.dll" or one of its dependencies.
```

Nguyên nhân: app này dùng CẢ PyTorch (xóa phụ đề STTN) lẫn PaddlePaddle (OCR) trong cùng
một tiến trình. Hai bên cùng cần file `cublas64_12.dll` nhưng KHÁC phiên bản CUDA
(vd torch cu128 = CUDA 12.8, còn paddle = CUDA 12.9) -> Windows chỉ nạp được 1 bản,
bên còn lại gọi hàm không có -> WinError 127.

Cách sửa (làm cho 2 bên dùng CÙNG CUDA 12.9):

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

Kiểm tra lại, cả 2 lệnh dưới phải chạy KHÔNG lỗi:

```bash
python -c "import torch; print(torch.version.cuda, torch.cuda.get_arch_list())"
python -c "import torch; import paddle; paddle.utils.run_check()"
```

> Nếu vì lý do nào đó buộc phải giữ torch cu128, giải pháp dự phòng là chạy OCR ở chế độ
> CPU (chỉ STTN dùng GPU) để Paddle không nạp DLL CUDA nữa -> hết xung đột (đổi lại OCR chậm hơn).

Sau khi sửa cublas, đôi khi còn lỗi tương tự với file **cuDNN** (vd `cudnn_engines_precompiled64_9.dll`).
Cách xử lý (cho paddle dùng chung bộ cuDNN của torch thay vì bộ riêng của nó):

Cách 1 - gỡ bộ cuDNN riêng của paddle (đơn giản nhất):

```bash
pip uninstall -y nvidia-cudnn-cu12
python -c "import torch; import paddle; paddle.utils.run_check()"
```

Cách 2 - nếu Cách 1 báo thiếu cuDNN, cài lại rồi copy đè DLL cuDNN của torch sang paddle:

```bat
pip install nvidia-cudnn-cu12
copy /Y "%CONDA_PREFIX%\Lib\site-packages\torch\lib\cudnn*64_9.dll" "%CONDA_PREFIX%\Lib\site-packages\nvidia\cudnn\bin\"
```

---

## 5. Cài phần thư viện còn lại

```bash
pip install -r requirements.txt
```

Nếu dùng giao diện web (Gradio) + upload Google Drive, cài thêm:

```bash
pip install -r requirements-ui.txt
```

---

## 6. Tải các Model AI

Tạo thư mục `models` trong dự án rồi tải:

- **PaddleOCR** (giải nén file `.tar`):
  - [det](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_server_infer.tar) — nhận diện vị trí chữ
  - [rec](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_infer.tar) — đọc nội dung chữ
- **STTN** (model xóa phụ đề): tải tại [Google Drive](https://drive.google.com/file/d/1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv/view?usp=sharing)

Sắp xếp đúng cấu trúc:

```
models
├── ch_PP-OCRv4_det_server_infer
├── ch_PP-OCRv4_rec_server_infer
└── sttn.pth
```

Hai file `.tar` sau khi tải cần giải nén thành thư mục (Windows có thể dùng 7-Zip; Linux dùng `tar -xf file.tar`).
Riêng `sttn.pth` để nguyên file.

---

## 7. Tạo file cấu hình

- Windows (CMD): `copy config-template.yaml config.yaml`
- Windows (Git Bash) / Linux: `cp config-template.yaml config.yaml`

Sau đó mở `config.yaml` điền API key ChatGPT (phần dịch) và chỉnh tùy chọn khác.

---

## 8. Cài phần mềm hỗ trợ (FFmpeg + ImageMagick)

- **FFmpeg**: xử lý cắt/ghép video, tách khung hình.
- **ImageMagick**: thư viện `moviepy` cần nó để render/ghép phụ đề.

### Windows

Mở PowerShell, chạy:

```powershell
winget install Gyan.FFmpeg
winget install ImageMagick.ImageMagick
```

> `winget` là trình quản lý cài đặt có sẵn trên Windows 10/11, tự thêm vào PATH.
> Cài xong, đóng PowerShell và mở lại để PATH nạp mới.

Bật UTF-8 (tránh lỗi đọc file tiếng Trung):

```cmd
set PYTHONUTF8=1
```

### Linux

```bash
sudo apt update
sudo apt install -y ffmpeg imagemagick
```

> Trên Windows KHÔNG cần cài `gcc` như README gốc (đó là cho Linux).
> Nếu `moviepy` báo không tìm thấy ImageMagick, trỏ đường dẫn tới `magick.exe`
> (thường ở `...\envs\ste\Library\bin\magick.exe`) trong cấu hình moviepy.

Kiểm tra:

```bash
ffmpeg -version
magick -version
```

---

## 9. Kiểm tra GPU đã sẵn sàng (khuyến nghị làm trước khi chạy thật)

Trong môi trường `ste`, chạy:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Card:', torch.cuda.get_device_name(0)); print('Arch:', torch.cuda.get_arch_list())"
```

- `CUDA: True` + tên card -> PyTorch thấy GPU.
- Với CASE B: trong `Arch:` phải có `sm_120`. Nếu KHÔNG có -> bạn đang cài nhầm bản torch cũ (cu121), cài lại theo CASE B.

Kiểm tra PaddlePaddle:

```bash
python -c "import paddle; paddle.utils.run_check()"
```

In ra `PaddlePaddle is installed successfully!` là OK.

---

## 10. Chạy thử

```bash
python main.py --video input_video.mp4 --language Vietnamese
```

Trong đó:

- `--video`: đường dẫn file video đầu vào.
- `--language`: ngôn ngữ muốn dịch sang (ví dụ `Vietnamese`, `English`).
- `--delete` (tùy chọn): tự xóa thư mục tạm sau khi xử lý xong.

Chạy giao diện web (Gradio):

- **Windows**: chạy `run_ui.bat` (cổng **7860**)
- **Linux server (headless, GPU)**: xem **[LINUX_SERVER.md](LINUX_SERVER.md)** — hướng dẫn đầy đủ terminal, firewall, cổng **7860** (public) và **7861** (nội bộ)

```bash
chmod +x setup_linux_gpu.sh start_linux.sh
./setup_linux_gpu.sh    # cài lần đầu
./start_linux.sh        # khởi động → http://<IP-server>:7860
```

---

## 11. Tối ưu thời gian khi thuê VPS theo giờ (tránh dựng lại từ đầu mỗi lần)

Mỗi lần thuê VPS mới mà cài lại từ đầu (tải model vài GB, cài torch/paddle...) rất tốn thời gian =>
tiền thuê theo giờ bị lãng phí. Dưới đây là các cách tối ưu theo từng hệ điều hành.

### A. Linux — KHUYẾN NGHỊ MẠNH: dùng Docker

Build 1 lần thành 1 image (đã có sẵn conda env + torch + paddle + ffmpeg + imagemagick),
đẩy lên registry (Docker Hub / GHCR). Lần sau chỉ cần kéo về:

```bash
# Lần đầu: build và push (xem Dockerfile kèm repo)
docker build -t <user>/ste:cuda128 .
docker push <user>/ste:cuda128

# Mỗi VPS mới: kéo về và chạy (vài phút thay vì cả tiếng)
docker pull <user>/ste:cuda128
docker run --gpus all -it -v $PWD/models:/app/models -v $PWD/config.yaml:/app/config.yaml <user>/ste:cuda128
```

> Mẹo: KHÔNG nhồi model nặng vào image. Để model + `config.yaml` ở **volume/ổ lưu trữ bền (persistent)**
> rồi mount vào container. Image chỉ chứa thư viện -> nhẹ và pull nhanh.

### B. Windows — không có Docker GPU ngon, dùng các cách sau (ưu tiên từ trên xuống)

1. **Snapshot / Custom Image của nhà cung cấp (nhanh nhất):**
   Cài hoàn chỉnh 1 lần rồi bấm "Create Image/Snapshot". Lần sau tạo VPS từ snapshot đó là
   chạy được ngay, gần như không phải cài lại.

2. **`conda-pack` — đóng gói nguyên môi trường thành 1 file:**

```bash
# Trên máy đã cài xong:
pip install conda-pack
conda pack -n ste -o ste-env.tar.gz
# Upload ste-env.tar.gz lên cloud (Google Drive / S3)

# Trên VPS mới:
mkdir ste && tar -xzf ste-env.tar.gz -C ste
ste\Scripts\activate
conda-unpack
```

3. **Script tự động + cache:** dùng `setup.bat` (kèm repo) để cài một phát.
   Đồng thời cache lại để khỏi tải lại mỗi lần:
   - Cache pip wheels: `pip download -r requirements.txt -d wheels` rồi lần sau
     `pip install --no-index --find-links wheels -r requirements.txt`.
   - Cache **models** (nặng nhất): lưu thư mục `models/` lên cloud hoặc ổ lưu trữ bền,
     lần sau chỉ copy về.

### C. Nguyên tắc chung cho cả hai

- Luôn tách 3 thứ ra khỏi máy ephemeral (máy bị xóa khi hết giờ): **models**, **config.yaml**, **cache wheel/env**.
- Để chúng ở ổ lưu trữ bền (persistent volume) hoặc cloud, mỗi lần chỉ "gắn lại" thay vì tải lại.

---

## 12. Cơ chế "chạy tiếp" (resume) khi bị lỗi giữa chừng

Pipeline gồm các bước: tách khung hình -> OCR -> tạo phụ đề -> xóa phụ đề -> dịch -> chèn phụ đề.
Code đã được bổ sung để **chạy lại sẽ tận dụng kết quả cũ thay vì làm lại từ đầu** (tiết kiệm thời gian/tiền VPS):

- **OCR**: nếu đã có `<tên video>_ocr_check.json` + `<tên video>_ocr_center.json` thì đọc lại,
  KHÔNG quét lại bằng PaddleOCR (đây là bước tốn thời gian nhất).
- **Xóa phụ đề (STTN) + tách khung hình**: nếu đã có video `<tên video>_output.<ext>` (video đã xóa phụ đề)
  cùng với 2 file OCR ở trên thì BỎ QUA luôn cả 3 bước nặng (tách khung hình, OCR, xóa phụ đề).
- **Dịch**: nếu file phụ đề đã dịch tồn tại thì không gọi lại API (vốn đã có sẵn từ trước).
- **Chèn phụ đề**: nếu video kết quả cuối `<tên video>_<ngôn ngữ>.<ext>` đã tồn tại thì bỏ qua.

> Muốn chạy lại HOÀN TOÀN từ đầu (ví dụ sau khi đổi cấu hình OCR): xóa các file trung gian
> `*_ocr*.json`, video `*_output.*`, và thư mục khung hình tạm (trùng tên file video) trước khi chạy.

> Lưu ý: cơ chế resume này áp dụng cho cách chạy bằng dòng lệnh `python main.py`.
> Bản giao diện web `app.py` tạo thư mục mới theo thời gian cho mỗi lần chạy nên sẽ luôn chạy mới.

---

## 13. (Tuỳ chọn) Lồng tiếng dịch bằng OmniVoice

Tính năng này tạo **giọng đọc bản dịch** (có thể clone giọng từ 1 đoạn mẫu) rồi
**trộn với tiếng gốc** (mặc định gốc 30% + giọng dịch 70%) và ghép vào video đã
cháy phụ đề. Kết quả: video vừa có phụ đề, vừa được lồng tiếng dịch khớp đúng timeline.

### 13.1. Vì sao phải tạo môi trường RIÊNG?

OmniVoice cần `torch 2.8 + CUDA 12.8` và Python 3.12, **xung đột** với
`paddleocr` của pipeline xoá phụ đề. Vì vậy ta chạy OmniVoice như 1 **service
riêng** (môi trường conda `omnivoice`), còn pipeline chính (`ste`) gọi sang qua API.

```mermaid
flowchart LR
    UI["app.py (env ste, cong 7860)"] -->|"goi API gradio_client"| SV["audio.py (env omnivoice, cong 7861)"]
    SV -->|"tts.wav khop timeline"| UI
```

### 13.2. Tạo môi trường `omnivoice`

```bash
conda create -n omnivoice python=3.12 -y
conda activate omnivoice
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 torchvision==0.23.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-omnivoice.txt
```

> Card 50-series (sm_120) dùng đúng `cu128` ở trên là phù hợp. Hai môi trường
> `ste` và `omnivoice` là 2 tiến trình tách biệt nên KHÔNG bị lỗi cublas/cuDNN
> như khi nhồi chung (xem mục 4).

### 13.3. Bổ sung cho môi trường chính `ste`

Pipeline chính cần `gradio_client` để gọi sang OmniVoice (đã thêm sẵn vào
`requirements-ui.txt`):

```bash
conda activate ste
pip install -r requirements-ui.txt
```

### 13.4. Chạy

Mở **2 cửa sổ** (mỗi cái 1 môi trường):

- Cửa sổ 1 — service lồng tiếng: chạy `run_omnivoice.bat` (mở cổng 7861).
  Đợi đến khi thấy dòng `Model loaded.` thì service đã sẵn sàng.
- Cửa sổ 2 — giao diện chính: chạy `run_ui.bat` (mở cổng 7860).

Trong giao diện web, tab **Cài đặt** -> mục **Long tieng (OmniVoice)**:

- Bật "Bat long tieng".
- "OmniVoice server URL": để mặc định `http://127.0.0.1:7861` (nếu chạy cùng máy).
- "Giong mau": upload 1 đoạn 3-10s để clone giọng (bỏ trống = giọng mặc định).
- "Loi thoai cua giong mau": gõ đúng lời của đoạn mẫu (giúp clone chuẩn hơn; có
  thể bỏ trống nếu service bật ASR tự nhận diện).
- Chỉnh tỉ lệ trộn nếu muốn (mặc định 0.3 / 0.7).

### 13.5. Tối ưu tốc độ / chi phí GPU

- Giữ service `omnivoice` **chạy thường trực** để không phải nạp model lại mỗi video.
- Hạ `num_step` (vd 16-24) nếu chấp nhận chất lượng thấp hơn chút để chạy nhanh hơn.
- Nếu LUÔN nhập sẵn "Reference Text": thêm `--no-asr` vào `run_omnivoice.bat`
  để khỏi nạp Whisper -> nhẹ VRAM, khởi động nhanh hơn.
- VRAM hạn chế: chạy `ste` và `omnivoice` **lần lượt** thay vì song song (xong
  bước xoá phụ đề rồi mới bật service lồng tiếng).

### 13.6. Đảm bảo "không trượt timeline"

Service dùng chế độ **Hard sync**: mỗi câu được đặt đúng mốc thời gian của phụ đề
và co/giãn audio cho vừa khít ô (giữ cao độ bằng `librosa`). Nhờ vậy tổng độ dài
audio luôn bằng độ dài video -> không bao giờ lệch giờ. Câu thoại quá dài so với
ô thời gian sẽ được đọc nhanh hơn (đánh đổi để giữ đúng nhịp video).
