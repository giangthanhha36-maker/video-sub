from openai import OpenAI

from modules.config import load_config

# ============================================================================
# [GIU NGUYEN - code cu] Tao client ngay khi import file.
# Nhuoc diem: bat buoc phai co config.yaml hop le luc import, va khong the
# thay doi api_key/model luc chay (vd: nhap tu giao dien Gradio).
# Da comment lai de tham khao, thay bang co che "runtime override" ben duoi.
# ----------------------------------------------------------------------------
# config = load_config()
# client = OpenAI(
#     api_key=config["translation"]["api_key"],
#     base_url=config["translation"]["api_base_url"],
# )
# ============================================================================


# ============================================================================
# [THAY DOI - code moi] Cho phep ghi de cau hinh luc chay (runtime).
# Giao dien (tab "Cai dat") se goi set_runtime(...) de truyen API key / base_url
# / model do nguoi dung nhap. Cac gia tri nay chi nam trong bo nho phien chay,
# KHONG ghi ra file -> an toan voi thong tin nhay cam.
# Neu giao dien khong truyen gi, code se tu doc lai tu config.yaml nhu cu.
# ============================================================================
_runtime = {"api_key": None, "api_base_url": None, "model": None}


def set_runtime(api_key: str = None, api_base_url: str = None, model: str = None):
    """Cho phep giao dien ghi de API key / base_url / model luc chay."""
    if api_key:
        _runtime["api_key"] = api_key
    if api_base_url:
        _runtime["api_base_url"] = api_base_url
    if model:
        _runtime["model"] = model


def _resolve_settings():
    """Tra ve (api_key, base_url, model): uu tien runtime, neu khong co thi lay tu config.yaml."""
    cfg = load_config()
    api_key = _runtime["api_key"] or cfg["translation"]["api_key"]
    base_url = _runtime["api_base_url"] or cfg["translation"]["api_base_url"]
    model = _runtime["model"] or cfg["translation"].get("model", "gpt-4o-mini")
    return api_key, base_url, model


def get_client(api_key: str = None, base_url: str = None) -> OpenAI:
    """Tao OpenAI client. Neu khong truyen tham so se dung cau hinh runtime/config."""
    if api_key is None or base_url is None:
        r_key, r_base, _ = _resolve_settings()
        api_key = api_key or r_key
        base_url = base_url or r_base
    return OpenAI(api_key=api_key, base_url=base_url)


def get_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = None,
    temperature: float = 0.3,
) -> str:
    """
        Generate a completion using the OpenAI API.

    Args:
        prompt (str): The user's prompt or query.
        system_message (str, optional): The system message to set the context for the assistant.
            Defaults to "You are a helpful assistant.".
        model (str, optional): The name of the OpenAI model to use for generating the completion.
            Defaults to None -> tu lay tu runtime/config.
        temperature (float, optional): The sampling temperature for controlling the randomness of the generated text.
            Defaults to 0.3.
        json_mode (bool, optional): Whether to return the response in JSON format.
            Defaults to False.

    Returns:
        Union[str]: The generated completion. returns the generated text as a string.
    """

    # [THAY DOI] Lay cau hinh luc chay va tao client moi cho moi lan goi.
    api_key, base_url, default_model = _resolve_settings()
    if model is None:
        model = default_model
    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        top_p=1,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content
