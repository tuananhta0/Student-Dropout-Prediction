"""Tạo file .env bị Git bỏ qua để cùng một khóa Gemini phục vụ máy demo.

Không bao giờ in khóa hoặc đẩy lên Git. Chia sẻ file qua kênh nhóm riêng.
"""
import getpass
from pathlib import Path
from core.data_pipeline import ROOT


def save_env(path, key):
    path=Path(path)
    if path.exists(): raise FileExistsError('.env đã tồn tại; không ghi đè cấu hình nhóm')
    if not key or len(key)<20 or any(c in key for c in '\r\n\x00\"\''):
        raise ValueError('Khóa API không hợp lệ')
    path.write_text('GEMINI_API_KEY="'+key+'"\nCMC_RAG_CLOUD=1\nCMC_EMBED_MODEL=gemini-embedding-001\nCMC_LLM_MODEL=gemini-3.1-flash-lite\n',encoding='utf-8')
    try:path.chmod(0o600)
    except OSError:pass
    return path


def main():
    secret=getpass.getpass('Nhập API key DÙNG CHUNG (không hiển thị): ')
    try:path=save_env(ROOT/'.env',secret)
    except (FileExistsError,ValueError) as exc:
        print('Không tạo file:',type(exc).__name__)
        return 1
    print('Đã tạo file .env bị .gitignore loại trừ. Chia sẻ nội bộ, không commit; thử Gemini bằng scripts.benchmark_rag --cloud --probe.')
    return 0

if __name__=='__main__':raise SystemExit(main())
