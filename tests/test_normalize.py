"""Test bước chuẩn bị corpus. Chạy được ngoài Colab, không cần soffice hay pdftotext."""

from __future__ import annotations

import textwrap

import pytest

from tools.normalize_raw import (
    ExtractionError,
    NormalizeReport,
    check_extraction_quality,
    detect_format,
    diacritic_ratio,
    normalize,
    outline,
    outline_tree,
    render_frontmatter,
    select_articles,
    word_count,
    write_corpus_file,
)


def body(text: str) -> str:
    return textwrap.dedent(text).strip("\n")


# --- nối dòng bị wrap -----------------------------------------------------
def test_noi_dong_wrap():
    """Dòng thường bị ngắt cứng được nối; dòng mở đơn vị cấu trúc thì không."""
    raw = body(
        """
        Điều 1. Phạm vi điều chỉnh
        Nghị định này quy định về bảo vệ dữ liệu cá
        nhân và trách nhiệm của cơ quan có liên quan
        1. Khoản một bắt đầu ở đây, không được nối vào dòng trên
        b) điểm b cũng không được nối
        Chương III
        Điều 5. Điều khoản thi hành
        """
    )
    text, report = normalize(raw)
    lines = text.strip().split("\n")

    assert "dữ liệu cá nhân và trách nhiệm" in text
    assert report.n_lines_joined == 1
    assert any(line.startswith("1. Khoản một") for line in lines)
    assert any(line.startswith("b) điểm b") for line in lines)
    assert any(line.strip() == "Chương III" for line in lines)
    assert any(line.startswith("Điều 5.") for line in lines)


def test_khong_noi_khi_dong_truoc_ket_thuc_bang_dau_cham():
    raw = body(
        """
        Điều 1. Phạm vi
        Câu thứ nhất kết thúc ở đây.
        Câu thứ hai là một dòng riêng
        """
    )
    text, report = normalize(raw)
    assert "kết thúc ở đây. Câu thứ hai" not in text
    assert report.n_lines_joined == 0


def test_tieu_de_dieu_khong_nuot_dong_ke_tiep():
    """Nếu tiêu đề nuốt câu đầu thì ingest sẽ thấy Điều rỗng — hỏng toàn tập."""
    raw = body(
        """
        Điều 1. Phạm vi điều chỉnh
        Nghị định này quy định chi tiết một số điều.
        """
    )
    text, _ = normalize(raw)
    assert text.split("\n")[0] == "Điều 1. Phạm vi điều chỉnh"


def test_tieu_de_chuong_khong_nuot_ten_chuong():
    """"Chương I QUY ĐỊNH CHUNG" trên một dòng sẽ không khớp CHAPTER_RE của ingest."""
    raw = body(
        """
        Chương I
        QUY ĐỊNH CHUNG
        Điều 1. Phạm vi
        Nội dung.
        """
    )
    text, _ = normalize(raw)
    assert text.split("\n")[0] == "Chương I"
    assert text.split("\n")[1] == "QUY ĐỊNH CHUNG"


def test_noi_dong_trong_khoan_van_hoat_dong():
    """Khoản bị wrap vẫn phải nối — chốt tiêu đề không được chặn nhầm ca này."""
    raw = body(
        """
        Điều 1. Phạm vi
        1. Khoản này bị ngắt giữa chừng vì lề
        trang hẹp và cần được nối lại.
        """
    )
    text, report = normalize(raw)
    assert "vì lề trang hẹp" in text
    assert report.n_lines_joined == 1


# --- dòng nhiễu -----------------------------------------------------------
def test_xoa_header_lap():
    raw = body(
        """
        Điều 1. Phạm vi
        CÔNG BÁO SỐ 123
        Nội dung một.
        CÔNG BÁO SỐ 123
        Nội dung hai.
        CÔNG BÁO SỐ 123
        Nội dung ba.
        CÔNG BÁO SỐ 123
        Nội dung bốn.
        Ghi chú chỉ lặp hai lần.
        Ghi chú chỉ lặp hai lần.
        """
    )
    text, report = normalize(raw)
    assert "CÔNG BÁO SỐ 123" not in text
    assert text.count("Ghi chú chỉ lặp hai lần.") == 2
    assert report.n_lines_dropped >= 4


def test_xoa_so_trang():
    raw = body(
        """
        Điều 1. Phạm vi
        Nội dung của điều.
        12
        Nội dung tiếp theo.
        """
    )
    text, _ = normalize(raw)
    assert "\n12\n" not in text
    assert "Nội dung tiếp theo." in text


# --- ký tự ----------------------------------------------------------------
def test_nbsp_va_soft_hyphen():
    raw = "Điều 1. Phạm vi\nDữ\u00a0liệu cá nhân bảo\u00advệ theo “quy định” – khoản 1.\n"
    text, _ = normalize(raw)
    assert "\u00a0" not in text
    assert "\u00ad" not in text
    assert "Dữ liệu" in text
    assert "bảovệ" in text
    assert '"quy định"' in text
    assert "”" not in text and "–" not in text


def test_gop_space_va_dong_trong():
    raw = "Điều 1. Phạm vi\nNội   dung    thừa space.   \n\n\n\nĐiều 2. Khác\nNội dung.\n"
    text, _ = normalize(raw)
    assert "Nội dung thừa space." in text
    assert "\n\n\n" not in text


# --- cắt phạm vi ----------------------------------------------------------
def test_cat_truoc_dieu_1():
    raw = body(
        """
        CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
        Độc lập - Tự do - Hạnh phúc
        Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam.
        Căn cứ Luật Tổ chức Chính phủ.
        Điều 1. Phạm vi điều chỉnh
        Nội dung của điều một.
        """
    )
    text, _ = normalize(raw)
    assert text.startswith("Điều 1. Phạm vi điều chỉnh")
    assert "CỘNG HÒA" not in text
    assert "Căn cứ" not in text


def test_cat_truoc_chuong_dau_tien():
    raw = body(
        """
        CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
        Chương I
        QUY ĐỊNH CHUNG
        Điều 1. Phạm vi
        Nội dung.
        """
    )
    text, _ = normalize(raw)
    assert text.startswith("Chương I")


def test_cat_tu_phu_luc():
    raw = body(
        """
        Điều 1. Phạm vi
        Nội dung của điều một.
        Điều 2. Hiệu lực
        Nội dung của điều hai.
        PHỤ LỤC
        Mẫu số 01 kèm theo.
        """
    )
    text, report = normalize(raw)
    assert "PHỤ LỤC" not in text
    assert "Mẫu số 01" not in text
    assert "Điều 2. Hiệu lực" in text
    assert any("cắt từ dòng" in w for w in report.warnings)


def test_cat_tu_noi_nhan():
    raw = "Điều 1. Phạm vi\nNội dung.\nNơi nhận:\n- Như trên;\n"
    text, _ = normalize(raw)
    assert "Nơi nhận" not in text
    assert "Như trên" not in text


def test_cat_tu_dong_gach_duoi():
    """Dòng kẻ ngăn trước phần công bố; bước nối dòng đã dính câu công bố vào nó."""
    raw = body(
        """
        Điều 1. Phạm vi
        Nội dung của điều một.
        ______________________________________
        Luật này được Quốc hội thông qua ngày 26 tháng 6 năm 2025.
        """
    )
    text, _ = normalize(raw)
    assert "____" not in text
    assert "thông qua ngày" not in text
    assert "Nội dung của điều một." in text


def test_select_articles_bo_muc_mo_coi():
    """Tiêu đề Mục không còn Điều nào bên dưới sẽ tạo lỗ thủng coverage."""
    raw = body(
        """
        Điều 1. Một
        Nội dung một.
        Mục 4
        BIỆN PHÁP BẢO ĐẢM
        Điều 9. Chín
        Nội dung chín.
        """
    )
    lines = select_articles(raw, {"articles": [1]}).split("\n")
    assert "Mục 4" not in lines
    assert "BIỆN PHÁP BẢO ĐẢM" not in lines
    assert "Điều 1. Một" in lines


def test_select_articles_giu_tieu_de_muc_khi_con_dieu():
    raw = body(
        """
        Điều 1. Một
        Nội dung một.
        Mục 4
        BIỆN PHÁP BẢO ĐẢM
        Điều 9. Chín
        Nội dung chín.
        """
    )
    lines = select_articles(raw, {"articles": [9]}).split("\n")
    assert "Mục 4" in lines
    assert "BIỆN PHÁP BẢO ĐẢM" in lines
    assert "Điều 1. Một" not in lines


def test_khong_thay_moc_bat_dau_thi_canh_bao():
    text, report = normalize("Một đoạn văn không có cấu trúc gì.\n")
    assert any("không tìm thấy" in w for w in report.warnings)
    assert "Một đoạn văn" in text


# --- báo cáo --------------------------------------------------------------
def test_report_dem_dung_cau_truc():
    raw = body(
        """
        Chương I
        QUY ĐỊNH CHUNG
        Điều 1. Phạm vi
        1. Khoản một.
        2. Khoản hai.
        Điều 2. Giải thích
        1. Khoản một.
        """
    )
    _, report = normalize(raw)
    assert report.n_chapters == 1
    assert report.n_articles == 2
    assert report.n_clauses == 3
    assert isinstance(report, NormalizeReport)


def test_report_canh_bao_so_dieu_nhay_coc():
    raw = "Điều 1. Một\nNội dung.\nĐiều 7. Bảy\nNội dung.\n"
    _, report = normalize(raw)
    assert any("không liên tiếp" in w for w in report.warnings)


# --- outline --------------------------------------------------------------
def test_outline_canh_bao_dieu_khong_khoan_va_qua_dai():
    raw = body(
        """
        Chương I
        QUY ĐỊNH CHUNG
        Điều 1. Không có khoản
        Chỉ một đoạn văn.
        Điều 2. Rất dài
        """
    ) + "\n1. " + ("từ ngữ dài " * 1400) + "\n"
    entries = outline_tree(raw)
    assert entries[0].n_clauses == 0
    assert "0 khoản" in entries[0].warnings
    assert any("wrap" in w for w in entries[1].warnings)
    assert "Chương I — QUY ĐỊNH CHUNG" in outline(raw)


def test_outline_canh_bao_khoan_nhay_coc():
    raw = "Điều 1. Thử\n1. Một.\n3. Ba.\n"
    assert any("nhảy cóc" in w for w in outline_tree(raw)[0].warnings)


# --- chọn Điều ------------------------------------------------------------
def test_select_articles_giu_tieu_de_chuong():
    raw = body(
        """
        Chương I
        QUY ĐỊNH CHUNG
        Điều 1. Một
        Nội dung một.
        Điều 2. Hai
        Nội dung hai.
        Chương II
        ĐIỀU KHOẢN THI HÀNH
        Điều 3. Ba
        Nội dung ba.
        """
    )
    result = select_articles(raw, {"articles": [2, 3]})

    assert "Chương I" in result
    assert "QUY ĐỊNH CHUNG" in result
    assert "Chương II" in result
    assert "ĐIỀU KHOẢN THI HÀNH" in result
    assert "Điều 2. Hai" in result and "Điều 3. Ba" in result
    assert "Điều 1. Một" not in result and "Nội dung một." not in result


def test_select_articles_bo_chuong_khong_con_dieu_nao():
    raw = body(
        """
        Chương I
        QUY ĐỊNH CHUNG
        Điều 1. Một
        Nội dung một.
        Chương II
        THI HÀNH
        Điều 2. Hai
        Nội dung hai.
        """
    )
    lines = select_articles(raw, {"articles": [2]}).split("\n")
    assert "Chương I" not in lines
    assert "QUY ĐỊNH CHUNG" not in lines
    assert "Chương II" in lines


def test_select_articles_nhan_range_va_giu_so_goc():
    raw = "Điều 5. Năm\nNội dung năm.\nĐiều 6. Sáu\nNội dung sáu.\nĐiều 9. Chín\nNội dung chín.\n"
    result = select_articles(raw, {"articles": range(5, 7)})
    assert result.startswith("Điều 5.")
    assert "Điều 6. Sáu" in result
    assert "Điều 9." not in result


def test_select_articles_nhan_iterable_tran():
    raw = "Điều 1. Một\nNội dung.\nĐiều 2. Hai\nNội dung.\n"
    assert "Điều 2. Hai" in select_articles(raw, [2])


def test_word_count():
    assert word_count("một hai ba  bốn\nnăm") == 5


# --- nhận dạng định dạng --------------------------------------------------
def test_detect_format_theo_magic_bytes(tmp_path):
    cases = {
        "a.bin": (b"%PDF-1.7 rest", "pdf"),
        "b.bin": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "doc"),
        "c.bin": (b"PK\x03\x04 rest", "docx"),
    }
    for name, (payload, expected) in cases.items():
        path = tmp_path / name
        path.write_bytes(payload)
        assert detect_format(path) == expected


def test_detect_format_magic_thang_duoi_file(tmp_path):
    """File .doc cũ đặt tên .docx vẫn phải ra 'doc'."""
    path = tmp_path / "nhamten.docx"
    path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    assert detect_format(path) == "doc"


def test_detect_format_khong_biet(tmp_path):
    path = tmp_path / "la.xyz"
    path.write_bytes(b"noi dung tho")
    assert detect_format(path) == "unknown"


# --- kiểm chất lượng trích ------------------------------------------------
def test_check_extraction_quality_qua_ngan():
    with pytest.raises(ExtractionError, match="bản scan"):
        check_extraction_quality("ngắn quá", "x.pdf")


def test_check_extraction_quality_mat_dau():
    with pytest.raises(ExtractionError, match="ky tu co dau|có dấu"):
        check_extraction_quality("a" * 2000, "x.pdf")


def test_check_extraction_quality_van_ban_tieng_viet_thi_qua():
    check_extraction_quality("Nghị định này quy định về bảo vệ dữ liệu cá nhân. " * 40, "x.pdf")


def test_diacritic_ratio():
    assert diacritic_ratio("") == 0.0
    assert diacritic_ratio("abcdef") == 0.0
    assert diacritic_ratio("đđđđ") == 1.0


# --- frontmatter ----------------------------------------------------------
def test_render_frontmatter_khop_dinh_dang_ingest():
    rendered = render_frontmatter(
        {
            "doc_id": "luat_91_2025",
            "title": "Luật Bảo vệ dữ liệu cá nhân số 91/2025/QH15",
            "doc_type": "luat",
            "issued_date": "2025-06-26",
            "effective_from": "2026-01-01",
            "effective_to": None,
            "status": "active",
        }
    )
    assert rendered.startswith("---\n") and rendered.rstrip().endswith("---")
    assert "doc_id: luat_91_2025" in rendered
    assert 'title: "Luật Bảo vệ dữ liệu cá nhân số 91/2025/QH15"' in rendered
    assert 'issued_date: "2025-06-26"' in rendered
    assert "effective_to: null" in rendered
    assert "status: active" in rendered


def test_render_frontmatter_thieu_key_thi_raise():
    with pytest.raises(ValueError, match="thiếu key bắt buộc"):
        render_frontmatter({"doc_id": "a", "title": "b"})


def test_render_frontmatter_status_sai_thi_raise():
    meta = {
        "doc_id": "a",
        "title": "b c",
        "doc_type": "luat",
        "issued_date": "2025-01-01",
        "effective_from": "2025-01-01",
        "effective_to": None,
        "status": "draft",
    }
    with pytest.raises(ValueError, match="status="):
        render_frontmatter(meta)


def test_corpus_audit_bao_lech_so_dieu(tmp_path):
    """KEEP nói giữ 3 Điều mà parser chỉ nhận 1 -> phải báo đỏ, không im lặng."""
    from src.chunk import chunk_corpus
    from src.ingest import load_corpus
    from tools.normalize_raw import corpus_audit

    meta = {
        "doc_id": "nd_13_2023",
        "title": "Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân",
        "doc_type": "nghi_dinh",
        "issued_date": "2023-04-17",
        "effective_from": "2023-07-01",
        "effective_to": None,
        "status": "active",
    }
    text, _ = normalize("Điều 1. Phạm vi\n1. Nội dung của khoản một.\n")
    write_corpus_file(tmp_path, meta, text)

    documents = load_corpus(tmp_path)
    chunks = chunk_corpus(documents)

    khop = corpus_audit(documents, chunks, {"nd_13_2023": {"articles": [1]}})
    assert "LỆCH" not in khop
    assert "phủ body" in khop

    lech = corpus_audit(documents, chunks, {"nd_13_2023": {"articles": [1, 2, 3]}})
    assert "LỆCH nd_13_2023" in lech
    assert "thiếu [2, 3]" in lech


def test_write_corpus_file_ingest_doc_lai_duoc(tmp_path):
    """Khép vòng: file ghi ra phải được src/ingest.py đọc lại đúng."""
    from src.ingest import load_corpus

    meta = {
        "doc_id": "nd_13_2023",
        "title": "Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân",
        "doc_type": "nghi_dinh",
        "issued_date": "2023-04-17",
        "effective_from": "2023-07-01",
        "effective_to": "2026-01-01",
        "status": "expired",
    }
    text, _ = normalize(
        body(
            """
            Chương I
            QUY ĐỊNH CHUNG
            Điều 1. Phạm vi điều chỉnh
            1. Nghị định này quy định về bảo vệ dữ liệu cá nhân.
            2. Nghị định áp dụng cho cơ quan, tổ chức có liên quan.
            """
        )
    )
    write_corpus_file(tmp_path, meta, text)

    document = load_corpus(tmp_path)[0]
    assert document.meta.doc_id == "nd_13_2023"
    assert document.meta.status == "expired"
    assert document.meta.effective_to == "2026-01-01"
    assert [a.article_no for a in document.articles] == [1]
    assert [c.clause_no for c in document.articles[0].clauses] == [1, 2]
    assert document.articles[0].chapter == "Chương I. QUY ĐỊNH CHUNG"
