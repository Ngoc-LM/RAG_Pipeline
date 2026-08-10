"""Test số học khoảng — chỗ dễ sai nhất là các ca chồng lấn và lồng nhau."""

from __future__ import annotations

import pytest

from src.intervals import coverage, intersection_length, interval_union


def test_union_rong():
    assert interval_union([]) == []


def test_union_bo_khoang_rong_va_nghich_dao():
    assert interval_union([(5, 5), (9, 3), (1, 4)]) == [(1, 4)]


def test_union_khong_giao_giu_nguyen_va_sap_xep():
    assert interval_union([(10, 20), (0, 5)]) == [(0, 5), (10, 20)]


def test_union_chong_lan_mot_phan():
    assert interval_union([(0, 10), (5, 15)]) == [(0, 15)]


def test_union_ke_nhau_thi_gop():
    assert interval_union([(0, 5), (5, 9)]) == [(0, 9)]


def test_union_long_nhau_hoan_toan():
    assert interval_union([(0, 100), (10, 20), (30, 40)]) == [(0, 100)]


def test_union_trung_lap_hoan_toan():
    assert interval_union([(3, 8), (3, 8), (3, 8)]) == [(3, 8)]


def test_union_chuoi_chong_lan_day_chuyen():
    assert interval_union([(0, 4), (3, 7), (6, 10), (20, 25)]) == [(0, 10), (20, 25)]


def test_chong_lan_khong_bi_dem_hai_lan():
    """Hai chunk chồng lấn phủ [0,15); tổng độ dài text là 20 nhưng phủ chỉ 15."""
    union = interval_union([(0, 10), (5, 15)])
    assert intersection_length(union, (0, 15)) == 15
    assert coverage(union, (0, 15)) == 1.0


def test_giao_mot_phan():
    union = interval_union([(0, 10)])
    assert intersection_length(union, (5, 25)) == 5
    assert coverage(union, (5, 25)) == 0.25


def test_giao_rong_khi_tach_roi():
    union = interval_union([(0, 10)])
    assert intersection_length(union, (50, 60)) == 0
    assert coverage(union, (50, 60)) == 0.0


def test_coverage_span_rong_tra_ve_khong():
    assert coverage([(0, 100)], (7, 7)) == 0.0


def test_coverage_khong_bao_gio_vuot_mot():
    union = interval_union([(0, 50), (25, 75), (40, 60)])
    assert coverage(union, (10, 40)) == pytest.approx(1.0)


def test_coverage_cong_don_tu_nhieu_manh_roi_nhau():
    """Span được phủ bởi hai chunk rời nhau, mỗi chunk phủ một nửa."""
    union = interval_union([(0, 10), (20, 30)])
    assert coverage(union, (0, 30)) == pytest.approx(20 / 30)
